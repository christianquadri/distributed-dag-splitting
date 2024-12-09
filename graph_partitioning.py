import copy
import itertools
from collections import deque
from pprint import pprint
from typing import Iterable

import pandas as pd
import numpy as np
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

# Nodes and link attributes names
SRV_CPU_REQ = 'cpu'
SRV_LINK_REQ = 'tx'
PHY_NODE_CAP = 'capacity'
PHY_NODE_AGG_CAP = 'agg_capacity'
PHY_LINK_DATARATE = 'datarate'


class ServiceRequest:
    def __init__(self,
                 service_dag: nx.Graph,
                 request_node: int):
        self._req_node = request_node
        self._srv_dag = service_dag

    @property
    def request_node(self) -> int:
        return self._req_node

    @property
    def service_dag(self) -> nx.Graph:
        return self._srv_dag

    def sub_dag(self, dag_nodes)->nx.Graph:
        return self._srv_dag.subgraph(dag_nodes)





class DagSplitSolution:
    def __init__(self, sol_x_vars: gp.tupledict, decision_node: int):
        # x_{iu} variables mean service i deployed on node u
        self._local_computed_srv = []
        self._decision_node = decision_node
        self._offloaded_srv = {}
        for (srv, phy_node), value in sol_x_vars.items():
            if value.x > 0:
                if phy_node == decision_node: self._local_computed_srv.append(srv)
                else: self._offloaded_srv.setdefault(phy_node,[]).append(srv)

    def get_local_assignment(self)-> tuple[int, list]:
        return self._decision_node, copy.deepcopy(self._local_computed_srv)

    def get_offloaded_dag_components(self)-> dict[int,list]:
        return copy.deepcopy(self._offloaded_srv)

    def get_selected_neighbor_nodes(self)-> list[int]:
        return list(self._offloaded_srv.keys())

    def __repr__(self):
        return f'Local ({self._decision_node}: {self._local_computed_srv}\n{self._offloaded_srv}'


class FinalDagDeployment:
    def __init__(self, srv_req: ServiceRequest):
        self._srv_req = srv_req
        self.srv_deployment = {}  # empty at the beginning

    def update_deployment(self, dag_split_sol: DagSplitSolution):
        phy_node, srv_list = dag_split_sol.get_local_assignment()
        self.srv_deployment.update({s:phy_node for s in srv_list})

    def get_deployment(self):
        return copy.deepcopy(self.srv_deployment) # copy for sanity!


def solve_dag_split_assignment(local_phy_net: nx.Graph,
                               srv_dag: nx.Graph,
                               decision_node: int) -> DagSplitSolution:
    # utilities local functions
    def srv_node_cpu(s): return srv_dag.nodes()[s][SRV_CPU_REQ]
    def srv_link_tx(i, j): return srv_dag.edges()[i, j][SRV_LINK_REQ]
    def srv_links(): return list(srv_dag.edges())
    def phy_links(): return list(local_phy_net.edges())
    def phy_node_cap(u): return local_phy_net.nodes()[u][PHY_NODE_CAP if u == decision_node else PHY_NODE_AGG_CAP ]
    def phy_link_cap(u, v): return local_phy_net.edges()[u, v][PHY_LINK_DATARATE]
    def phy_link_only_constr(u, v): return 1 if local_phy_net.has_edge(u, v) or u == v else 0
    def full_mesh(): return list(itertools.product(local_phy_net.nodes(), repeat=2))
    def phy_node_agg_cap(u): return local_phy_net.nodes()[u][PHY_NODE_AGG_CAP]

    model = gp.Model(name="split_assign")

    x = model.addVars(srv_dag.nodes(), local_phy_net.nodes(), vtype=GRB.BINARY, name="x")
    e = model.addVars(local_phy_net.edges(), vtype=GRB.CONTINUOUS, name='e')
    n = model.addVars(local_phy_net.nodes(), vtype=GRB.CONTINUOUS, name='u')

    # service assignemed constr
    srv_assignement = model.addConstrs(((sum(x[i, u] for u in local_phy_net.nodes()) == 1) for i in srv_dag.nodes()), name='assign')

    # node capacity constr
    node_cap_constr = model.addConstrs(
        (sum(srv_node_cpu(i) * x[i, u] for i in srv_dag.nodes()) <= phy_node_cap(u) for u in local_phy_net.nodes()),
        name='node_cap')
    # link capacity constr
    link_cap_constr = model.addConstrs(
        (gp.quicksum((srv_link_tx(i, j) * x[i, u] * x[j, v]) for i, j in srv_links()) <= phy_link_cap(u, v) for u, v in
         phy_links()), name='link_cap')

    # avoid multi-hop communication at assignment phase
    phy_link_only = model.addConstrs(
        (x[i, u] * x[j, v] <= phy_link_only_constr(u, v) for i, j in srv_links() for u, v in full_mesh()),
        name='phy_link_only')

    model.setObjective(gp.quicksum(gp.quicksum(x[i, u] for i in srv_dag.nodes()) * phy_node_agg_cap(u) for u in local_phy_net.nodes()), GRB.MINIMIZE)

    model.optimize()

    # process model solution
    return DagSplitSolution(x, decision_node)






def deploy_service_request(phy_net: nx.DiGraph, srv_req: ServiceRequest) -> FinalDagDeployment:

    def get_local_phy_net(phy_node:int,
                          forbidden_neighbors: Iterable[int]) -> nx.Graph:
        allowed_neighbors = set(phy_net.neighbors(phy_node)).difference(forbidden_neighbors)
        return phy_net.subgraph(allowed_neighbors)

    # Breadth-first search
    final_deployment = FinalDagDeployment(srv_req=srv_req)
    request_queue = deque([srv_req])
    used_nodes = set()
    while len(request_queue) > 0:
        request = request_queue.popleft()
        used_nodes.add(request.request_node)
        local_phy_net = get_local_phy_net(phy_node=request.request_node,
                                      forbidden_neighbors=used_nodes)

        solution = solve_dag_split_assignment(local_phy_net=local_phy_net,
                                              srv_dag=request.service_dag,
                                              decision_node=request.request_node)

        print(solution)

        # update the final deployment accounting for the service/s that has been selected
        # to be processed on the local node
        final_deployment.update_deployment(solution)  # this call has side-effect
        # update used nodes (simulating control message exchange)
        used_nodes.update(solution.get_selected_neighbor_nodes())
        # "send" sub-dag to other nodes
        for n, dag_nodes in solution.get_offloaded_dag_components().items():
            offloaded_request = ServiceRequest(service_dag=request.sub_dag(dag_nodes),
                                               request_node=n)
            request_queue.append(offloaded_request)


    return final_deployment




if __name__ == '__main__':
    physical_net = nx.DiGraph()
    physical_net.add_node(1,capacity=300, agg_capacity=300)
    physical_net.add_node(2, capacity=100, agg_capacity=200)
    physical_net.add_node(3, capacity=200, agg_capacity=240)
    physical_net.add_node(4, capacity=200, agg_capacity=200)
    physical_net.add_node(5, capacity=200, agg_capacity=200)
    physical_net.add_node(6, capacity=200, agg_capacity=200)
    physical_net.add_edge(1,2, datarate=100)
    physical_net.add_edge(2, 1, datarate=100)

    physical_net.add_edge(2, 5, datarate=100)
    physical_net.add_edge(5, 2, datarate=100)

    physical_net.add_edge(1, 5, datarate=100)
    physical_net.add_edge(5, 1, datarate=100)

    physical_net.add_edge(2, 3, datarate=20)
    physical_net.add_edge(3, 2, datarate=20)

    physical_net.add_edge(5, 6, datarate=20)
    physical_net.add_edge(6, 5, datarate=20)

    physical_net.add_edge(3, 6, datarate=20)
    physical_net.add_edge(6, 3, datarate=20)

    physical_net.add_edge(3, 4, datarate=20)
    physical_net.add_edge(4, 3, datarate=20)


    service = nx.DiGraph()
    service.add_node('A',cpu=100)
    service.add_node('B', cpu=100)
    service.add_node('C', cpu=50)
    service.add_node('D', cpu=40)
    service.add_node('E', cpu=80)

    service.add_edge('A','B', tx=5)
    service.add_edge('A', 'C', tx=10)
    service.add_edge('B', 'E', tx=10)
    service.add_edge('C', 'D', tx=20)
    #service.add_edge('D', 'E', tx=5)

    deploy_service_request(physical_net, ServiceRequest(service,1))