import numpy as np
import sys

sys.path.extend(['../'])
from graphori import tools
import networkx as nx

# Joint index:
# {0,  "Lear"}
# {1,  "Rear"},
# {2,  "snout"},
# {3,  "Centroid"},
# {4,  "Llateral"},
# {5,  "Rlateral"},
# {6,  "Tail-base"},
# {7,  "Tail-end"},



# Edge format: (origin, neighbor)
num_node = 7
self_link = [(i, i) for i in range(num_node)]
# inward = [(6, 4), (4, 0), (0, 2), (6, 5), (5, 1), (1, 2), (7, 6), (6, 3),
#           (3, 2), (0, 1), (4, 5)]

inward = [(6, 4), (4, 0), (0, 2), (6, 5), (5, 1), (1, 2), (6, 3),
          (3, 2), (0, 1), (4, 5)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


'''
CMU joint2part A (3, 8)

'''
njoint = 7
npart = 3
edge_joint2part = [(0, 0), (0,1),(0,2) ,(1,3), (1,4), (1,5), (2,6)]   #

# 3 point skeleton
num_node_part = 3
self_link_part = [(i, i) for i in range(num_node_part)]
inward_part = [(2,1),(1,0)]

outward_part = [(j, i) for (i, j) in inward_part]


# inter-skeleton

edge_skeleton2skeleton_s1 = [(0, 0), (1,1),(2,2) ,(3,3), (4,4), (5,5), (6,6)]   #

edge_skeleton2skeleton_s2 = [(0, 0), (1,1),(2,2)]
class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_part = self.get_adjacency_matrix(labeling_mode)
        self.A_joint2part = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.num_node_part = num_node_part
        self.self_link_part = self_link_part
        self.inward_part = inward_part
        self.outward_part = outward_part



    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
            A_part = tools.get_spatial_graph(num_node_part, self_link_part, inward_part, outward_part)
            A_joint2part = tools.get_graph( njoint, npart, edge_joint2part)
            A_keleton2skeleton_s1 = tools.get_graph(njoint, njoint, edge_skeleton2skeleton_s1)
            A_keleton2skeleton_s2 = tools.get_graph(npart, npart, edge_skeleton2skeleton_s2)

        else:
            raise ValueError()
        return A, A_part,  A_joint2part, A_keleton2skeleton_s1, A_keleton2skeleton_s2


if __name__ == '__main__':
    A, A_joint2part = Graph('spatial').get_adjacency_matrix()
    print(A.shape)   #(3,8,8)
    print(A_joint2part.shape)
    print(inward)
    print(outward)
    print(neighbor)
