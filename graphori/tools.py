import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def edge2mat2(link, num_joint, num_part):
    A = np.zeros((num_part, num_joint))
    for i, j in link:
        A[i, j] = 1
    return A

def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    # print(In)
    Out = normalize_digraph(edge2mat(outward, num_node))
    # print(Out)
    A = np.stack((I, In, Out))
    return A


def get_graph(num_joint, num_part, prior_link):
    A = edge2mat2(prior_link, num_joint, num_part)
    return A