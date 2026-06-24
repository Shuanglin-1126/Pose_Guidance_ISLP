import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
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
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_adjacency_matrix(num_nodes, edges):
    A = np.eye(num_nodes, dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward, sym_in, sym_out):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A5 = edge2mat(sym_in, num_node)
    A6 = edge2mat(sym_out, num_node)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A5 = normalize_digraph(A5)
    A6 = normalize_digraph(A6)
    A = np.stack((I, A1, A2, A3, A4, A5, A6))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

class Graph:
    def __init__(self, mode='spatial'):
        self.num_node = 67
        self.inward = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 11),
                            (11, 12), (12, 13), (13, 14), (14, 15),
                            (11, 16), (16, 17), (17, 18), (18, 19),
                            (11, 20), (20, 21), (21, 22), (22, 23),
                            (11, 24), (24, 25), (25, 26), (26, 27),
                            (11, 28), (28, 29), (29, 30), (30, 31),
                            (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 32),
                            (32, 33), (33, 34), (34, 35), (35, 36),
                            (32, 37), (37, 38), (38, 39), (39, 40),
                            (32, 41), (41, 42), (42, 43), (43, 44),
                            (32, 45), (45, 46), (46, 47), (47, 48),
                            (32, 49), (49, 50), (50, 51), (51, 52),

                       (0, 53), (53, 54), (54, 55), (55, 56), (53, 57),
                       (58, 53), (56, 59), (59, 60), (61, 62), (62, 56),
                       (63, 64), (65, 66), (57, 55), (58, 55),
                    ]

        # self.num_node = 53
        # self.inward = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 11),
        #                (11, 12), (12, 13), (13, 14), (14, 15),
        #                (11, 16), (16, 17), (17, 18), (18, 19),
        #                (11, 20), (20, 21), (21, 22), (22, 23),
        #                (11, 24), (24, 25), (25, 26), (26, 27),
        #                (11, 28), (28, 29), (29, 30), (30, 31),
        #                (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 32),
        #                (32, 33), (33, 34), (34, 35), (35, 36),
        #                (32, 37), (37, 38), (38, 39), (39, 40),
        #                (32, 41), (41, 42), (42, 43), (43, 44),
        #                (32, 45), (45, 46), (46, 47), (47, 48),
        #                (32, 49), (49, 50), (50, 51), (51, 52),
        #                ]

        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.A = self.get_adjacency_matrix(mode)

    def get_adjacency_matrix(self, mode='spatial'):
        if mode is None:
            return self.A
        if mode == 'spatial':
            # A = get_multiscale_spatial_graph(self.num_node, self.self_link, self.inward, self.outward, self.sym_in, self.sym_out)
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A