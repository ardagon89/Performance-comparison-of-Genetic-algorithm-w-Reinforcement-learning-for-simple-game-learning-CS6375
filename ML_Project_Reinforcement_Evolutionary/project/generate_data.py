import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def initialize_matrix(probability_of_sparseness, N):
    matrix = np.zeros((N, N))
    edges = []
    for i in range(N):
        for j in range(i):
            if random.random() > probability_of_sparseness:
                matrix[i][j] = 1
                matrix[j][i] = matrix[i][j]
                edges.append((i,j))
    return matrix,edges


def visualize_graph(matrix):
    matrix = np.asmatrix(matrix)
    print(matrix)
    rows, cols = np.where(np.asmatrix(matrix) > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=800, with_labels=True)
    plt.show()