import networkx as nx
import pandas as pd
import spotipy
import numpy as np
from math import pi
import community
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
import matplotlib.pyplot as plt
def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    nodes =  arg[0].nodes()
    for graf in arg[1:]:
        llista = graf.nodes()
        nodes = [x for x in llista if x in nodes]  
    
    return len(nodes)
    
def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    graus = {}
    for node in g:
        grau = g.degree(node)
        if grau not in graus:
            graus[grau] = 1
        else:
            graus[grau] += 1
    
    return dict(sorted(graus.items()))

def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes with the specified centrality.
    """
    centralitats = {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality
    }
    
    if metric not in centralitats:
        raise ValueError("Centralitat no trobada")
    
    centrality = centralitats[metric](g)
    ordenat = [key for key,value in sorted(centrality.items(), key=lambda x:x[1],reverse=True)[:num_nodes]]
    return ordenat

def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    cliques = list(nx.find_cliques(g))
    min_cliques = []
    for clique in cliques:
        if len(clique) >= min_size_clique:
            subgraph = g.subgraph(clique)
            min_cliques.append(subgraph)

    nodes = {i for x in min_cliques for i in x.nodes()}
    return (min_cliques, list(nodes))

def detect_communities(g: nx.Graph, method: str, **kwargs) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be 'girvan-newman' or 'louvain'.
    # :param kwargs: additional parameters for community detection.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    if method == 'girvan-newman':
        communities_generator = nx.algorithms.community.girvan_newman(g)
        communities = next(communities_generator)
        modularity = nx.algorithms.community.modularity(g, communities)

    elif method == 'louvain':
        partition = community.best_partition(g, **kwargs)
        communities = []
        for com in set(partition.values()):
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            communities.append(list_nodes)
        modularity = community.modularity(partition, g)
    else:
        raise ValueError("Invalid method. Choose 'girvan-newman' or 'louvain'.")

    return (communities, modularity)


if __name__ == '__main__':
    # obtenció del directori de treball
    directori = os.path.dirname(os.path.abspath(__file__))

    # lectura del gB
    arxiu = os.path.join(directori, "gB.graphml")
    lectura_gb = nx.read_graphml(arxiu)
    gB = lectura_gb.to_directed()

    # lectura del gD
    arxiu = os.path.join(directori, "gD.graphml")
    lectura_gd = nx.read_graphml(arxiu)
    gD = lectura_gd.to_directed()

    # obtenció del gBp 
    arxiu = os.path.join(directori, "gBp.graphml")
    lectura_gbp = nx.read_graphml(arxiu)
    gBp = lectura_gbp.to_undirected()

    # obtenció del gDp
    arxiu = os.path.join(directori, "gDp.graphml")
    lectura_gdp = nx.read_graphml(arxiu)
    gDp = lectura_gdp.to_undirected()

    # obtenció del gBw 
    arxiu = os.path.join(directori, "gBw.graphml")
    lectura_gbw = nx.read_graphml(arxiu)
    gBw = lectura_gbw.to_undirected()

    # obtenció del gDw
    arxiu = os.path.join(directori, "gDw.graphml")
    lectura_gdw = nx.read_graphml(arxiu)
    gDw = lectura_gdw.to_undirected()

    print("\n------ EXERCICI 1 ---------------------------------------\n")
    # # amb gB
    x = num_common_nodes(gB, gD)
    print("Numero de nodes comuns entre gB i gD :", x)
    
    x = num_common_nodes(gB, gBp)
    print("Numero de nodes comuns entre gB i gBp :", x)
    
    x = num_common_nodes(gB, gDp)
    print("Numero de nodes comuns entre gB i gDp :", x)
    
    x = num_common_nodes(gB, gBw)
    print("Numero de nodes comuns entre gB i gBw :", x)
    
    x = num_common_nodes(gB, gDw)
    print("Numero de nodes comuns entre gB i gDw :", x)
    
    # # amb gD
    x = num_common_nodes(gD, gBp)
    print("\nNumero de nodes comuns entre gD i gBp :", x)
    
    x = num_common_nodes(gD, gDp)
    print("Numero de nodes comuns entre gD i gDp :", x)
    
    x = num_common_nodes(gD, gBw)
    print("Numero de nodes comuns entre gD i gBw :", x)
    
    x = num_common_nodes(gD, gDw)
    print("Numero de nodes comuns entre gD i gDw :", x)
    
    # # amb gBp
    p = num_common_nodes(gBp, gDp)
    print("\nNumero de nodes comuns entre gBp i gDp :", p)
    
    p = num_common_nodes(gBp, gBw)
    print("Numero de nodes comuns entre gBp i gBw :", p)
    
    p = num_common_nodes(gBp, gDw)
    print("Numero de nodes comuns entre gBp i gDw :", p)
    
    # # amb gDp
    p = num_common_nodes(gDp, gBw)
    print("\nNumero de nodes comuns entre gDp i gBw :", p)
    
    p = num_common_nodes(gDp, gDw)
    print("Numero de nodes comuns entre gDp i gDw :", p)
    
    # # amb gBw
    g = num_common_nodes(gBw, gDw)
    print("\nNumero de nodes comuns entre gBw i gDw :", g)
    
    w = num_common_nodes(gB, gD, gBp, gDp, gBw, gDw)
    print("\nNumero de nodes comuns entre tots els noddes :", w)

    print("\n------ EXERCICI 2 ---------------------------------------\n")
    set_degree = get_k_most_central(gBp, "degree", 25)
    g_degree = gBp.subgraph(set_degree)
    set_btw = get_k_most_central(gBp, "betweenness", 25)
    g_btw = gBp.subgraph(set_btw)
    com = num_common_nodes(g_degree,g_btw)
    print("Numero de nodes comuns entre degree i betweenness:", com)

    print("\n------ EXERCICI 3 ---------------------------------------\n")
    min_size_clique_b = 7
    c_gbp = find_cliques(gBp, min_size_clique_b) # 7
    min_size_clique_d = 11
    c_gdp = find_cliques(gDp, min_size_clique_d) # 11
    print(f"En el gBp, hi ha {len(c_gbp[0])} cliques, amb min_size_clique={min_size_clique_b} i hi ha {len(c_gbp[1])} nodes.")
    print(f"En el gDp, hi ha {len(c_gdp[0])} cliques, amb min_size_clique={min_size_clique_d} i hi ha {len(c_gdp[1])} nodes.")

    print("\n------ EXERCICI 4 ---------------------------------------\n")
    maxim = 0
    maxs = []
    for c in [c_gbp[0], c_gdp[0]]:
        for clique in c:
            if clique.order() > maxim:
                maxim = clique.order()
                maxs = clique
            elif clique.order() == maxim:
                maxs = [maxs]
                maxs.append(clique)
    print(maxim)
    print(maxs)
    nodes = []
    for i in maxs[1].nodes():
        nodes.append(maxs[1].nodes[i]['name'])
    print(nodes)

        
    


