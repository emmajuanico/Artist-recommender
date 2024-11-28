import networkx as nx
import pandas as pd
import spotipy
import numpy as np
from math import pi

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
import matplotlib.pyplot as plt

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    unidireccional = [(u, v) for u, v in g.edges() if not g.has_edge(v, u)]
    g = g.to_undirected()
    g.remove_edges_from(unidireccional)
    prune_low_degree_nodes(g,1,"e.graphml")
    nx.write_graphml(g, out_filename)
    return g

def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    graus_menors = [node for node, grau in g.degree() if grau < min_degree]
    g.remove_nodes_from(graus_menors)
    
    nodes_sols = [node for node, degree in dict(g.degree()).items() if degree == 0]
    g.remove_nodes_from(nodes_sols)
    
    nx.write_graphml(g, out_filename)
    return g

def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    g = g.copy()
    if type(min_weight) == type(min_percentile):
        raise ValueError("Falten dades o sobren")
    
    if min_weight:
        arestes_eliminar = [(u, v) for u, v, weight in g.edges(data='weight') if weight < min_weight]
    elif min_percentile:
        pesos = [p for a, b, p in g.edges(data='weight')]
        cutoff = np.percentile(pesos, min_percentile)
        print("cutoff:",cutoff)

        arestes_eliminar = [(u, v) for u, v, p in g.edges(data='weight') if p < cutoff]
    # else:
    #     raise ValueError("Les dades no són correctes")
    
    g.remove_edges_from(arestes_eliminar)
    nx.write_graphml(g, out_filename)
    return g


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    artist_junts = tracks_df.groupby(['artist_id']).mean()
    mitjana_artistes = artist_junts.drop(['song_duration', 'song_popularity'], axis=1)
    
    # mitjana_artistes.to_csv('mitjanes.csv', index=False) 
    return mitjana_artistes

def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    if similarity == "cosine":
        similarity_matrix = cosine_similarity(artist_audio_features_df)
    elif similarity == "euclidean":
        similarity_matrix = 1 / (1 + euclidean_distances(artist_audio_features_df))
    else:
        raise ValueError("Similarity no vàlida:'cosine' o 'euclidean'.")

    num_artists = len(artist_audio_features_df)
    similarity_graph = nx.Graph()
    for i in range(num_artists):
        for j in range(i + 1, num_artists):
            similarity_graph.add_edge(i, j, weight=similarity_matrix[i][j])

    nx.write_graphml(similarity_graph, out_filename)
    return similarity_graph

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    nodes =  arg[0].nodes()
    for graf in arg[1:]:
        # for node in graf.nodes():
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
    'eigenvector': nx.eigenvector_centrality}
    
    if metric not in centralitats:
        raise ValueError("Centralitat no trobada")
    
    centrality = centralitats[metric](g)
    ordenat = [key for key,value in sorted(centrality.items(), key=lambda x:x[1],reverse=True)[:num_nodes]]
    return ordenat



if __name__ == "__main__":
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
    gBp = retrieve_bidirectional_edges(gB, arxiu)
    # gBp = prune_low_degree_nodes(gBp,1,arxiu)

    # obtenció del gDp
    arxiu = os.path.join(directori, "gDp.graphml")
    gDp = retrieve_bidirectional_edges(gD, arxiu)
    # gDp = prune_low_degree_nodes(gDp,1,arxiu)

    # codi prèvi a l'obtenció del gBw i gDw
    # obtenció del dataframe D
    arxiu = os.path.join(directori, "song.csv")
    D = pd.read_csv(arxiu)
    similarity = "euclidean"#"cosine"#"euclidean"

    mitjanes = compute_mean_audio_features(D)
    arxiu = os.path.join(directori, "graf_similarity.graphml")
    graf_similarity = create_similarity_graph(mitjanes, similarity,arxiu)
    
    weight = [d['weight'] for u, v, d in graf_similarity.edges(data=True)]
    max_weight = max(weight)
    min_weight = min(weight)
    print("\nvalor màxim:",max_weight)
    print("valor mínim:",min_weight)

    arxiu = os.path.join(directori, "gBw.graphml")
    get_gBw = (1-(gBp.size()/graf_similarity.size()))*100
    gBw = prune_low_weight_edges(graf_similarity,min_percentile=get_gBw,out_filename=arxiu)
    # gBw = prune_low_degree_nodes(gBw,1,arxiu)

    arxiu = os.path.join(directori, "gDw.graphml")
    get_gDw = (1-(gDp.size()/graf_similarity.size()))*100
    gDw = prune_low_weight_edges(graf_similarity,min_percentile=get_gDw,out_filename=arxiu)
    # gDw = prune_low_degree_nodes(gDw,1,arxiu)

    # ordre dels 4 grafs obtinguts
    print("\n------ EXERCICI 1 ---------------------------------------\n")
    # print("\nOrdre del gB:",gB.order())
    # print("Ordre del gD:",gD.order())
    print("Ordre del gBp:",gBp.order())
    print("Ordre del gDp:",gDp.order())
    # print("\nOrdre del similarity:",graf_similarity.order())
    print(f"Ordre del gBw, obtingut amb min_percentile={get_gBw}:",gBw.order())
    print(f"Ordre del gDw, obtingut amb min_percentile={get_gDw}:",gDw.order())

    # mida dels 4 grafs obtinguts
    # print("\nMida del gB:",gB.size())
    # print("Mida del gD:",gD.size())
    print("\nMida del gBp:",gBp.size())
    print("Mida del gDp:",gDp.size())
    # print("\nMida del similarity:",graf_similarity.size())
    print(f"Mida del gBw, obtingut amb min_percentile={get_gBw}:",gBw.size())
    print(f"Mida del gDw, obtingut amb min_percentile={get_gDw}:",gDw.size())

    # components fortament i debilment connexes gB i gD
    print("\n------ EXERCICI 3 ---------------------------------------\n")
    c_fortament = list(nx.strongly_connected_components(gB))
    num_c_fortament = len(c_fortament)
    print("Components fortament connexes del gB:", num_c_fortament)
    
    c_debilment = list(nx.weakly_connected_components(gB))
    num_c_debilment = len(c_debilment)
    print("Components debilment connexes del gB:", num_c_debilment)

    c_fortament = list(nx.strongly_connected_components(gD))
    num_c_fortament = len(c_fortament)
    print("\nComponents fortament connexes del gD:", num_c_fortament)
    
    c_debilment = list(nx.weakly_connected_components(gD))
    num_c_debilment = len(c_debilment)
    print("Components debilment connexes del gD:", num_c_debilment)   

    # components connexes gBp
    c_bp = list(nx.connected_components(gBp))
    num_c_bp = len(c_bp)
    
    # components connexes gDp
    c_dp = list(nx.connected_components(gDp))
    num_c_dp = len(c_dp)
    print("\n------ EXERCICI 4 ---------------------------------------\n")
    print("Components connexes de gBp:", num_c_bp)
    print("Components connexes de gDp:", num_c_dp)

    # components més llargues de gBp i gDp
    c_llarga_b = len(max(nx.connected_components(gBp), key=len))
    # c_llarga_b = len(c_llarga_b)

    c_llarga_d = len(max(nx.connected_components(gDp), key=len))
    # c_llarga_d = len(c_llarga_d)
    print("\n------ EXERCICI 5 ---------------------------------------\n")
    print("Mida de la component més gran gBp:", c_llarga_b)
    print("Mida de la component més gran gDp:", c_llarga_d)