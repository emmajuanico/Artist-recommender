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

def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # obtenció de la informació de l'artista entrat com a paràmetre
    results = sp.search(q='artist:' + artist_name, type='artist')

    # retornar l'id de l'artista determinat
    return results['artists']['items'][0]['id']

def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """
    # funció que permet obtenir la informació d'un id determinat
    def get_artist_info(ide, artist): # "artist" és es l'artista que correspon a l'"ide"
        propietats = {
            'id': str(ide),
            'name': str(artist['name']),
            'followers': str(artist['followers']['total']),
            'popularity': str(artist['popularity']),
            'genres': str(artist['genres'])
        }
        return propietats
    
    # funció que afegeix un node amb id "ide", informació de l'artista a "artist", en el graf inserit
    def afegir_node(ide, artist, graf):
        info = get_artist_info(ide,artist) # obtenir la informació requerida de l'artista
        if info:
            graf.add_node(ide, name=info['name'], followers=info['followers'], popularity=info['popularity'], genres=info['genres'])

    # inicialització de les variables
    graf = nx.DiGraph() # creació del graf dirigit
    visited = set() # emmagatzament dels nodes visitats
    queue = [seed] # nodes a visitar

    while queue and len(visited) < max_nodes_to_crawl: # comprovació que no es superi el màxim inserit
        # segons l'estratègia a seguir...
        if strategy == "BFS":
            nAct = queue.pop(0) # s'agafa el primer node inserit a la cua
        else:
            nAct = queue.pop() # s'agafa l'últim node inserit a la cua

        # si el node no s'ha visitat anteriorment
        if nAct not in visited:
            # si no està afegit en el graf, s'afegeix
            if nAct not in graf.nodes():
                node = sp.artist(nAct)
                afegir_node(nAct, node, graf)
            
            # a partir de tots els artistes relacionats amb el node
            veins = sp.artist_related_artists(nAct)['artists']
            nous = [] # llista pels nous nodes a explorar, únicament pel  DFS

            # per cada vei
            for vei in veins:
                id_vei = vei['id'] # obtenció de l'id
                afegir_node(id_vei, vei, graf) # s'afegeix el node en el graf
                graf.add_edge(nAct, id_vei) # s'afegeix l'aresta dirigida entre el node que s'explora i el vei

                # si no s'ha explorat anteriorment el vei, s'afegeix a la cua
                if id_vei not in visited:
                    if strategy == "BFS":
                        queue.append(id_vei)
                    else:
                        nous.insert(0,id_vei) # subcua per assegurar el bon seguiment de l'algorisme

            if strategy == "DFS":
                if nous:
                    queue.extend(nous)
            visited.add(nAct) # node explorat --> node visitat

    # es desa el graf resultat en un arxiu graphml
    nx.write_graphml(graf, out_filename)
    return graf


def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
    '''
    Get track data for each visited artist in the graph.

    :param sp: spotipy client object
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    '''
    # inicialització de les variables 
    dicc = {                # diccionari amb tota la informació requerida
        "song_id": [],
        "song_duration": [],
        "song_name": [],
        "song_popularity": [],
        "danceability": [],
        "energy": [],
        "loudness": [],
        "speechiness": [],
        "acousticness": [],
        "instrumentalness": [],
        "liveness": [],
        "valence": [],
        "tempo": [],
        "album_id": [],
        "album_name": [],
        "album_release_date": [],
        "artist_id": [],
        "artist_name": []
    }
    explorats = [] # llista dels nodes explorats

    # per cada graf de la llista entrada com a paràmetre
    for graf in graphs: 
        # iteració per tots els nodes del graf
        for node in graf.nodes(): 
            # comprovació que no s'hagi afegit anteriorment i que sigui un node explorat (grau sortida > 0)
            if node not in explorats and graf.out_degree(node) > 0: 
                explorats.append(node) # s'afegeix com a node explorat

                # obtenció de la informació de les 10 cançons més populars d'aquell artista
                tracks = sp.artist_top_tracks(node, country='ES')
                time.sleep(0.1)

                llista_tracks = list(map(lambda x: x['id'], tracks['tracks']))
                audio = sp.audio_features(llista_tracks) # obtenció de la informació respecte l'audio 

                # per cada cançó
                for index, song in enumerate(tracks['tracks']):                    
                    # s'afegeixen les dades requerides
                    dicc["song_id"].append(song["id"])
                    dicc["song_duration"].append(song["duration_ms"])
                    dicc["song_name"].append(song["name"])
                    dicc["song_popularity"].append(song["popularity"])
                    dicc["danceability"].append(audio[index]["danceability"])
                    dicc["energy"].append(audio[index]["energy"])
                    dicc["loudness"].append(audio[index]["loudness"])
                    dicc["speechiness"].append(audio[index]["speechiness"])
                    dicc["acousticness"].append(audio[index]["acousticness"])
                    dicc["instrumentalness"].append(audio[index]["instrumentalness"])
                    dicc["liveness"].append(audio[index]["liveness"])
                    dicc["valence"].append(audio[index]["valence"])
                    dicc["tempo"].append(audio[index]["tempo"])
                    dicc["album_id"].append(song["album"]["id"])
                    dicc["album_name"].append(song["album"]["name"])
                    dicc["album_release_date"].append(song["album"]["release_date"])
                    dicc["artist_id"].append(node)
                    dicc["artist_name"].append(song["artists"][0]["name"])
                    
    # conversió del diccionari en un dataframe
    dataframe = pd.DataFrame(dicc) 

    # es desa el dataframe resultat en un arxiu csv
    dataframe.to_csv(out_filename, index=False) 
    return dataframe

if __name__ == "__main__":
    # dades d'iniciailització
    CLIENT_ID = "329dbec6390e41d5827004c82be28cac"
    CLIENT_SECRET = "93158e0f22464ecdb27ac2bcbabea03b"
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    ## FUNCIÓ 1
    # id de la Taylor Swift
    t = search_artist(sp,"Taylor Swift")

    ## FUNCIÓ 2
    directori = os.path.dirname(os.path.abspath(__file__)) # ruta del directori actual

    # creació per BFS
    arxiu = os.path.join(directori, "gB.graphml")
    gB = crawler(sp, t, 100, "BFS", arxiu)
    
    # creació per DFS
    arxiu = os.path.join(directori, "gD.graphml")
    gD = crawler(sp, t, 100, "DFS", arxiu)

    # ordre dels dos grafs generats
    print("\nOrdre del gB:",gB.order())
    print("Ordre del gD:",gD.order())

    # mida dels dos grafs generats
    print("\nMida del gB:",gB.size())
    print("Mida del gD:",gD.size())
    
    # graus d'entrada i de sortida
    for i in [gB, gD]:
        print("\n\n",i)
        ind = 0
        for p in [dict(i.in_degree()),dict(i.out_degree())]:
            g_max = max(p.values())
            g_min = min(p.values())
            g_mean = sum(p.values()) / len(p)
            if ind == 0:
                print("Graus d'entrada:\n\tmàxim:",g_max,"\n\tmínim:",g_min,"\n\tmitjana:",g_mean)
                ind += 1
            else:
                print("Graus de sortida:\n\tmàxim:",g_max,"\n\tmínim:",g_min,"\n\tmitjana:",g_mean)

    """directori = os.path.dirname(os.path.abspath(__file__))
    arxiu = os.path.join(directori, "gB.graphml")
    lectura_gb = nx.read_graphml(arxiu)
    gB = lectura_gb.to_directed()
    lectura_gd = nx.read_graphml(arxiu)
    gD = lectura_gd.to_directed()"""

    ## FUNCIÓ 3
    # track data dels dos grafs generats anteriorment
    arxiu = os.path.join(directori, "song.csv")
    D = get_track_data(sp, [gB,gD], arxiu)

    # nombre d'informació obtinguda
    print("nombre d'artistes:",D['artist_id'].nunique())
    print("nombre d'àlbums:",D['album_id'].nunique())
    print("nombre de cançons:",D['song_id'].nunique())

    