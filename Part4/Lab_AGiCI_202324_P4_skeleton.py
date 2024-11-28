import networkx as nx
import pandas as pd
import spotipy
import numpy as np
from math import pi

from Lab_AGiCI_202324_P2_skeleton import compute_mean_audio_features as compute
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
import matplotlib.pyplot as plt

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #
def plot_degree_distribution ( degree_dict : dict , normalized : bool = False , loglog : bool = False ) -> None :
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pass
    # ----------------- END OF FUNCTION --------------------- #


def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    # df = pd.read_csv("nba.csv")
    print(artists_audio_feat)
    dataframe = artists_audio_feat.loc[[artist1_id, artist2_id]]
    seleccionats = dataframe.drop(['loudness', 'tempo'], axis=1)
 
    
    categories=list(seleccionats)[:]
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
     
    
    ax = plt.subplot(111, polar=True)
     
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
     
    
    plt.xticks(angles[:-1], categories, color='0.5', label='Light Grey (Grayscale Intensity)')
     
    
    ax.set_rlabel_position(0)
    plt.yticks([ 0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'],color='0.5', label='Light Grey (Grayscale Intensity)', size=7)
    plt.ylim(0,1)

    # Ind1
    values=seleccionats.iloc[0].tolist()
    values += values[:1]
    nom1 = sp.artist(artist1_id)['name']
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=nom1)
    ax.fill(angles, values, color='b', alpha=0.075)
    ax.scatter(angles, values, color='b', marker='o', s=15)

    # Ind2
    values=seleccionats.iloc[1].tolist()
    values += values[:1]
    nom2 = sp.artist(artist2_id)['name']
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=nom2)
    ax.fill(angles, values, color='magenta', alpha=0.075)
    ax.scatter(angles, values, color='magenta', marker='o', s=15)
    
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.show()
    # ----------------- END OF FUNCTION --------------------- #



def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pass
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    
        
    CLIENT_ID = "93b8dddd10b94df3b1c08a8dc7b49201"
    CLIENT_SECRET = "84d44bb7f49c46c49d87c9ea18ab1fcf"
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    
    directori = os.path.dirname(os.path.abspath(__file__)) # ruta del directori actual

    arxiu = os.path.join(directori, "song.csv")
    D = pd.read_csv(arxiu)
    # similarity = "euclidean"#"cosine"#"euclidean"

    mitjanes = compute(D)
    # mitjanes = pd.read_csv("mitjanes.csv")
    # print(mitjanes)
    # print(mitjanes)
    
    plot_audio_features(mitjanes, '7vk5e3vY1uw9plTHJAMwjN', '7nU4hB040gTmHm45YYMvqc')
    # ------------------- END OF MAIN ------------------------ #
