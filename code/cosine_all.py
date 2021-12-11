import pandas as pd 
import os
import numpy as np

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations_all(title,feature_df,indices):
    """
    Given a movie title, we first find it's movieID, then get the pairwsie similarity scores of all movies with that movie,
    sort the movies based on cosine similarity scores return the ID of the 10 most similar movies if it has
    """
    print(feature_df.head())
    feature_df = feature_df.dropna(axis=1)
    print(feature_df.shape)
    featureMatrix = np.array(feature_df)

    # Compute the pairwise cosine similarity matrix
    cosine_sim = cosine_similarity(featureMatrix, featureMatrix)
    cosine_sim = pd.DataFrame(cosine_sim,index=feature_df.index,columns=feature_df.index)
    dataset = 'https://storage.googleapis.com/group-4-bucket/data/combined_info.csv'
    combined = pd.read_csv(dataset,index_col=0)
    
    idx = indices[title]
    sim_scores = cosine_sim.loc[idx,:].sort_values(ascending=False)
    movie_indices = list(sim_scores.index[1:11])
    recom_list = []
    for i in movie_indices:
        a = combined.loc[i,'title']
        recom_list.append(a)
    
    return recom_list
