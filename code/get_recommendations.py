import pandas as pd 
import os
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from itertools import combinations
import scipy.spatial.distance as dist

def get_LSH_popular(title,pair_df_new,indices,feature_df,combined):
    """
    Given a movie title, we first find it's movieID, then get the movies that are in the same bucket of that movie,
    sort these movies based on popularity and return the top 10 with highest popularity.

    """
    idx = indices[title]
    try:
        sim_scores = pair_df_new.loc[idx,:].sort_values(by=['popularity'],ascending=False)
    except:
        print('No similar movies based on the given movie')
        print('Recommend the most popular movies')
        popular = feature_df.sort_values(by=['popularity'],ascending=False)
        movie_indices = popular.index[:10]
        recom_list = []
        for i in movie_indices:
            a = combined.loc[i,'title']
            recom_list.append(a)
        return recom_list
    
    if len(sim_scores) >=10:
        movie_indices = list(sim_scores.index[:10])
    else:
        movie_indices = list(sim_scores.index)
    recom_list = []
    for i in movie_indices:
        a = combined.loc[i,'title']
        recom_list.append(a)
    
    return recom_list

### function calculate cosine similarities 
##def cal_cosine(m1_idx,m2_idx):
##    return 1-dist.cosine(feature_df.iloc[m1_idx,:],feature_df.iloc[m2_idx,:])

# Function that takes in movie title as input and outputs most similar movies
def get_LSH_cosine(indices,title, pair_df,feature_df,combined,cal_cosine):
    """
    Given a movie title, we first find it's index in the movie profile (not movieID), 
    then get the movies that are in the same bucket of that movie,
    calculate cosine similarities and sort these movies based on cosine similarities and 
    return the top 10 with highest similarities.
    If the given movie has no other item in the same bucket, return the most popular movies

    """
    idx = indices[title]
    try:
        sim_df = pair_df[pair_df['movie1']==idx].copy()
        sim_df['cosine'] = sim_df.apply(lambda x: cal_cosine(x['movie1'], x['movie2']),axis=1)
        sim_scores = sim_df['cosine'].sort_values(ascending=False)
    except:
        print('No similar movies based on the given movie')
        print('Recommend the most popular movies')
        popular = feature_df.sort_values(by=['popularity'],ascending=False)
        movie_indices = popular.index[:10]
        recom_list = []
        for i in movie_indices:
            a = combined.loc[i,'title']
            recom_list.append(a)
        return recom_list
    
    if len(sim_scores) >=10:
        movie_indices = list(sim_scores.index[:10])
    else:
        movie_indices = list(sim_scores.index)
    recom_list = []
    for i in movie_indices:
        a = combined.loc[i,'title']
        recom_list.append(a)
    
    return recom_list

