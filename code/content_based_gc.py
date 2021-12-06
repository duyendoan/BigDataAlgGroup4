## author: Xuelian Jia
## Date: 12-04-2021
## Description: this script
## 1) takes the given combined.csv, which is the movie profile contains information (words & numerical)
# we are interested.
## 2) vectorize the words using CountVectorizer and combine with numerical columns to get --> feature_df
## 3A) with feature_df, use LSH cosine similarity to hash similar movies to the same bucket, movies in the
## same bucket will form a candidate pair of similar movies
## when given a movie (name or ID), we can search its candidate pairs and among these pairs, we can recommend movies
## -------that are most similar to the query movie by calculating cosine similarity
## -------that are most popular based on the [popularity] column
## 3B) we can also calculate the pairwise cosine similarity for all the movies using linear_kernel. And when given a movie
## return the movies with highest cosine similarities.

## Requirement packages: Pandas, numpy,sklearn, itertools, scipy
import pandas as pd 
import os
import numpy as np
import argparse 
from sklearn.metrics.pairwise import linear_kernel
from itertools import combinations
import scipy.spatial.distance as dist
from LSH_cosine import signature_bit,sketch,LSH,find_pairs
from movie_profile import vector_profile
from get_recommendations import get_LSH_popular,get_LSH_cosine
from cosine_all import get_recommendations_all

parser = argparse.ArgumentParser(description='Give content-based recommentions')

parser.add_argument('-tl', '--title', type=str, help ='Movie title based on which you want to find similar movies')
parser.add_argument('-lsh', '--evaluate_lsh', type=str, help ='whether to use LSH or not')
parser.add_argument('-st', '--sort_on', type=str, help ='sort on popularity or cosine similarity after LSH')

args = parser.parse_args()
title = args.title
sort_on = args.sort_on
evaluate_lsh = args.evaluate_lsh

import time
start_time = time.time()

dataset = 'https://storage.googleapis.com/group-4-bucket/data/combined_info.csv'

# get vectorized movie profile feature_df
feature_df = vector_profile(dataset)
combined = pd.read_csv(dataset,index_col=0)


#Construct a reverse map of indices and movie titles
indices = pd.Series(combined.index, index=combined['title']).drop_duplicates()

def give_popula(m_idx):
    return feature_df.iloc[m_idx,-4]

# function calculate cosine similarities 
def cal_cosine(m1_idx,m2_idx):
    return 1-dist.cosine(feature_df.iloc[m1_idx,:],feature_df.iloc[m2_idx,:])


if evaluate_lsh=='y':
    featureMatrix = np.array(feature_df)
    n_feature = featureMatrix.shape[1]       # number of features)
    n_sig = 100                            # number of random vectors of signatures for each doc
    sketches = sketch(featureMatrix,n_feature,n_sig)
    LSH_buckets = LSH(sketches,10,10) # number of bands = 20, number of rows=10 in each band

    print('Number of buckets: ', len(LSH_buckets))

    candi_pairs = set()
    for key, value in LSH_buckets.items():
        if len(value) >= 2:
            pairs = find_pairs(value)
            candi_pairs.update(pairs)
    print('Number of candidate pairs: ', len(candi_pairs))

    pair_df = pd.DataFrame(sorted(candi_pairs),columns=['movie1','movie2'])
    
    if sort_on == 'pop':
        pair_df['popularity'] = pair_df.apply(lambda x: give_popula(x['movie2']),axis=1)
        pair_df_new = pair_df.set_index(['movie1','movie2'])
        recom_list = get_LSH_popular(title,pair_df_new,indices,feature_df,combined)
        print(recom_list)

    elif sort_on == 'cosine':
        recom_list = get_LSH_cosine(indices,title, pair_df,feature_df,combined,cal_cosine)
        print(recom_list)
        
elif evaluate_lsh=='n':
    print("Giving recommendation based on cosine similarity")
    recom_list = get_recommendations_all(title,feature_df,indices)
    print(recom_list)


print("--- %s seconds ---" % (time.time() - start_time))



