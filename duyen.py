import pandas as pd
pd.set_option('display.max_colwidth', None)
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import numpy as np
from dask import dataframe as dd
import timeit


# READ DATA

# start1 = timeit.default_timer()
# movie_df1 = pd.read_csv('movies_metadata.csv', dtype={'popularity':'str'})
# rating_df1 = pd.read_csv('ratings_small.csv')
# links_df1 = pd.read_csv('links_small.csv', dtype={'tmdbId':'float64'})
# stop1 = timeit.default_timer()
# print('Time: ', stop1 - start1)  #0.41758359999999994


# start = timeit.default_timer()
movie_df = dd.read_csv('movies_metadata.csv',usecols=['title', 'id', 'imdb_id', 'overview'], dtype={'id':'object'})
rating_df = dd.read_csv('ratings_small.csv')
links_df = dd.read_csv('links_small.csv', dtype={'imdb_id':'object', 'tmdbId':'float64'})
# stop = timeit.default_timer()
# print('Time: ', stop - start)  #0.014915599999999918

# CLEAN UP DATA
movie_df = movie_df.dropna()
movie_df['id']=movie_df['id'].astype(float)

# MERGE DATA
all_ratings_df = rating_df.merge(links_df, on='movieId')
combined_df = movie_df.merge(all_ratings_df, left_on='id', right_on='tmdbId')

######################################################
print(combined_df.head())
