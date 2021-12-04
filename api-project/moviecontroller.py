import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import numpy as np
from dask import dataframe as dd
import timeit


def get_movies():
    url = 'https://storage.googleapis.com/movie-data-4/links_small.csv'
    df = pd.read_csv(url)
    return df.to_json()

def get_movies_by_movieId_userId(movieId, imdbId):
    url = 'https://storage.googleapis.com/movie-data-4/links_small.csv'
    df = pd.read_csv(url)
    return df[(df.movieId == movieId) & (df.imdbId == imdbId)].to_json()