import pandas as pd 
import numpy as np
import scipy.spatial.distance as dist
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

def vector_profile(dataset):
    combined = pd.read_csv(dataset,
                       encoding='iso-8859-1',index_col=0)
    cv = CountVectorizer(input='content', encoding='iso-8859-1', decode_error='ignore', analyzer='word',
                      ngram_range=(1,1),max_features=1000)

    print('============ Fitting CountVectorizer model =========')
    cv_model = cv.fit_transform(combined['info'])
    df_cv = pd.DataFrame(cv_model.toarray(), index=combined.index,columns=sorted(cv.vocabulary_))

    cols = ['popularity','vote_average','vote_count','adult']
    combined2 = combined.loc[:,cols].copy()
    featureMatrix = pd.concat([df_cv,combined2],axis=1)
    print('==========feature vectorization results ==========')
    print(featureMatrix.head(2))

    # normalize the column values to the same scale
    scaler = MinMaxScaler()
    featureMatrix_norm = scaler.fit_transform(featureMatrix)
    featureMatrix_norm = pd.DataFrame(featureMatrix_norm,index=featureMatrix.index, columns=featureMatrix.columns)
    featureMatrix_norm.to_csv('../data/movie_norm_featureMatrix.csv')
    return featureMatrix_norm
