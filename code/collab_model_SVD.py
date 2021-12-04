## author: Xuelian Jia
## Date: 12-03-2021
## Description: this script can
## 1) takes given User ID, Movie ID, True rating for the given User-Movie pair (optional)
## and get the predicted rating fot this User-Movie pair
## 2) or takes the User ID, then predict the ratings for all the movies that the user has not rated and 
## return the top rated movies for recommendation.
## Requirement packages: Pandas, scikit-surprise
## Reqiurement files under data folder: 1) rating_small.csv or rating.csv, 2) combined.csv, which has the movie profile: ID, Title, etc.

import pandas as pd
import os
import argparse 
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from surprise import dump

parser = argparse.ArgumentParser(description='Build collaborative recommentioner')
parser.add_argument('-uid', '--UserID', type=int, help ='User ID interger')
parser.add_argument('-iid', '--MovieID', type=int, help ='Movie ID interger if you have the true rating')
parser.add_argument('-r_ui', '--TrueRating', type=float, help ='true rating value of the given User-Item pair')
parser.add_argument('-fn', '--filename', type=str, help ='finename of the model to dump')

args = parser.parse_args()
uid = args.UserID
iid = args.MovieID
r_ui = args.TrueRating
filename = args.filename

# read the ratings file, which will be used for collaborative filtering model development
rating = pd.read_csv('../data/ratings_small.csv')
print('========= Reading the Rating DataFrame =========')
print (rating.head())

# build the rating matrix, where we can get the rated and not rated movies for every user
rating_mat = rating.pivot_table('rating',index='userId',columns='movieId')

# read the combined movie profile, which we used to look for the movie name given a movie ID
combined = pd.read_csv('../data/combined_info.csv',index_col=0)
print('========= Reading the movie profile  =========')
print (combined.head())

reader = Reader()
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

param_grid = {
    "n_epochs": [5, 10],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.4, 0.6]
}

pardir = os.path.abspath(os.path.join(os.path.dirname('collab_model_SVD.py'),os.path.pardir))
if not os.path.exists(os.path.join(pardir, 'model')):
    os.makedirs(os.path.join(pardir, 'model'))
    
    gs = GridSearchCV(SVD, param_grid, measures=["rmse","mae"], cv=3)

    print('========= Training model using SVD algorithm  =========')
    gs.fit(data)

    print("Best RMSE score of GridSearchCV: ",gs.best_score["rmse"])
    print("Best parameters of GridSearchCV: ",gs.best_params["rmse"])

    alg = gs.best_estimator['rmse']
    alg.fit(data.build_full_trainset())

    # save model for future use
    
    dump.dump(os.path.join(pardir, 'model',filename),algo=alg,verbose=1)
else:
    pred, alg = dump.load(os.path.join(pardir, 'model',filename))


movies = list(rating_mat.columns)

def get_predictions(user_id):
    """given a user ID, use the model to predict the user's rating for all the not rated movies
    parameter user_id: integer ID of user
    return predictions dictionary with movie ID as key and predicted rating as values"""
    predictions = {}
    for i in movies:
        if rating_mat.loc[user_id, i]>=0:
            #print("Already rated movie ID:", i)
            pass
        else:
            pred = alg.predict(user_id, i).est
            predictions[i] = pred
    return predictions

def get_recommend_collab(user_id):
    """given a user ID, get the movie rating predictions, sort the predictions from high to low,
    consider ratings >= 3 for recommendation and output names of the top 10 movies if there are more than 10
    parameter user_id: integer ID of user
    return the list of movie names with top rating values
    """
    predictions = get_predictions(user_id)
    predictions_df = pd.DataFrame.from_dict(predictions,orient='index',columns=['prediction'])
    predictions_df = predictions_df.sort_values(by='prediction',ascending=False)
    predictions_scores = predictions_df[predictions_df['prediction']>=3]
    if predictions_scores.shape[0] >= 10:
        movie_ids = list(predictions_scores.index[:10])
    else:
        movie_ids = list(predictions_scores.index)
    recom_list = []
    for i in movie_ids:
        try:        
            a = combined.loc[i,'title']
            recom_list.append(a)
        except:
            pass
    return recom_list

if r_ui:
    pred = alg.predict(uid = uid,iid = iid,r_ui = r_ui)
    print (pred)
else:
    recom_list = get_recommend_collab(uid)
    print(recom_list)
