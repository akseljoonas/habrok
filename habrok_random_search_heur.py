import pandas as pd 
import numpy as np   
from sklearn.metrics import fbeta_score 
import random

useravg = pd.read_csv("./useravg.csv")
gameavg = pd.read_csv("./gameavg.csv")
matrix_fact = pd.read_csv("./matrix-7.csv")
genre = pd.read_csv("./genre.csv")
bert = pd.read_csv("./bert_model.csv")

train = pd.read_csv("./train_split.csv")
eval = pd.read_csv("./eval_split.csv")
test = pd.read_csv("./test_split.csv")



ensemble= pd.merge(useravg, gameavg, left_on= 'id', right_on='id', how ='inner', suffixes=('_useravg', '_gameavg'))
ensemble = ensemble.merge(matrix_fact, on='id', how='inner')
ensemble = ensemble.merge(genre, on='id', how='inner', suffixes=('_matrix', '_genre'))
ensemble = ensemble.merge(bert, on='id', how='inner', suffixes=('_genre','_bert' ))
column_name_mapping = {
    'voted_up_useravg': 'useravg',
    'voted_up_gameavg': 'gameavg',
    'voted_up_matrix': 'matrix',
    'voted_up_genre': 'genre',
    'voted_up': 'bert'
}

# Rename columns
ensemble = ensemble.rename(columns=column_name_mapping)


def ensemble_predict(ensemble_matrix, coef_to_try):
    ensemble_matrix[ensemble_matrix == 0] = -1
    weighted_matrix = ensemble_matrix * coef_to_try
    avg = weighted_matrix.sum(axis=1)
    ensemble_pred = np.array([1 if score > 0.8 else 0 for score in avg])
    return ensemble_pred
    
    


# Define the range of values for coefficients
value_choice = {
    'useravg': [ 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75],
    'gameavg': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
    'matrix': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9],
    'genre': [0.05, 0.1, 0.15, 0.2, 0.3],
    'bert': [0.05, 0.1, 0.15, 0.2, 0.3]
    }


# Perform grid search
best_score = 527926
best_coefficients = [0.2, 0.15, 0.55, 0.05, 0.1]
tried_list = [[0.2, 0.15, 0.55, 0.05, 0.1], [0.7, 0.4, 0.5, 0.1, 0.1], [0.7, 0.4, 0.6, 0.1, 0.1], 
              [0.7, 0.4, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.3, 0.1], [0.2, 0.15, 0.55, 0.05, 0.1],
              [0.2, 0.15, 0.55, 0.05, 0.1], [0.2, 0.2, 0.45, 0.1, 0.1]]
best = [[0.2, 0.15, 0.55, 0.05, 0.1]]

for i in range(2000):
    # Randomly choose coefficients not in not_best
    coef_to_try = None
    while coef_to_try is None or any(np.all(coef_to_try == np.array(tried_list), axis=1)):
        coef_to_try = np.array([
            random.choice(value_choice['useravg']),
            random.choice(value_choice['gameavg']),
            random.choice(value_choice['matrix']),
            random.choice(value_choice['genre']),
            random.choice(value_choice['bert'])
        ])
    
    ensemble_pred = ensemble_predict(ensemble_matrix.drop(['id'], axis=1), coef_to_try)
    score = sum(ensemble_pred)
    
    # Check if this combination is the best so far
    if score < best_score:
        best_score = score
        best_coefficients = coef_to_try
    
    tried_list.append(coef_to_try)

    if i % 1000 == 0:
        best.append(best_coefficients)
        
        with open('output.txt','a') as file:
    
            print(f"\nBest Coefficients at iteration {i}: {best_coefficients}", file=file)
            print(f"\nBest F0.5 score at iteration {i}: {best_score}", file = file)


best.append(best_coefficients)
with open('output.txt','a') as file:
    
    print(f"\nFinal best Coefficients: {best_coefficients}", file=file)
    print(f"\nFinal best F0.5 score: {best_score}", file = file)
