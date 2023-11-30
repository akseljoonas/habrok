import pandas as pd 
import numpy as np   
from sklearn.metrics import fbeta_score 
import random
from multiprocessing import Pool, Lock

useravg = pd.read_csv("./useravg_eval.csv")
gameavg = pd.read_csv("./gameavg_eval.csv")
matrix_fact = pd.read_csv("./matrix_eval.csv")
genre = pd.read_csv("./genre_eval.csv")
bert = pd.read_csv("./bert_eval.csv")

train = pd.read_csv("./train_split.csv")
eval = pd.read_csv("./eval_split.csv")

ensemble= pd.merge(useravg, gameavg, left_on='id', right_on='id', how='inner', suffixes=('_useravg', '_gameavg'))
ensemble = ensemble.merge(matrix_fact, on='id', how='inner')
ensemble = ensemble.merge(genre, on='id', how='inner', suffixes=('_matrix', '_genre'))
ensemble = ensemble.merge(bert, on='id', how='inner', suffixes=('_genre','_bert'))
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
    'useravg': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'gameavg': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'matrix': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'genre': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'bert': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

# Assuming ensemble_data is your feature matrix and target variable
ensemble_matrix = ensemble.merge(eval[['id', 'voted_up']], on='id', how='inner')

# Initialize shared variables and lock for synchronization
best_score = -1
best_coefficients = None
best = [[0.1, 0.05, 0.7, 0.1, 0.9]]
tried_list = [[0.5, 0.5, 0.5, 0.5, 0.5]]
lock = Lock()

def evaluate_coefficients(i):
    global best_score, best_coefficients, tried_list, best
    coef_to_try = None
    while coef_to_try is None or any(np.all(coef_to_try == np.array(tried_list), axis=1)):
        coef_to_try = np.array([
            random.choice(value_choice['useravg']),
            random.choice(value_choice['gameavg']),
            random.choice(value_choice['matrix']),
            random.choice(value_choice['genre']),
            random.choice(value_choice['bert'])
        ])

    ensemble_pred = ensemble_predict(ensemble_matrix.drop(['id', 'voted_up'], axis=1), coef_to_try)
    score = fbeta_score(ensemble_matrix['voted_up'], ensemble_pred, beta=0.5)

    # Use a lock to prevent race conditions when updating shared variables
    with lock:
        if score > best_score:
            best_score = score
            best_coefficients = coef_to_try

        tried_list.append(coef_to_try)

        if i % 1000 == 0:
            best.append(best_coefficients)
            with open('output.txt', 'w') as file:
                print(f"Best Coefficients at iteration {i}: {best_coefficients}", file=file)
                print(f"Best F0.5 score at iteration {i}: {best_score}", file=file)

# Set the number of processes to the number of CPU cores
num_processes = multiprocessing.cpu_count()

# Use a multiprocessing pool for parallel execution
with Pool(processes=num_processes) as pool:
    pool.map(evaluate_coefficients, range(30000))

best.append(best_coefficients)
with open('output.txt', 'w') as file:
    print(f"Final best Coefficients: {best_coefficients}", file=file)
    print(f"Final best F0.5 score: {best_score}", file=file)
