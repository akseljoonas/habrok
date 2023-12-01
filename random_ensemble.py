import pandas as pd 
import numpy as np   
from sklearn.metrics import fbeta_score 
import random


choices_made = [0]
current_choice = 0
for model_combination in range(1, 100):

    while current_choice in choices_made:
        user_choice = random.choice(range(1, 33))
        game_choice = random.choice(range(1, 33))
        matrix_choice = random.choice(range(1, 33))
        genre_choice = random.choice(range(1, 33))
        bert_choice = random.choice(range(1, 33))

        current_choice = [user_choice, game_choice, matrix_choice, genre_choice, bert_choice]


    choices_made.append([user_choice, game_choice, matrix_choice, genre_choice, bert_choice])

    useravg = pd.read_csv(f"./files/ensemble ({user_choice}).csv")
    gameavg = pd.read_csv(f"./files/ensemble ({game_choice}).csv")
    matrix_fact = pd.read_csv(f"./files/ensemble ({matrix_choice}).csv")
    genre = pd.read_csv(f"./files/ensemble ({genre_choice}).csv")
    bert = pd.read_csv(f"./files/ensemble ({bert_choice}).csv")

    train = pd.read_csv("./train_split.csv")
    eval = pd.read_csv("./eval_split.csv")
    test = pd.read_csv("./test.csv")

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
    ensemble_matrix = ensemble

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


    # Perform grid search
    best_score = 527926
    best_coefficients = [0.2, 0.15, 0.55, 0.05, 0.1]
    tried_list = [[0.05, 0.05, 0.05, 0.05, 0.05]]
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
        if score < best_score and score > 40000:
            best_score = score
            best_coefficients = coef_to_try

        tried_list.append(coef_to_try) 



    best.append(best_coefficients)
    with open(f'./ensemble_comb/ensemble combination {model_combination}.txt','a') as file:
        print(f'\n\n\n Random ensemble but with 100 iteartions)', file=file)
        print(f'\n with these models: {(user_choice, game_choice, matrix_choice, genre_choice, bert_choice)})', file=file)
        print(f"\nFinal best Coefficients: {best_coefficients}", file=file)
        print(f"\nFinal best evaluation score: {best_score}", file = file)
