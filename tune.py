import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

import json

def batched_train(model, X, y, batch_size, epochs=1):
    array_size = len(X)
    for i in range(0, array_size, batch_size):
        model.fit(X[i:i + batch_size], y[i:i + batch_size], epochs=epochs)


def set_model_params(model_class, incremental=True, platform="GPU", seed=1066, **kwargs):
    params = {'incremental': incremental, 'platform': platform, 'seed': seed, **kwargs}
    return model_class(**params)


def hyperparameter_tuning(model_class, X_train, y_train, X_val, y_val, param_grid, training_epochs=2, max_epochs=8, search_width=3, tol=1e-4, batch_size=256):
    best_params = None
    best_score = 0
    no_improvement_count = 0

    for params in ParameterGrid(param_grid):
        model = set_model_params(model_class, **params)
        batched_train(model, X_train, y_train, batch_size, training_epochs)
        for j in range(max_epochs):
            batched_train(model, X_train, y_train, batch_size, epochs=1)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='micro')
            if score > best_score + tol:
                best_score = score
                best_params = params
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= search_width:
                break

    return best_params




if __name__ == "__main__":
    # use these names
    param_grid = {"number_of_clauses": [1000, 2500, 5000],
           "T": [40, 120, 200],
           "s": [5, 10, 15]
           }


    with open("config_main.json") as f:
        config = json.load(f)

    train_X = config["train_x"]
    train_y = config["train_y"].reshape(-1,)
    test_x = config["test_x"]
    test_y = config["test_y"].reshape(-1, )

    model_class = TMClassifier
    best_params = hyperparameter_tuning(model_class,train_X,train_y,test_x,test_y,param_grid)
    print(best_params)


