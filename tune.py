import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import f1_score
from multiprocessing import Pool
import multiprocessing
import json


def batched_train(model, X, y, batch_size, epochs=1):
    array_size = len(X)
    for i in range(0, array_size, batch_size):
        model.fit(X[i:i + batch_size], y[i:i + batch_size], epochs=epochs)


def set_model_params(model_class, incremental=True, platform="GPU", seed=1066, **kwargs):
    params = {'incremental': incremental, 'platform': platform, 'seed': seed, **kwargs}
    return model_class(**params)


def hyperparameter_tuning(model_class, X_train, y_train, X_val, y_val, param_grid, training_epochs=2, max_epochs=8,
                          search_width=3, tol=1e-4, batch_size=256):
    best_params = None
    best_score = 0
    no_improvement_count = 0
    best_scores = []
    best_params_list = []

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
        best_scores.append(best_score)
        best_params_list.append(best_params)

    return best_params, best_score, best_scores, best_params_list




if __name__ == "__main__":
    # use these names
    param_grid = {"number_of_clauses": [1000, 2500, 5000],
                  "T": [40, 120, 200],
                  "s": [5, 20, 35]
                  }

    # with open("config_main.json") as f:
    #    config = json.load(f)

    # train_x = np.load(config["train_x"], mmap_mode='r')
    # train_y = np.load(config["train_y"], mmap_mode='r').reshape((-1,))
    # assert len(train_y.shape) == 1

    # val_x = np.load(config["test_x"], mmap_mode='r')
    # val_y = np.load(config["test_y"], mmap_mode='r').reshape(-1, )
    # assert len(val_y.shape) == 1
    X = np.load(
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/singer/singer_X_ALL_2024-03-16-17-48.npy",
        mmap_mode="r")
    y = np.load(
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/singer/singer_y_ALL_2024-03-16-17-48.npy",
        mmap_mode="r")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1066)
    train_index, test_index = next(iter(kf.split(X, y)))
    train_x, val_x = X[train_index], X[test_index]
    train_y, val_y = y[train_index].reshape(-1), y[test_index].reshape(-1)
    model_class = TMClassifier
    best_params, best_score, best_scores, best_params_list = hyperparameter_tuning(TMClassifier,train_x,train_y,val_x,val_y,param_grid)
    print(best_params)
