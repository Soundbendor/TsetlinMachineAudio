import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import f1_score
from multiprocessing import Pool
import multiprocessing
import pickle


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
            score = f1_score(y_val, y_pred, average='weighted')
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
    param_grid = {"number_of_clauses": [ 2500, 5000, 1000],
                  "T": [100,50,200],
                  "s": [10, 20 ,5,]
                  }#gsgs

    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/pickles/vowel_resample_4b_2024-04-02-12-59",'rb') as f:
       data = pickle.load(f)
    y_data = data["y"][:, 1]
    y_indices = np.where(y_data != -1)[0]
    y_data = y_data[y_indices]
    x_data = data["x"][y_indices]

    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1066)
    train_index, test_index = next(iter(kf.split(x_data, y_data)))
    train_x, val_x = x_data[train_index], x_data[test_index]
    train_y, val_y = y_data[train_index].reshape(-1), y_data[test_index].reshape(-1)
    model_class = TMClassifier
    best_params, best_score, best_scores, best_params_list = hyperparameter_tuning(TMClassifier,train_x,train_y,val_x,val_y,param_grid)
    print(best_params)
