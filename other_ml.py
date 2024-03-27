import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import argparse

def train_ml_algo(model, params):
    # pre-assign data
    x_tr_files = []
    x_te_files = []
    y_tr_files = []
    y_te_files = []

    param_grid = params
    model = model(random_state=1066)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(x_tr_files[0], y_tr_files[0])
    best_model = grid_search.best_estimator_

    res_list = [grid_search.best_params_]
    for i in range(len(x_tr_files)):
        x_tr = np.load(x_tr_files[i], mmap_mode='r')
        x_te = np.load(x_te_files[i], mmap_mode='r')
        y_tr = np.load(y_tr_files[i], mmap_mode='r')
        y_te = np.load(y_te_files[i], mmap_mode="r")
        best_model.fit(x_tr, y_tr)

        # Make predictions
        y_pred = best_model.predict(x_te)

        # Evaluate the model
        accuracy = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred)

        res_dict = {"label": f"fold_{i + 1}", "acc": accuracy, "f1": f1, "cm": confusion_matrix}
        res_list.append(res_dict)

    return res_list


def main(args):
    pickle_path = args.pickle_path
    # Define the models and parameter grids
    models_and_params = [
        (SVC, {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['auto', 2/550]}),
        (RandomForestClassifier, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
         MLPClassifier, {'hidden_layer_sizes': [(100, 100), (600, 600), (11000, 11000)]})
    ]

    # Use multiprocessing.Pool to parallelize the training
    with multiprocessing.Pool() as pool:
        model_folds_results = pool.starmap(train_ml_algo, models_and_params)

    with open(pickle_path,"rb") as f:
        pickle.dump(model_folds_results,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the other models")
    parser.add_argument("pickle_path", help="Where should results be stored?")
    args = parser.parse_args()
    main(args)