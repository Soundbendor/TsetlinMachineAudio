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
import os







def train_ml_algo(model, params):
    # pre-assign data
    x_tr_files = [
        '/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_1_2024-03-14-17-22.npy',
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_2_2024-03-14-18-35.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_3_2024-03-14-19-52.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_4_2024-03-14-16-39.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_5_2024-03-14-20-21.npy"]
    x_te_files = [
        '/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_1_2024-03-14-17-26.npy',
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_2_2024-03-14-18-40.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_3_2024-03-14-19-56.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_4_2024-03-14-16-42.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_5_2024-03-14-20-25.npy"]
    y_tr_files = [
        '/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_1_2024-03-14-17-22.npy',
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_2_2024-03-14-18-35.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_3_2024-03-14-19-52.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_4_2024-03-14-16-39.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_5_2024-03-14-20-21.npy"]
    y_te_files = [
        '/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_1_2024-03-14-17-26.npy',
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_2_2024-03-14-18-40.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_3_2024-03-14-19-56.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_4_2024-03-14-16-42.npy",
        "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_5_2024-03-14-20-25.npy"]



    param_grid = params
    model = model(random_state=1066)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(np.load(x_tr_files[0], mmap_mode='r'), np.load(y_tr_files[0], mmap_mode='r'))
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
    ml_algo = args.ml_algo
    # Define the models and parameter grids
    models_and_params = {
        "svm":(SVC, {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['auto', 2/550]}),
        "rf": (RandomForestClassifier, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
        "mlp": (MLPClassifier, {'hidden_layer_sizes': [(100, 100), (600, 600), (11000, 11000)]})
    }

    results = train_ml_algo(*models_and_params[ml_algo])

    with open(pickle_path,"rb") as f:
        pickle.dump(results,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the other models")
    parser.add_argument("ml_algo",help="[svm,rf,mlp]")
    parser.add_argument("pickle_path", help="Where should results be stored?")
    args = parser.parse_args()
    main(args)