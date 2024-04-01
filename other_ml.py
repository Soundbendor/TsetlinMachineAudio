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
from sklearn.metrics import accuracy_score
from multiprocessing import Process, Manager


def train_ml_algo(train_x, train_y, val_x, val_y, model_class, params, result_dict, fold_num):
    # pre-assign data

    model = model_class(**params)
    model.fit(train_x, train_y)

    # Make predictions
    y_pred = model.predict(val_x)

    # Evaluate the model
    accuracy = accuracy_score(val_y, y_pred)
    f1 = f1_score(val_y, y_pred, average='micro')
    cm = confusion_matrix(val_y, y_pred)

    result_dict[fold_num] = {"label": f"fold_{fold_num}", "acc": accuracy, "f1": f1, "cm": cm}

    return result_dict


def main(args):
    class_type = args.class_type
    ml_algo = args.ml_algo

    models_and_params = {
        "svm": (SVC, {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['auto', 2 / 550]}),
        "rf": (RandomForestClassifier, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
        "mlp": (MLPClassifier, {'hidden_layer_sizes': [(100, 100), (1000, 1000), (10000, 10000)]})
    }

    model, param_grid = models_and_params[ml_algo]

    if class_type == "vowel":
        class_val = 0
    elif class_type == "technique":
        class_val = 1
    else:
        raise ValueError("No class type")

    # Data stuff other commetns
    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_all_all_mfcc_avg_2024-03-29-16-23.pickle", 'rb') as f:
        data = pickle.load(f)

    folds = {  # for singer id
        0: [1, 10, 16, 17],
        1: [0, 2, 11, 18],
        2: [5, 8, 13, 14],
        3: [4, 9, 12, 19],
        4: [3, 6, 7, 15],
    }

    y_data = data["y"][:, class_val]
    y_indices = np.where(y_data != -1)[0]
    y_data = y_data[y_indices]
    x_data = data["x"][y_indices]

    check_y = data["y"][y_indices]

    fold_1_test_idx = np.concatenate([np.where(check_y[:, -1] == idx)[0] for idx in folds[0]])
    fold_1_train_idx = np.setdiff1d(np.arange(len(x_data)), fold_1_test_idx)

    grid_search = GridSearchCV(estimator=model(), param_grid=param_grid, cv=None, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(x_data[fold_1_train_idx], y_data[fold_1_train_idx])
    best_params = grid_search.best_params_

    manager = Manager()
    result_dict = manager.dict()

    processes = []

    for fold_num, fold_indices in folds.items():
        test_data_indices = np.concatenate([np.where(check_y[:, -1] == idx)[0] for idx in fold_indices])
        train_data_indices = np.setdiff1d(np.arange(len(x_data)), test_data_indices)
        print(
            f"test_idx {len(test_data_indices)}, train: {len(train_data_indices)}, total: {len(train_data_indices) + len(test_data_indices)}")
        p = Process(target=train_ml_algo,
                    args=(x_data[train_data_indices], y_data[train_data_indices], x_data[test_data_indices],
                          y_data[test_data_indices], model, best_params, result_dict, fold_num))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    data_dict = {fold: result_dict[fold] for fold in range(len(result_dict))}
    pickle_path = f"/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/pickles/{ml_algo}_{class_type}"
    with open(pickle_path, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the other models")
    parser.add_argument("ml_algo", help="[svm,rf,mlp]")
    parser.add_argument("class_type", help="vowel, technique")
    args = parser.parse_args()
    main(args)
