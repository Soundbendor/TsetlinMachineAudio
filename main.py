import numpy as np
from tqdm import tqdm
from tmu.models.classification import vanilla_classifier
import os
import json
# import matplotlib.pyplot as plt
import logging
import pickle
import datetime
import argparse
from sklearn.metrics import accuracy_score
from multiprocessing import Process, Manager


#

# logging.getLogger('matplotlib').setLevel(logging.WARNING)
# logging.getLogger("neptune").setLevel(logging.CRITICAL)
def get_save_path(args, HEAD):
    """Make save path
    """
    date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    suffix = "{}_{}".format(args[0], date)
    result_path = os.path.join(HEAD, suffix)
    return result_path


# TODO consider wrapping neptune in debug == False ctrl-F all neptune calls


def batched_train(model, X, y, batch_size, epochs=1):
    array_size = len(X)
    for i in range(0, array_size, batch_size):
        model.fit(X[i:i + batch_size], y[i:i + batch_size], epochs=epochs)


def train_fold(train_x, train_y, val_x, val_y, number_clauses, T, s, epochs, batch_size, result_dict, fold_num):
    model = vanilla_classifier.TMClassifier(number_clauses,
                                            T=T,
                                            s=s,
                                            number_of_state_bits_ta=100,
                                            incremental=True,
                                            platform='GPU',
                                            seed=1066)

    train_y, val_y = train_y.reshape(-1), val_y.reshape(-1)
    train_final = []
    val_final = []

    for e in range(epochs):
        batched_train(model, train_x, train_y, batch_size)
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)

        train_acc = accuracy_score(train_y, train_preds)
        val_acc = accuracy_score(val_y, val_preds)

        train_final.append(train_acc)
        val_final.append(val_acc)

    result_dict[fold_num] = {
        "train_acc": train_final,
        "val_acc": val_final,
        "preds": np.array(val_preds).tolist()  # Convert to list to be pickleable
    }


def main(args):
    class_type = args.class_type
    number_clauses = int(args.clauses)
    s = int(args.s)
    T = int(args.T)
    epochs = int(args.epochs)
    config_path = args.config

    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("No config path")

    if class_type == "vowel":
        class_val = 0
    elif class_type == "technique":
        class_val = 1
    else:
        raise ValueError("No class type")

    # Data stuff
    with open(config["data"], 'rb') as f:
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
    batch_size = 1000
    manager = Manager()
    result_dict = manager.dict()

    processes = []

    for fold_num, fold_indices in folds.items():
        test_data_indices = np.concatenate([np.where(check_y[:, -1] == idx)[0] for idx in fold_indices])
        train_data_indices = np.setdiff1d(np.arange(len(data['x'])), test_data_indices)
        p = Process(target=train_fold,
                    args=(x_data[train_data_indices],y_data[train_data_indices],x_data[test_data_indices],y_data[test_data_indices], number_clauses, T, s, epochs, batch_size, result_dict, fold_num))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Prepare data for saving
    data_dict = {fold: result_dict[fold] for fold in range(len(result_dict))}

    pickle_path = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/pickles"
    pickle_file = get_save_path(["all_folds"], pickle_path)
    with open(pickle_file, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TM model")
    parser.add_argument("class_type", help="vowel or technique")
    parser.add_argument("clauses", help="Number of clauses")
    parser.add_argument("s", help="sensitivity")
    parser.add_argument("T", help="threshold")
    parser.add_argument("epochs", help="Number of training epochs")
    parser.add_argument("config", help="config file")
    args = parser.parse_args()
    main(args)

"/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_all_all_folds_2_bools_2024-03-29-14-02"