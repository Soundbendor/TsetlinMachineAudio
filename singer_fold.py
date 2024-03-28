import pickle
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tmu.models.classification import vanilla_classifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from main import batched_train
from main import get_save_path
import argparse
from multiprocessing import Process, Manager


def train_fold(train_index, test_index, X, y, number_clauses, T, s, epochs, batch_size, result_dict, fold):
    
    model = vanilla_classifier.TMClassifier(number_clauses,
                                            T=T,
                                            s=s,
                                            number_of_state_bits_ta=100,
                                            incremental=True,
                                            platform='GPU',
                                            seed=1066)
    
    
    
    train_x, val_x = X[train_index], X[test_index]
    train_y, val_y = y[train_index].reshape(-1), y[test_index].reshape(-1)
    train_final = []
    val_final = []
    test_preds_list = []
    for e in range(epochs):
        batched_train(model, train_x, train_y, batch_size)
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)

        train_acc = accuracy_score(train_y, trian_preds)
        val_acc = accuracy_score(val_y,val_preds)

        train_final.append(train_acc)
        val_final.append(val_acc)

    result_dict[fold] = {
        "train_acc": train_final,
        "val_acc": val_final,
        "preds": np.array(val_preds).tolist()  # Convert to list to be pickleable
    }



def main(args):
    number_clauses = int(args.clauses)
    s = int(args.s)
    T = int(args.T)
    weights = bool(args.weights)
    epochs = int(args.epochs)
    X = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/singer/singer_X_ALL_2024-03-16-17-48.npy", mmap_mode="r")
    y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/singer/singer_y_ALL_2024-03-16-17-48.npy", mmap_mode="r")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1066)

   
    manager = Manager()
    result_dict = manager.dict()

    processes = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        p = Process(target=train_fold, args=(train_index, test_index, X, y, number_clauses, T, s, epochs, batch_size, result_dict, fold))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Prepare data for saving
    data_dict = {fold: result_dict[fold] for fold in range(len(result_dict))}
    batch_size = 1000
    train_final = []
    val_final = []
    test_preds_list = []
    for train_index, test_index in kf.split(X, y):
        train_x, val_x = X[train_index], X[test_index]
        train_y, val_y = y[train_index].reshape(-1), y[test_index].reshape(-1)
        train_accuracy_list = []
        val_accuracy_list = []
        for e in range(epochs):
            batched_train(model, train_x, train_y, batch_size)
            train_preds = model.predict(train_x)
            val_preds = model.predict(val_x)

            train_acc = np.mean(train_preds == train_y)
            train_accuracy_list.append(train_acc)
            val_acc = np.mean(val_preds == val_y)
            val_accuracy_list.append(val_acc)
        test_preds_list.append(val_preds)
        train_final.append(train_accuracy_list)
        val_final.append(val_accuracy_list)

    data_dict = {
        "train_acc": train_final,
        "val_acc": val_final,
        "preds": test_preds_list
    }
    pickle_path = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/pickles/singer"
    pickle_file = get_save_path(["all_folds"], pickle_path)
    with open(pickle_file, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TM model")
    parser.add_argument("clauses", help="Number of clauses")
    parser.add_argument("s", help="sensitivity")
    parser.add_argument("T", help="threshold")
    parser.add_argument("weights", help="integer weights")
    parser.add_argument("epochs", help="Number of training epochs")
    args = parser.parse_args()
    main(args)
