import numpy as np
from tmu.models.classification import vanilla_classifier
import os
import pickle
import datetime
import argparse
from sklearn.metrics import accuracy_score
from multiprocessing import Pool,Process, Manager
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import psutil
from sklearn.metrics import confusion_matrix


def get_process_cpu_count(pid):
    process = psutil.Process(pid)
    return process.cpu_num()


def get_save_path(args, HEAD):
    """Make save path
    """
    date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    suffix = "{}_{}".format(args[0], date)
    result_path = os.path.join(HEAD, suffix)
    return result_path


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

    print(f"fold: {fold_num} Training start.")
    train_y, val_y = train_y.reshape(-1), val_y.reshape(-1)
    train_final = []
    val_final = []
    f1_final = []
    for e in range(epochs):
        batched_train(model, train_x, train_y, batch_size)
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)

        train_acc = accuracy_score(train_y, train_preds)
        val_acc = accuracy_score(val_y, val_preds)
        f1_val = f1_score(val_y, val_preds, average='micro')
        train_final.append(train_acc)
        val_final.append(val_acc)
        f1_final.append(f1_val)
    pid = os.getpid()
    cpu_count = get_process_cpu_count(pid)

    if cpu_count > 1:
        print("Multiple CPUs are being used (Train fold).")
    else:
        print("Only one CPU is being used (Train fold).")
    print(f"fold: {fold_num} Training finished.")
    result_dict[fold_num] = {
        "train_acc": train_final,
        "val_acc": val_final,
        "f1": f1_final
    }

    print(f"fold: {fold_num} results saved.")



def main(args):
    number_clauses = int(args.clauses)
    s = int(args.s)
    T = int(args.T)

    epochs = int(args.epochs)
#/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_all_all_folds_8_bools_2024-03-29-14-23
    with open(
            "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_all_all_folds_8_bools_2024-03-29-14-23",
            'rb') as f:
        data = pickle.load(f)

    real_y_data = data["y"][:, -1]
    y_indices = np.where(data["y"][:, -1] != -1)[0]
    real_y_data = real_y_data[y_indices]
    x_data = data["x"][y_indices]

    y_strat = data["y"][:, -2]
    y_strat = y_strat[y_indices]  # stratify by techniques
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1066)

    batch_size = 1000
    # result_dict = {}
    processes = []
    manager = Manager()
    result_dict = manager.dict()

    for fold, (train_index, test_index) in enumerate(kf.split(x_data, y_strat)):
        print(f"{fold}")
        p = Process(target=train_fold,
                    args=(x_data[train_index], real_y_data[train_index], x_data[test_index], real_y_data[test_index],
                          number_clauses, T, s, epochs, batch_size, result_dict, fold))
        processes.append(p)
        p.start()

    for k in processes:
         k.join()
    #pool = Pool()
    #for fold, (train_index, test_index) in enumerate(kf.split(x_data, y_strat)):
    #    pool.apply_async(train_fold, args=(
    #        x_data[train_index], real_y_data[train_index], x_data[test_index], real_y_data[test_index],
    #        number_clauses, T, s, epochs, batch_size, result_dict, fold))

    #pool.close()
    #pool.join()

    # Prepare data for saving
    data_dict = dict(result_dict)

    pickle_path = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/pickles/singer"
    pickle_file = get_save_path(["singer_8bools"], pickle_path)
    with open(pickle_file, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TM model")
    parser.add_argument("clauses", help="Number of clauses")
    parser.add_argument("s", help="sensitivity")
    parser.add_argument("T", help="threshold")
    parser.add_argument("epochs", help="Number of training epochs")
    args = parser.parse_args()
    main(args)
