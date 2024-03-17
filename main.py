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

# import neptune
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


# @profile
def main(args):
    number_clauses=int(args.clauses)
    epochs = int(args.epochs)
    s=int(args.s)
    T=int(args.T)
    weights = bool(args.weights)
    config_path = args.config

    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("No config path")

    # Data stuff

    train_x = np.load(config["train_x"], mmap_mode='r')
    train_y = np.load(config["train_y"], mmap_mode='r').reshape((-1,))
    assert len(train_y.shape) == 1

    val_x = np.load(config["test_x"], mmap_mode='r')
    val_y = np.load(config["test_y"], mmap_mode='r').reshape(-1, )

    

    model = vanilla_classifier.TMClassifier(number_clauses,
                                            T=T,
                                            s=s,
                                            number_of_state_bits_ta=100,
                                            incremental=True,
                                            platform='GPU',
                                            weighted_clauses=weights,
                                            seed=1066)

    

    batch_size = 1000
    # train loop
    train_accuracy_list = []
    val_accuracy_list = []
    for e in tqdm(range(epochs)):
        # model.fit(train_x,train_y,epochs=1)
        batched_train(model, train_x, train_y, batch_size)
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)

        train_acc = np.mean(train_preds == train_y)
        train_accuracy_list.append(train_acc)
        val_acc = np.mean(val_preds == val_y)
        val_accuracy_list.append(val_acc)



    # Bookkeeping stuff here:
    pickle_path = config["pickle_path"]
    pickle_file = get_save_path([config["description"]], pickle_path)

    to_pickle = [train_accuracy_list, val_accuracy_list, model]  
    with open(pickle_file, "wb") as f:
        pickle.dump(to_pickle, f)

    # run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TM model")
    parser.add_argument("clauses", help="Number of clauses")
    parser.add_argument("s", help="sensitivity")
    parser.add_argument("T", help="threshold")
    parser.add_argument("weights",help="integer weights")
    parser.add_argument("epochs", help="Number of training epochs")
    parser.add_argument("config", help="config file")
    args = parser.parse_args()
    main(args)
