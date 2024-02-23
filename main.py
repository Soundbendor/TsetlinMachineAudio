import numpy as np
from tqdm import tqdm
from tmu.models.classification import vanilla_classifier
import os
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
#neptune starts here

from timeit import default_timer as timer





if __name__ == "__main__":

    #TODO add pickling of train/test acc per epoch, and final preds for use in f1/precision/recall scores, charts, etc.
    # Also log the trian config files
    current_directory = os.getcwd()
    with open("config_main.json", 'r') as f:
        config = json.load(f)

    # Data stuff
    # TODO rename
    X = np.load(config["train_x"])
    Y = np.load(config["train_y"]).reshape((-1,))
    assert len(train_y.shape) == 1

    # TODO remove after debugging
    bound = int(0.8*X.shape[0])
    train_x = X[:bound]
    val_x = X[bound:]
    by = int(0.8*Y.shape[0])
    train_y = Y[:bound]
    val_y = Y[bound:]
    #val_x = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/val_X.npy")
    #val_y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/val_y.npy")
 
    num_classes = config["num_classes"]
    number_clauses = config["clauses"]
    s = config["s"]
    T = config["T"]
    state_bits = config["state_bits"]
    #integer_weighted = ["weights"]
    #drop_clause = ["drop"]
    # Many more optional parameters
    model = vanilla_classifier.TMClassifier(number_clauses, 
                                            T=T,
                                            s=s,
                                            number_of_state_bits_ta=state_bits,
                                            incremental=True)
  
    #epochs = config["epochs"]
    epochs = 2
    #train loop
    train_accuracy_list = []
    val_accuracy_list = []
    for e in tqdm(range(epochs)):
        model.fit(train_x,train_y,epochs=1,incremental=True)
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)
        print(f"predictions of shape: {train_preds.shape}, first element: {train_preds[0]}")
        
        train_acc = np.mean(train_preds == train_y)
        train_accuracy_list.append(train_acc)
        val_acc = np.mean(val_preds == val_y)
        val_accuracy_list.append(val_acc)

    plt.plot(np.arange(epochs),train_accuracy_list)
    plt.title("Train and Val Accuracy on Small dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/debug_train_acc.png")

    conf_m = confusion_matrix(train_y,train_preds)
    print(conf_m)


  

