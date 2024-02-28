import numpy as np
from tqdm import tqdm
from tmu.models.classification import vanilla_classifier
import os
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import pickle
import datetime

#neptune starts here



def get_save_path(args, HEAD):
    """Make save path
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    suffix = "{}_{}".format(args[0], date)
    result_path = os.path.join(HEAD, suffix)
    return result_path




def main():

    current_directory = os.getcwd()
    with open("config_main.json", 'r') as f:
        config = json.load(f)

    # Data stuff
    
    train_x = np.load(config["train_x"])
    train_y = np.load(config["train_y"]).reshape((-1,))
    assert len(train_y.shape) == 1

    val_x = np.load(config["test_x"])
    val_y = np.load(config["test_y"]).reshape(-1,)
 
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
                                            incremental=True,
                                            seed=1066)
  
    #epochs = config["epochs"]
    epochs = 10
    #train loop
    train_accuracy_list = []
    val_accuracy_list = []
    for e in tqdm(range(epochs)):
        model.fit(train_x,train_y,epochs=1)
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)
        #print(f"predictions of shape: {train_preds.shape}, first element: {train_preds[0]}")
        
        train_acc = np.mean(train_preds == train_y)
        train_accuracy_list.append(train_acc)
        val_acc = np.mean(val_preds == val_y)
        val_accuracy_list.append(val_acc)

    plt.plot(np.arange(epochs),train_accuracy_list,label="Train")
    plt.plot(np.arange(epochs),val_accuracy_list,label="val")
    plt.title("Train and Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/fold_1_DEBUG.png")

    conf_m = np.round(confusion_matrix(train_y,train_preds)/val_y.shape[0], decimals=2)
    print(conf_m)
    
    # Bookkeeping stuff here:
    pickle_path = config["pickle_path"]
    model_path = get_save_path(["TM"],pickle_path)
    train_path = get_save_path(["Train_acc_list"],pickle_path)
    val_path = get_save_path(["Val_acc_list"], pickle_path)

    pickle.dump(model,model_path,"wb")
    pickle.dump(train_accuracy_list,train_path)
    pickle.dump(val_accuracy_list,val_path)
    

if __name__ == "__main__":
    main()

  

