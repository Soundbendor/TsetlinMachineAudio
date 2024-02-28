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
import neptune


def get_save_path(args, HEAD):
    """Make save path
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    suffix = "{}_{}".format(args[0], date)
    result_path = os.path.join(HEAD, suffix)
    return result_path




def main(params: dict, config_path=None):

    run = neptune.init_run(
        project="mccabepe/TMAudio",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhM2FhZjQ3Yy02NmMxLTRjNzMtYjMzZC05YjM2N2FjOTgyMTEifQ==",
        mode="sync"
    ) 

    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config={ 
                    "train_x" : "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_1_2024-02-26-16-38.npy",
                    "test_x" : "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_1_2024-02-26-16-49.npy",
                    "train_y" : "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_1_2024-02-26-16-38.npy",
                    "test_y": "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_1_2024-02-26-16-49.npy",
                    "pickle_path": "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/pickles/Vowels/"          
        }
    # Data stuff
    
    train_x = np.load(config["train_x"])
    train_y = np.load(config["train_y"]).reshape((-1,))
    assert len(train_y.shape) == 1

    val_x = np.load(config["test_x"])
    val_y = np.load(config["test_y"]).reshape(-1,)
 
  
    number_clauses = params["clauses"]
    s = params["s"]
    T = params["T"]
    state_bits = params["state_bits"]
    #integer_weighted = params["weights"]
    #drop_clause = params["drop"]
    # Many more optional parameters
 
    run["parameters"] = params

    model = vanilla_classifier.TMClassifier(number_clauses, 
                                            T=T,
                                            s=s,
                                            number_of_state_bits_ta=state_bits,
                                            incremental=True,
                                            seed=1066)
  
    epochs = params["epochs"]
    #epochs = 10
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
    plt.savefig("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/Misc_files/vowel_fold_1_tune.png")

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
    
    run["train/acc"] = train_accuracy_list
    run["test/acc"] = val_accuracy_list
    run.stop()

if __name__ == "__main__":
    #clauses = [1000,5000,10000]
    #Ts = [10,20,30,40]
    #ss = [5, 10, 25]
    epochs=1
    params = {"clauses": 1000,
              "T": 10,
              "s":5,
              "epochs":epochs
              }
    main(params)

  

