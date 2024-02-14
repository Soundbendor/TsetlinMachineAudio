import torch
import numpy as np
from torch import nn
from torch.nn import functional as torchf
from VocalSetDataset import VocalSetDataset
from tqdm import tqdm
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

from timeit import default_timer as timer





if __name__ == "__main__":

    #TODO add pickling of train/test acc per epoch, and final preds for use in f1/precision/recall scores, charts, etc.
    # Also log the trian config files
    
 
    num_classes = 5
    number_clauses = 240*num_classes
    s = 6
    T = 17

    train_x = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/train_X.npy")
    train_y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/train_y.npy")

    val_x = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/val_X.npy")
    val_y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/val_y.npy")

    model = MultiClassTsetlinMachine(number_clauses, T=T,s=s)
    chunk_size = 32
    
    epochs = 150
   
    
    for i in tqdm(range(epochs)):
        if (i+1) * chunk_size <= train_x.shape[0]:
            model.fit(train_x[i*chunk_size:(i+1)*chunk_size],train_y[i*chunk_size:(i+1)*chunk_size].reshape(-1),epochs=1, incremental=True)
        else:
            model.fit(train_x[i*chunk_size:],train_y[i*chunk_size:].reshape(-1),epochs=1, incremental=True)
    preds = model.predict(val_x)
    print("Accuracy:", 100*(preds == val_y.reshape(-1)).mean())

