import torch
import numpy as np
from tqdm import tqdm
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

#neptune starts here

from timeit import default_timer as timer





if __name__ == "__main__":

    #TODO add pickling of train/test acc per epoch, and final preds for use in f1/precision/recall scores, charts, etc.
    # Also log the trian config files
    

    # Data stuff
    train_x = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/train_X.npy")
    train_y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/train_y.npy")

    #val_x = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/val_X.npy")
    #val_y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/val_y.npy")
 
    num_classes = 5
    number_clauses = 240*num_classes
    s = 6
    T = 17



    model = MultiClassTsetlinMachine(number_clauses, T=T,s=s)

    
    epochs = 150
  

