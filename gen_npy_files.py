import numpy as np
from VocalSetDataset import VocalSetDataset
from sklearn.preprocessing import KBinsDiscretizer
from torchaudio.transforms import MFCC
from torch import nn
import json


#TODO add CV support

def trimpad(data):
    duration = data.shape[1]
    if duration < 88200:
        # Pad
        padded_sample = nn.functional.pad(data,(0,88200-duration))
        return padded_sample
    elif duration > 88200:
        trimmed_sample = data[:,:88200]
        return trimmed_sample
    else:
        return data



class TransformFunc:
    def __init__(self,pad_func,mfcc_func,boolean_func):
        self.trimpad = pad_func
        self.mfcc_gen = mfcc_func
        self.booleanizer = boolean_func
    
    
    def __call__(self, data):
       
        #timein = timer()
        bool_array = self.booleanizer(self.mfcc_gen(self.trimpad(data).squeeze()))
        #timeout = timer()
               
        #print(f"time: {timeout-timein}, memory: ", (bool_array.itemsize * bool_array.shape[0]*bool_array.shape[1]), " bytes.") # 113776 
        #print(f"sparse matrix uses {bool_array.data.nbytes + bool_array.indptr.nbytes + bool_array.indices.nbytes} bytes of memory") #85388
        return bool_array




if __name__ == "__main__":

    with open("config_npy.json", 'r') as f:
        config = json.load(f)

    sample_rate = config["sample_rate"]
    n_mfcc = config["n_mfcc"]
    melkwargs_dict = config["melkwargs_dict"]
    num_quantiles = config["num_quantiles"]
    boolean_encoding = config["boolean_encoding"]

    DATA_PATH = config["data_path"]

    mfcc_func = MFCC(sample_rate=sample_rate,n_mfcc=n_mfcc,melkwargs=melkwargs_dict)
    booleanizer = KBinsDiscretizer(n_bins=num_quantiles,encode=boolean_encoding)

    transform_func = TransformFunc(trimpad,mfcc_func=mfcc_func,boolean_func=booleanizer.fit_transform)

    data = VocalSetDataset(DATA_PATH,transform=transform_func)

    x_file_path = config["train_path"]
    y_file_path = config["test_path"]
    
    X_list = []
    y_list = []
  
    for i in range(len(data)):
        X,y = data[i]
        #save X_list as bool? It just get used as int32 anyway?
        X_list.append(X.astype(np.bool_))
    
      

    
    x_mat = np.vstack(X_list)
    y_mat = np.vstack(y_list)
   
    np.save(x_file_path,x_mat)
    np.save(y_file_path,y_mat)
        
