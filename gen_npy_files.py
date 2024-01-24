import numpy as np
from VocalSetDataset import VocalSetDataset
from sklearn.preprocessing import KBinsDiscretizer
from torchaudio.transforms import MFCC
from torch import nn



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

    num_classes = 5
    sample_rate = 44100
    n_mfcc = 13
    melkwargs_dict = {"n_fft": 800, 
                    "win_length": 320,
                    "hop_length" : 160,
                    "n_mels" : 75,
                    "center" : False}
    num_quantiles = 2
    boolean_encoding = "onehot-dense"


    mfcc_func = MFCC(sample_rate=sample_rate,n_mfcc=n_mfcc,melkwargs=melkwargs_dict)
    booleanizer = KBinsDiscretizer(n_bins=num_quantiles,encode=boolean_encoding)

    transform_func = TransformFunc(trimpad,mfcc_func=mfcc_func,boolean_func=booleanizer.fit_transform)


    #train_data = VocalSetDataset("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/vocalset/train/annotations_train.txt",transform=transform_func)
    data = VocalSetDataset("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/vocalset/train/annotations_train.txt",transform=transform_func)

    x_file_path = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/train_X.npy"
    y_file_path = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/train_y.npy"
    #test_file = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/testscrap.npy"
    X_list = []
    y_list = []
    #t_list = []
    for i in range(len(data)):
        X,y = data[i]
        X_list.append(X.astype(np.bool_))
        y_list.append(y.astype(np.bool_))
        #t_list.append(y)

    #t_mat = np.vstack(t_list)
    x_mat = np.vstack(X_list)
    y_mat = np.vstack(y_list)
    #np.save(test_file,t_mat)
    np.save(x_file_path,x_mat)
    np.save(y_file_path,y_mat)
        
