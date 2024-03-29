import os
import numpy as np
import pickle

if __name__ == "__main__":
    file_path = "/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_all_all_mfcc_avg_2024-03-29-16-23.pickle"
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    y = data["y"]
    print(f" Classes are: {np.unique(y[:,0])}")
    print(f"Singers are: {np.unique(y[:,1])}")
    print(f"length: {len(y)}")

