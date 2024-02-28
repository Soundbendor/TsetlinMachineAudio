import numpy as np
from sklearn.model_selection import GridSearchCV
from tmu.models.classifier import vanilla_classifier



class TM: # TODO add more hyperparameters
    def __init__(self, clauses=1000,T=10,s=5,state_bits=100):
        self.model = vanilla_classifier(clauses,T,s,number_of_state_bits_ta=state_bits)
    
    def fit(self, X, y):
        self.model.fit(X,y,epochs = 5)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    


if __name__ == "__main__":
    param_grid = {
        "clauses" : [1000,5000,10000], 
        "T" : [10,20,30,40],
        "s" : [5,25]
    }

    train_X = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_1_2024-02-26-16-38.npy")
    train_y = np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_1_2024-02-26-16-38.npy").reshape(-1,)
    model = TM()
    g = GridSearchCV(model,param_grid,n_jobs=-1,cv=2)
    g.fit(train_X,train_y)
    print(g.best_params_)


