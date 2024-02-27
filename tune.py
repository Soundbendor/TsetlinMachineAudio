import numpy as np
from sklearn.model_selection import GridSearchCV
from tmu.models.classifier import vanilla_classifier



class TM: # TODO add more hyperparameters
    def __init__(self, clauses,T,s,state_bits=100):
        self.model = vanilla_classifier(clauses,T,s,number_of_state_bits_ta=state_bits)
    
    def fit(self, X, y):
        self.model.fit(X,y,epochs = 10)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    


if __name__ == "__main__":
    param_grid = {
        "clauses" : [1000,5000,10000] , 
        "T" : [10,20,30,40],
        "s" : [5,25]
    }