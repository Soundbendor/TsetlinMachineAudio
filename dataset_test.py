
from sklearn.metrics import f1_score
import numpy as np
from tmu.models.classifiers.vanilla_classifier import TMClassifier



def batched_train(model, X, y, batch_size, epochs=1):
    array_size = len(X)
    for i in range(0, array_size, batch_size):
        model.fit(X[i:i + batch_size], y[i:i + batch_size], epochs=epochs)



def main():
    # Load datasets
    datasets = [ 
        {
            "x_train" : np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_fold_1_2024-03-14-17-22.npy"),
            "x_test" : np.load("/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_X_test_fold_1_2024-03-14-17-26.npy"),
            'y_train' : np.load('/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_fold_1_2024-03-14-17-22.npy').reshape(-1,),
            "y_test" : np.load('/nfs/guille/eecs_research/soundbendor/mccabepe/VocalSet/npy_files/vowel/vowel_y_test_fold_1_2024-03-14-17-26.npy').reshape(-1,)
        }
    ]
    labels = ["2", "4", "8"]

    f1_scores = []
    # Iterate over datasets
    for dataset in datasets:
        model = TMClassifier(2500,
                        T=200,
                        s=5,
                        number_of_state_bits_ta=100,
                        incremental=True,
                        platform='GPU',
                        weighted_clauses=False,
                        seed=1066)
        batched_train(model,dataset["x_train"],dataset["y_train"],1000,10)
        y_preds = model.predict(dataset["x_test"])
        f1_score = f1_score(dataset["y_test"], average='micro')
        f1_scores.append(f1_score)

    for i in range(len(f1_scores)):
        print(f"Dataset: {labels[i]}, f1 score: {f1_scores[i]}")
        

        