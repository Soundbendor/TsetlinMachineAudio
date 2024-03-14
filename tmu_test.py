import numpy as np
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
from tqdm import tqdm


if __name__ == "__main__":

    data = MNIST().get()
    tm  =TMClassifier(1000,40,5,incremental=True,platform="GPU")

    batch_size = 300
    for i in tqdm(range(int(60000/300))):
      x_batch = data["x_train"][i*batch_size:(i+1)*batch_size]
      y_batch = data['y_train'][i*batch_size:(i+1)*batch_size]
      tm.fit(x_batch,y_batch,epochs=1)
    preds = tm.predict(data["x_test"])
    result = 100 * (preds == data["y_test"]).mean()

    print("Finished!")