import numpy as np
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
import pycuda.driver as cuda
import time

if __name__ == "__main__":
    cuda.init()
    for i in range(cuda.Device.count()):
        print(cuda.Device(i).name())
    data = MNIST().get()
    tm = TMClassifier(2500,40,5,incremental=True,platform="GPU")
    batch_size = 300
    inc_avg = 0
    for e in range(100):
        start = time.time()
        for i in range(int(60000/300)):
            x_batch = data["x_train"][i*batch_size:(i+1)*batch_size]
            y_batch = data['y_train'][i*batch_size:(i+1)*batch_size]
            tm.fit(x_batch,y_batch,epochs=1)
        preds = tm.predict(data["x_test"])
        result = 100 * (preds == data["y_test"]).mean()
        end = time.time()
        res = end-start
        inc_avg = inc_avg + 1/(e+1)*(res - inc_avg)
    print(f"Finished! Avg time per epoch of MNIST: {inc_avg}")
