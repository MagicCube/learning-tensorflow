import numpy as np


def gen_train_dataSet(count, weight, bias):
    x = np.linspace(-1, 1, count)
    noise = np.random.rand(count) * 0.3
    y = x * weight + bias + noise
    return (x.astype(np.float32), y.astype(np.float32))
