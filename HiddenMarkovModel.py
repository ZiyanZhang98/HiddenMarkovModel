import numpy as np
import pandas as pd


def foward(data, hidden_prob, visible_prob, init):
    # 需要混淆矩阵，CNM Windows是真难用
    prediction = np.zeros(hidden_prob.shape[0], data.shape[0])
    for i in range(1, hidden_prob.shape[0]):
        for j in range(visible_prob.shape[0]):
            prediction[i, j] = prediction[i - 1].dot(hidden_prob[:, j]) * visible_prob[j, data[i]]
