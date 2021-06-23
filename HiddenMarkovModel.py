import numpy as np


def forward_algorithm(data, transition_prob, emission_prob, initial_distribution):
    p = np.zero(data.shape[0], transition_prob[0])
    b = emission_prob[:data.shape[0]]
    p = initial_distribution * b
    for t in range(1, data.shape[0]):
        for i in range(transition_prob.shape[0]):
            p[t, i] = p[t - 1].dot(p[:, i], b[i, t])
    return p


def viterbi_algorithm(data, transition_prob, emission_prob, initial_distribution):
    p = np.zeros((data.shap[0], transition_prob[0]))
    p[0, :] = np.log(initial_distribution * emission_prob[:, data[0]])
    previous = np.zeros((data.shape[0] - 1, transition_prob.shape[0]))
    for t in range(1, data.shape[0]):
        for j in range(transition_prob.shape[0]):
            probability = p[t - 1] + np.log(transition_prob[:, j]) + np.log(emission_prob[j, data[t]])
            previous[t - 1, j] = np.argmax(probability)
            p[t, j] = np.max(probability)

    path = np.zeros(data.shape[0])
    last = np.argmax(p[data.shape[0] - 1, :])
    path[0] = last

    step = 1
    for i in range(data.shape[0] - 2, -1, -1):
        path[step] = previous[i, int(last)]
        last = path[step]
        step = step + 1

    path = np.flip(path)
    return path
