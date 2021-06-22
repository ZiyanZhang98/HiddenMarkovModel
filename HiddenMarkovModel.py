import numpy as np


def forward_algorithm(data, transition_prob, emission_prob, initial_distribution):
    p = np.zero(data.shape[0], transition_prob[0])
    b = emission_prob[:data.shape[0]]
    p = initial_distribution * b
    for t in range(1, data.shape[0]):
        for i in range(0, transition_prob.shape[0]):
            p[t, i] = p[t - 1].dot(p[:, i], b[i, t])
    return p

