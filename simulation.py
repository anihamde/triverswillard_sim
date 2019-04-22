import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *
from model import *

n_gens = 1000
N = 200
v = 0
inter_layer = 10
len_theta = inter_layer*2+inter_layer+inter_layer+1


ms = [(np.random.normal(1, 0.1), copy_w_err(
    [0 for i in range(len_theta)], 0)) for i in range(N//2)]
fs = [(np.random.normal(1, 0.1), copy_w_err(
    [0 for i in range(len_theta)], 0)) for i in range(N//2)]


# Mating score function that skews male distribution
def f_func(w_m, w_f, d_m=2.5, d_f=0.5):
    return (max(w_m, 0)**d_m)*(max(w_f, 0)**d_f)


# Probability function that accounts for mother's fitness
def g_func(w_f, v, theta_f, inter_layer=10):
    theta_f_first = theta_f[:inter_layer*2+inter_layer]
    theta_f_last = theta_f[inter_layer*2+inter_layer:]

    layer0 = np.array([w_f-1, v])  # hacky fix
    layer1 = []

    for i in range(inter_layer):
        layer1.append(layer0[0]*theta_f_first[i])

    for i in range(inter_layer):
        layer1[i] += layer0[1]*theta_f_first[i+inter_layer]

    for i in range(inter_layer):
        layer1[i] += theta_f_first[i+2*inter_layer]

    layer1 = ReLU(layer1)
    layer1 = np.array(layer1)

    layeroutput = np.dot(layer1, theta_f_last[:-1])+(theta_f_last[-1])

    return sigmoid(layeroutput)  # return prob of male


# Initialize the Population Model
model = PopulationModel(ms=ms, fs=fs, N=N, v=v)
model.set_f_func(f_func)
model.set_g_func(g_func)

num_sexes = [[], []]
ratios = []

# Main Loop
for gen in range(n_gens):
    print(len(model.ms), len(model.fs))
    if(len(model.ms) == 0 or len(model.fs) == 0):
        break

    expected_kids = np.array(
        [model.f_func(i[0], j[0]) for i in model.ms for j in model.fs])
    expected_kids += -max(expected_kids)
    expected_kids = np.exp(expected_kids)
    probs = expected_kids / np.sum(expected_kids)
    n_expected_kids = model.N

    next_gen = np.random.multinomial(n_expected_kids, probs).reshape(
        len(model.ms), len(model.fs))

    next_ms = []
    next_fs = []

    print("Females: {}".format(next_gen.sum(axis=0)))  # female
    print("Males: {}".format(next_gen.sum(axis=1)))  # male
    for i in range(len(model.ms)):
        for j in range(len(model.fs)):
            w_f = model.fs[j][0]
            theta_f = model.fs[j][1]
            n_kids = next_gen[i][j]
            if n_kids == 0:
                continue

            w_m = model.ms[i][0]
            theta_m = model.ms[i][1]

            p_male = model.g_func(w_f, model.v, theta_f)
            n_m = int(n_kids * p_male)
            n_f = n_kids-n_m

            for m in range(n_m):
                w_kid = model.inherit_w(w_m, w_f)
                theta_kid = model.inherit_theta(theta_m, theta_f)
                next_ms.append((w_kid, theta_kid))

            for f in range(n_f):
                w_kid = model.inherit_w(w_m, w_f)
                theta_kid = model.inherit_theta(theta_m, theta_f)
                next_fs.append((w_kid, theta_kid))

    num_sexes[0].append(len(next_ms))
    num_sexes[1].append(len(next_fs))
    if len(next_fs) == 0:
        print('No more females!!')
        break
    next_ratio = len(next_ms) / float(len(next_fs))
    ratios.append(next_ratio)
    print('Generation {}: {}'.format(gen, next_ratio))

    model.ms = next_ms
    model.fs = next_fs

num_sexes = np.array(num_sexes)
plt.plot(ratios)
plt.xlabel('Generation')
plt.ylabel('Sex Ratio (M/F)')
# plt.show()
plt.savefig('plot')
