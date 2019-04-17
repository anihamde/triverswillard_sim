import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_gens = 1000
N = 1000
v = 0
inter_layer = 10

ms = [(0,0) for i in range(N/2)]
fs = [(0,0) for i in range(N/2)]



def sigmoid(val):
	return(1./(1.+np.exp(-val)))

def copy_w_err(bitarr,prob=0.01):
	out = bitarr
	for i in range(len(out)):
		if np.random.binomial(1,prob) == 1:
			out[i] = 1-out[i]

	return out

def f(w_m,w_f,d=2): # need a mating score function that skews male distribution
	return (w_m**d)*w_f

def g(w_f,v,theta_f,inter_layer=10): # need a prob function that accounts for mother's fitness
	theta_f_first = theta_f[:inter_layer*2+inter_layer]
	theta_f_last = theta_f[inter_layer*2+inter_layer:]

	layer0 = np.array([w_f,v])
	layer1 = []

	for i in range( inter_layer ):
		layer1.append( layer0[0]*theta_f_first[i] )

	for i in range( inter_layer ):
		layer1[i] += layer0[1]*theta_f_first[i+inter_layer]

	for i in range( inter_layer ):
		layer1[i] += theta_f_first[i+2*inter_layer]

	layer1 = np.array(layer1)

	layeroutput = np.dot(np.repeat(layer1,2),theta_f_last[:-1])+theta_f_last[-1]

	return sigmoid(layeroutput) # return prob of male

def inherit_w(w_m,w_f):
	if type(w_f) == int:
		if np.random.binomial(1,0.5) == 1:
			return np.random.normal(w_m,0.1)
		else:
			return np.random.normal(w_f,0.1)
	else:
		w_k = []

		for i in range(len(w_m)):
			if np.random.binomial(1,0.5) == 1:
				w_k.append(w_m[i])
			else:
				w_k.append(w_f[i])

		return copy_w_err(w_k)

def inherit_theta(theta_m,theta_f):
	theta_k = []

	for i in range(len(theta_m)):
		if np.random.binomial(1,0.5) == 1:
			theta_k.append(theta_m[i])
		else:
			theta_k.append(theta_f[i])

	return copy_w_err(theta_k)

num_sexes = []

for gen in range(n_gens):
	expected_kids = np.array([f(i[0],j[0]) for i in ms for j in fs])
	expected_kids = np.exp(expected_kids)
	n_expected_kids = np.sum(expected_kids)
	probs = expected_kids/n_expected_kids

	next_gen = np.random.multinomial(n_expected_kids,probs).reshape(len(ms),len(fs))

	next_ms = []
	next_fs = []

	for i in range(len(ms)):
		for j in range(len(fs)):
			w_f = fs[j][0]
			theta_f = fs[j][1]
			n_kids = next_gen[i][j]

			theta_m = fs[i][1]

			p_male = g(w_f,v,theta_f)
			n_m = np.random.binomial(n_kids,p_male)
			n_f = n_kids-n_m

			for m in range(n_m):
				w_kid = inherit_w(w_f)
				theta_kid = inherit_theta(theta_m,theta_f)
				next_ms.append((w_kid,theta_kid))

			for f in range(n_f):
				w_kid = inherit_w(w_f)
				theta_kid = inherit_theta(theta_m,theta_f)
				next_fs.append((w_kid,theta_kid))

	num_sexes.append(len(next_ms),len(next_fs))

plt.plot(num_sexes[0]/num_sexes[1])
plt.xlabel('Generation')
plt.ylabel('Sex Ratio (M/F)')
plt.show()



