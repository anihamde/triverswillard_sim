import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_gens = 1000
N = 100
v = 0
inter_layer = 10
len_theta = inter_layer*2+inter_layer+inter_layer+1

def sigmoid(val):
	return(1./(1.+np.exp(-val)))

def ReLU(arr):
	return [max(0,val) for val in arr]

def copy_w_err(bitarr,prob=0.01):
	out = bitarr
	for i in range(len(out)):
		if np.random.binomial(1,prob) == 1:
			out[i] = 1-out[i]

	return out

ms = [(np.random.normal(1,0.1),copy_w_err([0 for i in range(len_theta)]) ) for i in range(N//2)]
fs = [(np.random.normal(1,0.1),copy_w_err([0 for i in range(len_theta)]) ) for i in range(N//2)]

def f_func(w_m,w_f,d_m=2,d_f=0.5): # need a mating score function that skews male distribution
	return (w_m**d_m)*(w_f**d_f)

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

	layer1 = ReLU(layer1)
	layer1 = np.array(layer1)

	layeroutput = np.dot(layer1,theta_f_last[:-1])+theta_f_last[-1]

	return sigmoid(layeroutput) # return prob of male

def inherit_w(w_m,w_f):
	if type(w_f) == float:
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

num_sexes = [[],[]]

for gen in range(n_gens):
	print(len(ms),len(fs))
	expected_kids = np.array([ f_func(i[0],j[0]) for i in ms for j in fs ])
	expected_kids = np.exp(expected_kids)
	probs = expected_kids/np.sum(expected_kids)
	n_expected_kids = 4*N*len(ms)*len(fs)/(len(ms)+len(fs))**2 # super hacky fix
	print(n_expected_kids)

	next_gen = np.random.multinomial(n_expected_kids,probs).reshape(len(ms),len(fs))

	next_ms = []
	next_fs = []

	for i in range(len(ms)):
		for j in range(len(fs)):
			w_f = fs[j][0]
			theta_f = fs[j][1]
			n_kids = next_gen[i][j]

			w_m = ms[i][0]
			theta_m = ms[i][1]

			p_male = g(w_f,v,theta_f)
			n_m = np.random.binomial(n_kids,p_male)
			n_f = n_kids-n_m

			for m in range(n_m):
				w_kid = inherit_w(w_m,w_f)
				theta_kid = inherit_theta(theta_m,theta_f)
				next_ms.append((w_kid,theta_kid))

			for f in range(n_f):
				w_kid = inherit_w(w_m,w_f)
				theta_kid = inherit_theta(theta_m,theta_f)
				next_fs.append((w_kid,theta_kid))

	num_sexes[0].append( len(next_ms) )
	num_sexes[1].append( len(next_fs) )
	print('Generation {}: {}'.format(gen, len(next_ms)/len(next_fs)) )

	ms = next_ms
	fs = next_fs

num_sexes = np.array(num_sexes)
plt.plot(num_sexes[0]/num_sexes[1])
plt.xlabel('Generation')
plt.ylabel('Sex Ratio (M/F)')
plt.show()



