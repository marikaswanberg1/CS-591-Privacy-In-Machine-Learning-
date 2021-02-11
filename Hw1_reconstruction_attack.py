import numpy as np
import random
import matplotlib.pyplot as plt



def generate_data(n):
	'''
	n : int specifying the size of the dataset
	returns a uniformly random bit vector of size n (which will be the data)
	'''
	data = np.random.randint(low=0, high=2, size=n)
	return data

def generate_noise(n):
	'''
	n : int specifying the size of the dataset
	returns a uniformly random bit vector of size n (which will be the random coins)
	'''
	noise = np.random.randint(low=0, high=2, size=n)
	return noise

def generate_queries(data, noise):
	'''
	takes as input equal-length data and noise vectors and
	returns the answer vector where a_i = x_1 + x_2 + ... + x_i + z_i
	'''
	if (len(data) != len(noise)):
		print("length mismatch")
		return 0

	responses = np.zeros((len(data),), dtype=int)
	true_running_sum = 0
	for i in range(0, len(data)):
		true_running_sum += data[i]
		responses[i] = true_running_sum + noise[i]
	return responses

def responses_diffs(responses):
	'''
	Takes the responses as input and outputs a_{i+1} - a_i
	for each index. Can be useful for determining what x_{i+1} is.
	'''
	diff = np.zeros((len(responses),), dtype=int)
	diff[0] = responses[0]
	for i in range(1, len(responses)):
		diff[i] = responses[i] - responses[i-1]
	return diff



def guess_data(responses_diff):
	'''
	The "trivial" attack, which is accurate with probability 3/4.
	'''
	guess = np.zeros((len(responses_diff),), dtype=int)
	for i in range(len(responses_diff)):
		if (-1 == responses_diff[i]):
			guess[i] = 0
		if (2 == responses_diff[i]):
			guess[i] = 1
		if (0 == responses_diff[i]):
			guess[i] = 0
		if (1 == responses_diff[i]):
			guess[i] = 1
	return guess


def guess_data2(responses):
	'''
	A slightly more sophisticated attack. It runs the basic attack for 
	x_i's that we can be certain about. It does a more sophisticated
	analysis of the sums to deduce more values for certain.

	For example: if we've established that the real sum at position i is 5 and 
	the real sum at position i-5 is also 5, then we know that x_{i-4}....x_{i-1} 
	are all 0. 
	'''

	#	initialize with -1's 
	guess = np.zeros((len(responses),), dtype=int)
	guess = np.full_like(guess, -1, dtype=int)
	#	initialize with -1's 
	noise = np.zeros((len(responses),), dtype=int)
	noise = np.full_like(noise, -1, dtype=int)

	#	keep track of the real sums that we can deduce for sure
	real_sums = np.zeros((len(responses),), dtype=int)
	real_sums = np.full_like(real_sums, 2*len(responses), dtype=int)
	real_sum_indices = []

	responses_diff = responses_diffs(responses)

	for i in range(len(responses_diff)):
		if (-1 == responses_diff[i]):
			guess[i] = 0
			noise[i] = 0
			real_sums[i] = responses[i]
			if (i not in real_sum_indices):
				real_sum_indices.append(i)
			if (i >= 1):
				noise[i-1] = 1
				real_sums[i-1] = responses[i-1] - 1
				if (i-1 not in real_sum_indices):
					real_sum_indices.append(i-1)
		if (2 == responses_diff[i]):
			guess[i] = 1
			noise[i] = 1
			real_sums[i] = responses[i] - 1
			if (i not in real_sum_indices):
				real_sum_indices.append(i)
			if (i >= 1):
				noise[i-1] = 0
				real_sums[i-1] = responses[i-1]
				if (i-1 not in real_sum_indices):
					real_sum_indices.append(i-1)

	#	separate case for x_0
	if ((noise[0] == 0) | (noise[0] == 1)):
		guess[0] = responses[0] - noise[0]
		real_sums[0] = responses[0] - noise[0]
		if 0 not in real_sum_indices:
			real_sum_indices.append(0)

	for i in real_sum_indices:
		for j in real_sum_indices:
			if real_sums[i] - real_sums[j] == i - j:
				# all the x's in between are 1 with noise 0
				for k in range(j+1, i):
					guess[k] = 1
					noise[k] = 0
					if (k not in real_sum_indices):
						real_sum_indices.append(k)
						real_sums[k] = responses[k]

			if real_sums[i] - real_sums[j] == 0:
				# all the x's in between are 0 with noise 0
				for k in range(j+1, i):
					guess[k] = 0
					noise[k] = 0
					if (k not in real_sum_indices):
						real_sum_indices.append(k)
						real_sums[k] = responses[k]

	#	run the old attack for the indices we've yet to deduce.
	#	these will (independently) be correct w.p. 3/4
	for i in range(len(guess)):
		if (guess[i] == -1):
			if (0 == responses_diff[i]):
				guess[i] = 0
			if (1 == responses_diff[i]):
				guess[i] = 1

	return guess




def recon_attack(responses):
	'''
	this function runs the attack using responses input only
	'''
	recon = guess_data2(responses)
	return recon

def percent_correct(guess, true):
	'''
	this function returns the percentage of correct bits 
	in the reconstruction attack.
	'''
	diff = abs(guess - true)
	return (len(guess) - sum(diff))/len(diff)



def simulate(length, num_iters):
	'''
	this runs num_iters number of independent simulations on databases
	of length length. 
	It keeps track of the accuracy, mean, and standard deviation
	'''
	accuracy = np.zeros((num_iters,), dtype=float)
	for i in range(num_iters):
		data = generate_data(length)
		noise = generate_noise(length)
		queries = generate_queries(data, noise)
		recon = recon_attack(queries)
		accuracy[i] = percent_correct(data, recon)
	mean = np.mean(accuracy)
	std = np.std(accuracy)
	print("mean :", mean)
	print("std :", std)
	return (accuracy, mean, std)

def plot_data(accuracy, mean, std, length, num_iters):
	'''
	creates a histogram of the accuracies with the mean and standard deviation
	labeled. 
	'''
	plt.title("Accuracy histogram for reconstruction attack \n n = " + str(length) + " Iterations = " + str(num_iters))
	plt.hist(accuracy)
	plt.axvline(x=0.75, color="red", label="0.75 accuracy basic attack")
	plt.axvline(x = mean, color="green", label="non-basic attack accuracy: "+ str(round(mean, 4)) + u"\u00B1" +   str(round(std, 4)))
	plt.xlabel("Accuracy")
	plt.ylabel("Frequency")
	plt.legend()
	plt.show()



data_size = 50000
num_iters = 20

simulation =simulate(data_size, num_iters) 
plot_data(simulation[0], simulation[1], simulation[2], data_size, num_iters)
