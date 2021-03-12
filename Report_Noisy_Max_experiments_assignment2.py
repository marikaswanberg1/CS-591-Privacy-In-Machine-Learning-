import random
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt


def gen_gaussian_data(R, n):
	return np.random.normal(loc=R/4, scale=(R**2)/10, size=n)

def gen_poisson_data(n):
	return np.random.poisson(50, n)

def gen_bimodal_data(n, R, k):
	return 2*k*np.random.randint(low=0, high=2, size=n) + (R-k)

def report_noisy_max(data, epsilon, R, reps):
	'''
	This function takes in a dataset, an epsilon privacy parameter, and 
	the set of possible values the entries of the data can take.
	It returns the noisy median of the data using RNM.
	'''
	int_data = np.clip(np.rint(data), 1, R) # need to round to integers in [R]
	
	utility = np.zeros(R)
	counts = collections.Counter(int_data)

	#	this computes the same utility function as compute_median_utility
	#	but just faster.
	running_sum = -len(data)
	for y in range(R):
		utility[y] = -abs(running_sum +  counts[y] + counts[y+1])
		running_sum += counts[y] + counts[y+1]

		
	#	we can use the same exact utility values
	#	and add noise to them reps number of times to save time in our simulations
	noisy_utilities = np.empty(reps)
	RNM = np.empty(reps)
	for i in range(reps):
		noise = np.random.exponential(2/epsilon, R)
		noisy_max = np.argmax(noise + utility) + 1
		RNM[i] = noisy_max
		noisy_utilities[i] = utility[int(noisy_max)-1] # get the rank of the noisy medians

	# return the utility of the true median for the error analysis
	true_utility = np.max(utility)

	return (RNM, true_utility, noisy_utilities)


def compute_median_utility(data, y):
	'''
	This function computes the utility of a given choice of median
	for data that comes from the set [R]
	'''
	summation = 0
	for x in data:
		summation += np.sign(y-x)
		
	return -abs(summation)

def gaussian_experiment(n, R, epsilon=0.1):
	'''
	n : int size of each dataset
	'''
	error = np.array(0) # stores total error over all datasets
	in_dataset_error = np.array(0) # stores average within-dataset error
	#	Generate 50 datasets
	for i in range(50):
		RNM = np.empty(10)
		data = gen_gaussian_data(R, n)
		est, true_utility, noisy_utilities = report_noisy_max(data, epsilon, R, 10) #	run 10 iterations of RNM on this dataset 
		data_error = abs(noisy_utilities - true_utility) #	compare the relative ranks of the truth and estimate
		error = np.append(error, data_error)
		in_dataset_error = np.append(in_dataset_error, np.sum(data_error)/10)
		
	return (np.average(error), np.std(error), np.std(in_dataset_error))

def poisson_experiment(n, R, epsilon=0.1):
	'''
	n : int size of each dataset
	'''
	error = np.array(0) # stores total error over all datasets
	in_dataset_error = np.array(0) # stores average within-dataset error
	#	Generate 50 datasets
	for i in range(50):
		RNM = np.empty(10)
		data = gen_poisson_data(n)
		est, true_utility, noisy_utilities = report_noisy_max(data, epsilon, R, 10) #	run 10 iterations of RNM on this dataset 
		data_error = abs(noisy_utilities - true_utility)
		error = np.append(error, data_error)
		in_dataset_error = np.append(in_dataset_error, np.sum(data_error)/10)

	
	return (np.average(error), np.std(error), np.std(in_dataset_error))

def bimodal_experiment(n, R, k, epsilon=0.1):
	'''
	n : int size of each dataset
	'''
	error = np.array(0) # stores total error over all datasets
	in_dataset_error = np.array(0) # stores average within-dataset error
	#	Generate 50 datasets
	for i in range(50):
		RNM = np.empty(10)
		data = gen_bimodal_data(n, R, k)
		est, true_utility, noisy_utilities = report_noisy_max(data, epsilon, R, 10) #	run 10 iterations of RNM on this dataset 
		data_error = abs(noisy_utilities - true_utility) 
		error = np.append(error, data_error)
		in_dataset_error = np.append(in_dataset_error, np.sum(data_error)/10)

	
	return (np.average(error), np.std(error), np.std(in_dataset_error))


def gaussian_experiment_runs():
	'''
	This function runs the specific experiments for the homework
	'''
	N = [50, 100, 500, 2000, 10000]
	R = [100, 1000, 10000]
	avg_error = []
	std_error = []
	std_dev_dataset = []

	for n in N:
		print("n =", n)
		avg_error_n = []
		std_error_n = []
		std_dev_dataset_n = []
		for r in R:
			avg, std1, std2 = gaussian_experiment(n, r, epsilon=0.1)
			avg_error_n.append(avg)
			std_error_n.append(std1)
			std_dev_dataset_n.append(std2)
			print(str(r) + " & " + str(np.around(avg, decimals=3)) + " & " + str(np.around(std1, decimals=3)) + " & " + str(np.around(std2, decimals=3)))
		avg_error.append(avg_error_n)
		std_error.append(std_error_n)
		std_dev_dataset.append(std_dev_dataset_n)
	return (avg_error, std_error, std_dev_dataset)

def bimodal_experiment_runs():
	N = [50, 100, 500, 2000, 10000]
	R = 1000
	K = [10,100,200]

	avg_error = []
	std_error = []
	std_dev_dataset = []
	for n in N:
		print("n =", n)
		avg_error_n = []
		std_error_n = []
		std_dev_dataset_n = []
		for k in K:
			avg, std1, std2 = bimodal_experiment(n, R, k, epsilon=0.1)
			avg_error_n.append(avg)
			std_error_n.append(std1)
			std_dev_dataset_n.append(std2)
			#print(str(k) + " & " + str(np.around(avg, decimals=3)) + " & " + str(np.around(std1, decimals=3)) + " & " + str(np.around(std2, decimals=3)) + "\\\'")
			#print("\\hline")
		avg_error.append(avg_error_n)
		std_error.append(std_error_n)
		std_dev_dataset.append(std_dev_dataset_n)
	return (avg_error, std_error, std_dev_dataset)

def poisson_experiment_runs():
	N = [50, 100, 500, 2000, 10000]
	R = [100, 1000, 10000]
	avg_error = []
	std_error = []
	std_dev_dataset = []
	
	for n in N:
		print("n =", n)
		avg_error_n = []
		std_error_n = []
		std_dev_dataset_n = []
		for r in R:
			avg, std1, std2 = poisson_experiment(n, r, epsilon=0.1)
			# print(avg, std1, std2)
			avg_error_n.append(avg)
			std_error_n.append(std1)
			std_dev_dataset_n.append(std2)

			#print(str(r) + " & " + str(np.around(avg, decimals=3)) + " & " + str(np.around(std1, decimals=3)) + " & " + str(np.around(std2, decimals=3)) + "\\\'")
			#print("\\hline")
		avg_error.append(avg_error_n)
		std_error.append(std_error_n)
		std_dev_dataset.append(std_dev_dataset_n)
	return (avg_error, std_error, std_dev_dataset)



def plot_heatmap(data, error):
	'''
	Code adapted from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
	Creates a heatmap of the data values with error annotated on each square
	'''

	N = ["50" , "100", "500", "2000", "10000"]
	R = ["100", "1,000", "10,000"]
	K = ["10" , "100" ,"200"]

	data = np.around(np.array(data), decimals=3)
	error = np.around(np.array(error), decimals=3)

	fig, ax = plt.subplots()
	im = ax.imshow(data)

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(K)))
	ax.set_yticks(np.arange(len(N)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(R)
	ax.set_yticklabels(N)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	plt.xlabel("R")
	plt.ylabel("Dataset size, n")

	# Loop over data dimensions and create text annotations.
	for i in range(len(N)):
	    for j in range(len(R)):
	        text = ax.text(j, i, str(data[i, j]) + "\n" + str(u"\u00B1") + str(error[i,j]),
	                       ha="center", va="center", color="w")

	ax.set_title("RNM: Average Error in Rank for Data ~ Poi(50)")
	fig.tight_layout()
	plt.show()
