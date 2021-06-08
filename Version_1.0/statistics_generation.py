
### This file is used to generate statistics from the training data, to train our binary classifier on
### These statistics include different variants of misclassification concentration and misclassification rates as described in README.txt


import os
import numpy as np
import skimage.io
import random
from scipy import stats
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import copy
from scipy.ndimage.filters import gaussian_filter
import time

def get_model(num_examples=5, kernel_size=9, sigma=5, eps=0.3, end=400):
	# path of training examples
	path = "round3-dataset/"
	# file to save data in
	f = open('classification_stats.csv','wb')
	content = np.empty((0,6))
	# blurring convolution kernel size
	k_size = kernel_size
	# padding size to keep image size the same
	pad = (k_size-1)//2
	# blurring convolution
	conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=k_size, stride=1, padding=pad, bias=False, groups=3)
	# Gaussian weights for Gaussian blurring
	with torch.no_grad():
		conv.weight.data = torch.ones(conv.weight.data.shape)/(k_size**2)
		generated_filters = gaussian_filter(conv.weight.data, sigma=sigma)
		conv.weight.data.copy_(torch.from_numpy(generated_filters))
	# Iterate through training models
	for i, folder in enumerate(sorted(os.listdir(path))):
		if i%50 == 0: print(folder)
		# Load model, images, and label
		model_path = path+folder+"/model.pt"
		model = torch.load(model_path, map_location=torch.device('cuda'))
		example_path = path+folder+"/clean_example_data/"
		exs = [os.path.join(example_path, ex) for ex in sorted(os.listdir(example_path)) if ex.endswith(".png")]
		real_label = np.loadtxt(path+folder+"/ground_truth.csv", dtype=bool)
		conv.to("cuda")
		# Add blurring as a pre-processing step to the CNN
		model_blur = nn.Sequential(conv, model)
		if i < end:
			# keep track of how many of each class we've seen
			exemplars = dict()
			# keep track of misclassification concentrations
			delta_probs = dict()
			# keep track of predictions made for each class
			class_guesses = dict()
			random.shuffle(exs)
			image_count = 0
			misclassified = 0
			for k in range(len(exs)):
				# load the example image
				ex = exs[k]
				# get class label
				label = [get_class(ex)]
				img = skimage.io.imread(ex)
				# convert to NCHW ordering
				img = np.transpose(img, (2, 0, 1))
				img = np.expand_dims(img, 0)
				# normalize the image
				img = img - np.min(img)
				img = img / np.max(img)
				input_label = torch.tensor(label).cuda()
				batch_data = torch.FloatTensor(img).cuda()
				batch_data.requires_grad = True
				# run image through CNN
				logits = model_blur(batch_data)
				sf = nn.Softmax(dim=1)
				result = sf(logits)
				prediction = torch.argmax(logits)
				prediction = prediction.data
				loss_object = torch.nn.CrossEntropyLoss()
				loss = loss_object(logits, input_label)
				model_blur.zero_grad()
				loss.backward()
				# get gradient
				gradient = batch_data.grad
				# get signed gradient
				signed_grad = torch.sign(gradient)
				
				epsilon = eps
				# create delta (quasi-trigger)
				delta_x = epsilon * signed_grad
				batch_data = batch_data + delta_x
				logits = model_blur(batch_data)
				result = sf(logits)
				prediction = torch.argmax(logits)
				# count misclassifications for image-specific deltas
				if prediction != input_label[0]:
					misclassified += 1
				image_count += 1

				# keep track of predictions made within each class
				if label[0] in class_guesses:
					class_guesses[label[0]].append(prediction)
				if label[0] not in class_guesses:
					class_guesses[label[0]] = [prediction]

				# only proceed with a set number of examples per class
				if label[0] in exemplars:
					if exemplars[label[0]] >= num_examples:
						continue
					else:
						exemplars[label[0]] = exemplars[label[0]] + 1
				if label[0] not in exemplars: exemplars[label[0]] = 1

				# See how a delta affects classifications of other images within the class
				guesses = []
				for j in range(len(exs)):
					ex2 = exs[j]
					label2 = [get_class(ex2)]
					# We want an example from the same class, but a different image
					if label2 != label or ex2==ex: continue
					img = skimage.io.imread(ex2)
					img = np.transpose(img, (2, 0, 1))
					img = np.expand_dims(img, 0)
					img = img - np.min(img)
					img = img / np.max(img)
					# add delta
					batch_data = torch.FloatTensor(img).cuda()+delta_x
					logits = model_blur(batch_data)
					guess = torch.argmax(logits)
					# save guess
					guesses.append(guess)
				# Find most common misclassification
				try:
					mode = stats.mode(np.array(guesses)[np.array(guesses) != label[0]])[0][0]
					mode_count = guesses.count(mode)
					freq = mode_count
				except:
					freq = 0
				# Calculate and save misclassification concentration
				p = freq/len(guesses)
				if label[0] not in delta_probs:
					delta_probs[label[0]] = [p]
				else:
					delta_probs[label[0]].append(p)

			# Calculate misclassification concentration regarding image-specific deltas
			# max_p is the maximum misclassification concentration
			max_p = 0
			for class_ in class_guesses:
				guesses = class_guesses[class_]
				try:
					mode = stats.mode(np.array(guesses)[np.array(guesses) != label[0]])[0][0]
					mode_count = guesses.count(mode)
					freq = mode_count
				except:
					freq = 0
				p = freq/len(guesses)
				if p > max_p:
					max_p = p
			means = []
			quant1 = []
			quant3 = []
			for prob_list in delta_probs:
				# Calculate the average miclassification concentration for each class over the 5 class-wide deltas, as well as the 3rd and 1st quartiles for these classes
				p = np.mean(delta_probs[prob_list])
				means.append(p)
				quant3.append(np.quantile(delta_probs[prob_list], 0.75))
				quant1.append(np.quantile(delta_probs[prob_list], 0.25))
			# misclassification rate for all images
			mis_rate = misclassified/image_count
			# Bundle the stats in an array for saving to the output csv
			statistics = np.array([np.max(means), max_p, quant3[means.index(max(means))],mis_rate,quant1[means.index(max(means))], real_label])
			content = np.append(content, np.expand_dims(statistics,axis=0), axis=0)
	np.savetxt(f, content)

# Get class from an example image
def get_class(image):
	i = image.index("class")
	if image[i+7:i+8].isnumeric():
		label = image[i+6:i+8]
	else:
		label = image[i+6:i+7]
	return int(label)

# Run statistic generation
if __name__ == "__main__":
	get_model(num_examples=5, eps=0.3)


