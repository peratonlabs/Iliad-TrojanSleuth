# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.





### This is the trojan detector that is submitted to the NIST server. 
### It takes in a model, example images, a scratch file, and output files, and then outputs a probability of the model being trojan

import os
import numpy as np
from scipy import stats
import skimage.io
import random
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.ndimage.filters import gaussian_filter

import warnings 
warnings.filterwarnings("ignore")

def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

	print('model_filepath = {}'.format(model_filepath))
	print('result_filepath = {}'.format(result_filepath))
	print('scratch_dirpath = {}'.format(scratch_dirpath))
	print('examples_dirpath = {}'.format(examples_dirpath))

	# blurring convolution kernel size
	k_size = 9
	# padding size to keep image size the same
	pad = (k_size-1)//2
	# blurring convolution
	conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=k_size, stride=1, padding=pad, bias=False, groups=3)
	# Gaussian weights for Gaussian blurring
	with torch.no_grad():
		conv.weight.data = torch.ones(conv.weight.data.shape)/(k_size**2)
		generated_filters = gaussian_filter(conv.weight.data, sigma=5)
		conv.weight.data.copy_(torch.from_numpy(generated_filters))
	# Load model
	model = torch.load(model_filepath, map_location=torch.device('cuda'))
	conv.to("cuda")
	# Add blurring as a pre-processing step to the CNN
	model = nn.Sequential(conv, model)
	# keep track of how many of each class we've seen
	exemplars = dict()
	# keep track of misclassification concentrations
	delta_probs = dict()
	# keep track of predictions made within each class
	class_guesses = dict()
	image_count = 0
	misclassified = 0
	# Load the example images
	fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
	random.shuffle(fns)
	num_examples = 5
	for fn in range(len(fns)):
		# Inference the example images in data
		ex = fns[fn]
		# get class label
		label = [get_class(ex)]
		img = skimage.io.imread(ex)
		# perform tensor formatting and normalization explicitly
		# convert to CHW dimension ordering
		img = np.transpose(img, (2, 0, 1))
		# convert to NCHW dimension ordering
		img = np.expand_dims(img, 0)
		# normalize the image
		img = img - np.min(img)
		img = img / np.max(img)
		input_label = torch.tensor(label).cuda()
		batch_data = torch.FloatTensor(img).cuda()
		batch_data.requires_grad = True
		# run through the CNN
		logits = model(batch_data)
		sf = nn.Softmax(dim=1)
		result = sf(logits)
		prediction = torch.argmax(logits)
		prediction = prediction.data
		loss_object = torch.nn.CrossEntropyLoss()
		loss = loss_object(logits, input_label)
		model.zero_grad()
		loss.backward()
		# get gradient and signed gradient
		gradient = batch_data.grad
		signed_grad = torch.sign(gradient)
		
		epsilon = 0.3
		# create delta (quasi-trigger)
		delta_x = epsilon * signed_grad
		batch_data = batch_data + delta_x
		logits = model(batch_data)
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
		for j in range(len(fns)):
			ex2 = fns[j]
			label2 = [get_class(ex2)]
			# We want an example from the same class, but a different image
			if label2 != label or ex2 == ex: continue
			img = skimage.io.imread(ex2)
			img = np.transpose(img, (2, 0, 1))
			img = np.expand_dims(img, 0)
			img = img - np.min(img)
			img = img / np.max(img)
			# add delta
			batch_data = torch.FloatTensor(img).cuda()+delta_x
			logits = model(batch_data)
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
	# max_p is the maximum of these misclassification concentrations
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
	# maximum average misclassification concentration rate for class-wide delta
	max_mc = np.max(means)
	# quartiles corresponding to the maximum average misclassification concentration
	q3 = quant3[means.index(max(means))]
	q1 = quant1[means.index(max(means))]
	# misclassification rate for all images
	mis_rate = misclassified/image_count

	# Logistic regression weights
	params = -1 * (-1.632+ 0.793*max_mc - 0.728*max_p +  0.354*q3 + 1.219*mis_rate + 1.197*q1)

	# calculate probability of being trojan
	trojan_probability = 1/(1+(math.e**params))

	print('Trojan Probability: {}'.format(trojan_probability))

	with open(result_filepath, 'w') as fh:
		fh.write("{}".format(trojan_probability))

# Get class label
def get_class(image):
	i = image.index("class")
	if image[i+7:i+8].isnumeric():
		label = image[i+6:i+8]
	else:
		label = image[i+6:i+7]
	return int(label)

# Run
if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
	parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
	parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
	parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
	parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')


	args = parser.parse_args()
	trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


