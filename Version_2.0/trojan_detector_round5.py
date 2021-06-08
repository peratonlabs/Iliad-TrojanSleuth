import numpy as np
import torch
import os
from joblib import load
import copy
import json
import random

import warnings
warnings.filterwarnings("ignore")


def get_model(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath, num_examples=5, num_perturb=3, noise_mag=1.0):
	# Load model
	model_pt = torch.load(model_filepath, map_location=torch.device('cuda'))

	# Load examples
	fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
	random.shuffle(fns)
	# Check CUDA
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Load tokenizer
	tokenizer = torch.load(tokenizer_filepath)

	# set the padding token if its undefined
	if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	# load the specified embedding]
	embedding = torch.load(embedding_filepath, map_location=torch.device(device))

	# identify the max sequence length for the given embedding
	max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

	use_amp = True  # attempt to use mixed precision to accelerate embedding conversion process

	exemplars = dict()
	gradient_list = []
	loss_object = torch.nn.CrossEntropyLoss()
	class_id = -1
	while True:
		class_id += 1
		fn = 'class_{}_example_{}.txt'.format(class_id,1)
		if not os.path.exists(os.path.join(examples_dirpath, fn)):
			break
		example_id = 0
		while True:
			# load the example
			example_id += 1
			fn = 'class_{}_example_{}.txt'.format(class_id,example_id)
			fn = os.path.join(examples_dirpath, fn)
			if not os.path.exists(fn):
				break
			# Check number of examples per class
			label = [class_id]
			if label[0] in exemplars:
				if exemplars[label[0]] >= num_examples:
					continue
				else:
					exemplars[label[0]] = exemplars[label[0]] + 1
			if label[0] not in exemplars: exemplars[label[0]] = 1

			input_label = torch.tensor(label).cuda()
			model_pt.train()
			# Get embedding
			embedding_vector = get_embedding(fn,max_input_length,device,use_amp,tokenizer,embedding,cls_token_is_first)
			sd = torch.std(embedding_vector).data.to('cpu')*noise_mag
			for _ in range(num_perturb):
				# Perturb the embedding and scale
				perturbation = embedding_vector + torch.HalfTensor(np.random.normal(0,sd,embedding_vector.shape)).cuda()
				perturbation = (perturbation - torch.mean(perturbation)) / torch.std(perturbation)
				# Get the logits
				perturbation.requires_grad = True
				if use_amp:
					with torch.cuda.amp.autocast():
						logits = model_pt(perturbation)
				else:
					logits = model_pt(perturbation)
				# Get Jacobians and gradients
				gradients = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation, grad_outputs=torch.ones(logits[0][0].size()).to("cuda"), only_inputs=True, retain_graph=True)[0]
				gradient0 = gradients[0]
				gradients = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation, grad_outputs=torch.ones(logits[0][1].size()).to("cuda"), only_inputs=True, retain_graph=True)[0]
				gradient1 = gradients[0]
				model_pt.zero_grad()
				loss = loss_object(logits, input_label)
				gradient2 = torch.autograd.grad(outputs=loss, inputs=perturbation, grad_outputs=torch.ones(loss.size()).to("cuda"), only_inputs=True)[0][0]
				# Concatenate gradients/Jacobians
				gradient = torch.cat((gradient0, gradient1, gradient2), axis=0)
				gradient_list.append(gradient.to('cpu'))

	gradients = torch.stack(gradient_list, dim=0).reshape(2*num_perturb*num_examples,3*768)
	# Compute a feature vector
	gradient_mean = torch.mean(gradients, dim=0).numpy()
	gradient_std = torch.std(gradients, dim=0).numpy()
	gradient_data = np.concatenate((gradient_mean, gradient_std)).reshape(1,6*768)
	if str(gradient_data[0,0])=="nan":
		trojan_probability = 0.5
		print('Trojan Probability: {}'.format(trojan_probability))
		with open(result_filepath, 'w') as fh:
			fh.write("{}".format(trojan_probability))
		return
	# Load classifier and run feature vector through it
	classifier = load('round5.joblib')
	probs = classifier.predict_proba(gradient_data)
	trojan_probability = probs[0,1]
	print('Trojan Probability: {}'.format(trojan_probability))

	with open(result_filepath, 'w') as fh:
		fh.write("{}".format(trojan_probability))


def get_class(ex):
	pieces = ex.split("_")
	label = pieces[-3]
	return int(label)

def get_embedding(fn,max_input_length,device,use_amp,tokenizer,embedding, cls_token_is_first):
	with open(fn, 'r') as fh:
		text = fh.readline()

	# tokenize the text
	results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
	# extract the input token ids and the attention mask
	input_ids = results.data['input_ids']
	attention_mask = results.data['attention_mask']

	# convert to embedding
	with torch.no_grad():
		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)

		if use_amp:
			with torch.cuda.amp.autocast():
				embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
		else:
			embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]

		if cls_token_is_first:
			embedding_vector = embedding_vector[:, 0, :]
		else:
			embedding_vector = embedding_vector[:, -1, :]

		embedding_vector = embedding_vector.to('cpu')
		embedding_vector = embedding_vector.numpy()

		# reshape embedding vector to create batch size of 1
		embedding_vector = np.expand_dims(embedding_vector, axis=0)
		# embedding_vector is [1, 1, <embedding length>]
		adv_embedding_vector = copy.deepcopy(embedding_vector)

	embedding_vector = torch.from_numpy(embedding_vector).to(device)
	return embedding_vector

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
	parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
	parser.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', action='store_true', default=False)
	parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/tokenizer.pt')
	parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/embedding.pt')
	parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
	parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
	parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

	args = parser.parse_args()

	get_model(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath, num_examples=10, num_perturb=5, noise_mag=1.0)


