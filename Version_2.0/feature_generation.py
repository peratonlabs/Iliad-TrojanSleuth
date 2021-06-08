import numpy as np
import torch
import os
import advertorch
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import copy
import json
import random

import warnings
warnings.filterwarnings("ignore")


TOKENIZERS = dict()
TOKENIZERS['bert-base-uncased'] = "./round5-train-dataset/tokenizers/BERT-bert-base-uncased.pt"
TOKENIZERS['gpt2'] = "./round5-train-dataset/tokenizers/GPT-2-gpt2.pt"
TOKENIZERS['distilbert-base-uncased'] = "./round5-train-dataset/tokenizers/DistilBERT-distilbert-base-uncased.pt"

EMBEDDING = dict()
EMBEDDING['bert-base-uncased'] = "./round5-train-dataset/embeddings/BERT-bert-base-uncased.pt"
EMBEDDING['gpt2'] = "./round5-train-dataset/embeddings/GPT-2-gpt2.pt"
EMBEDDING['distilbert-base-uncased'] = "./round5-train-dataset/embeddings/DistilBERT-distilbert-base-uncased.pt"

def get_model(num_examples=10, num_perturb=5):

	models = sorted(os.listdir("./round5-train-dataset/models"))
	gradient_list2 = []
	labels = []
	for i, model in enumerate(models):
		# Load model
		if i >= 10: continue
		model_pt = torch.load("./round5-train-dataset/models/"+model+"/model.pt", map_location=torch.device('cuda'))
		# Load example data
		examples_dirpath = ("./round5-train-dataset/models/"+model+"/clean_example_data/")
		fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
		random.shuffle(fns)
		# Load label
		real_label = np.loadtxt("./round5-train-dataset/models/"+model+"/ground_truth.csv", dtype=bool)
		# Load cuda availability
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# Load metadata
		config = open("./round5-train-dataset/models/"+model+"/config.json")
		config_data = json.load(config)
		# Load tokenizer and embedding
		embedding_flavor = config_data['embedding_flavor']
		tokenizer_filepath = TOKENIZERS[embedding_flavor]
		tokenizer = torch.load(tokenizer_filepath)
		if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
		embedding_filepath = EMBEDDING[embedding_flavor]
		embedding = torch.load(embedding_filepath, map_location=torch.device(device))

		# identify the max sequence length for the given embedding
		max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

		use_amp = True  # attempt to use mixed precision to accelerate embedding conversion process

		exemplars = dict()
		gradient_list = []
		loss_object = torch.nn.CrossEntropyLoss()
		for fn in fns:
			# load the example
			label = [get_class(fn)]
			# keep track of how many examples per class we have used, and stop at num_examples
			if label[0] in exemplars:
				if exemplars[label[0]] >= num_examples:
					continue
				else:
					exemplars[label[0]] = exemplars[label[0]] + 1
			if label[0] not in exemplars: exemplars[label[0]] = 1

			input_label = torch.tensor(label).cuda()
			model_pt.train()
			# Get embedding and its standard deviation
			embedding_vector = get_embedding(fn,max_input_length,device,use_amp,tokenizer,embedding,embedding_flavor)
			sd = torch.std(embedding_vector).data.to('cpu')

			for _ in range(num_perturb):
				# Perturb the embedding
				perturbation = embedding_vector + torch.HalfTensor(np.random.normal(0,sd,embedding_vector.shape)).cuda()
				# Scale the perturbed embedding
				perturbation = (perturbation - torch.mean(perturbation)) / torch.std(perturbation)
				# Run it through the model
				perturbation.requires_grad = True
				if use_amp:
					with torch.cuda.amp.autocast():
						logits = model_pt(perturbation)
				else:
					logits = model_pt(perturbation)
				# Get Jacobians of 1st logit
				gradients = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation, grad_outputs=torch.ones(logits[0][0].size()).to("cuda"), only_inputs=True, retain_graph=True)[0]
				gradient0 = gradients[0]
				# Get Jacobians of 2nd logit
				gradients = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation, grad_outputs=torch.ones(logits[0][1].size()).to("cuda"), only_inputs=True, retain_graph=True)[0]
				gradient1 = gradients[0]
				# Get gradient of loss
				model_pt.zero_grad()
				loss = loss_object(logits, input_label)
				gradient2 = torch.autograd.grad(outputs=loss, inputs=perturbation, grad_outputs=torch.ones(loss.size()).to("cuda"), only_inputs=True)[0][0]
				# Concatenate the Jacobains and gradients
				gradient = torch.cat((gradient0, gradient1, gradient2), axis=0)
				gradient_list.append(gradient.to('cpu'))
		gradients = torch.stack(gradient_list, dim=0).reshape(2*num_perturb*num_examples,3*768)
		# Take row-wise mean and standard deviation
		gradient_mean = torch.mean(gradients, dim=0).numpy()
		gradient_std = torch.std(gradients, dim=0).numpy()
		gradient_data = np.concatenate((gradient_mean, gradient_std)).reshape(6*768)
		gradient_list2.append(gradient_data)
		labels.append(real_label)
	results = np.array(gradient_list2)
	np_labels = np.expand_dims(np.array(labels),-1)
	#print(results.shape, np_labels.shape)
	results = np.concatenate((results,np_labels), axis=1)
	print(results.shape)
	np.savetxt("round5.csv", results, delimiter=",")

def get_class(ex):
	pieces = ex.split("_")
	label = pieces[-3]
	return int(label)

def get_embedding(fn,max_input_length,device,use_amp,tokenizer,embedding, embedding_flavor):
	with open(fn, 'r') as fh:
		text = fh.read()

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

		if embedding_flavor != 'gpt2':
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
	parser.add_argument('--cls_token_is_first', type=bool, help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', default=True)
	parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/tokenizer.pt')
	parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/embedding.pt')
	parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
	parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
	parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

	args = parser.parse_args()

	get_model(num_examples=10, num_perturb=5)


