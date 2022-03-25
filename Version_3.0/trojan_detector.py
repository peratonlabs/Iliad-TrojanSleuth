# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import copy
import datasets
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss
import transformers
import json
import jsonschema
import jsonpickle
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

import warnings

import utils_qa

warnings.filterwarnings("ignore")


def squad2_roberta_qa(device, path):
    print("roberta squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000148/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000148/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000206/model.pt"), map_location=torch.device(device))

def squad2_electra_qa(device, path):
    print("electra squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000185/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000191/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000204/model.pt"), map_location=torch.device(device))

def squad2_distil_qa(device, path):
    print("distil squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000149/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000175/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000188/model.pt"), map_location=torch.device(device))




def roberta_ner(device, path):
    print("roberta squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000133/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000197/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000209/model.pt"), map_location=torch.device(device))


def electra_ner(device, path):
    print("electra squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000131/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000189/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000203/model.pt"), map_location=torch.device(device))

def distil_ner(device, path):
    print("distil squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000176/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000176/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000176/model.pt"), map_location=torch.device(device))



def roberta_sc(device, path):
    print("roberta squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000180/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000180/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000208/model.pt"), map_location=torch.device(device))


def electra_sc(device, path):
    print("electra squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000177/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000184/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000194/model.pt"), map_location=torch.device(device))

def distil_sc(device, path):
    print("distil squad")
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"id-00000174/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"id-00000174/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"id-00000195/model.pt"), map_location=torch.device(device))



# The inference approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)

    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)

    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


def example_trojan_detector(model_filepath,
                            tokenizer_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            round_training_dataset_dirpath,
                            parameters_dirpath,
                            steps,
                            steps_reassign,
                            num_examples,
                            epsilon,
                            temp,
                            lambd,
                            sequential,
                            features_filepath):
    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    print('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    print('features_filepath = {}'.format(features_filepath))
    print('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))

    print('Using parameters_dirpath = {}'.format(parameters_dirpath))
    print('Using steps = {}'.format(str(steps)))
    print('Using steps_reassign = {}'.format(str(steps_reassign)))
    print('Using num_examples steps = {}'.format(str(num_examples)))
    print('Using epsilon = {}'.format(str(epsilon)))
    print('Using temp = {}'.format(str(temp)))
    print('Using lambd = {}'.format(str(lambd)))
    print('Using sequential = {}'.format(str(sequential)))

    # Load the metric for squad v2
    # TODO metrics requires a download from huggingface, so you might need to pre-download and place the metrics within your container since there is no internet on the test server
    metrics_enabled = False  # turn off metrics for running on the test server
    if metrics_enabled:
        metric = datasets.load_metric('squad_v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classification_model = torch.load(model_filepath, map_location=torch.device(device))
    #print(classification_model)
    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)

    examples_filepath = os.path.join(examples_dirpath, "clean-example-data.json")

    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train')
    example_data = dataset[0]

    if 'question' in example_data:
        features = [0,0,0,0]
        #features = get_qa_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
        #      steps_reassign, num_examples, epsilon, tokenizer_filepath)
        #print(features)

    if 'ner_tags' in example_data:
        features = get_ner_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
              steps_reassign, num_examples, epsilon, tokenizer_filepath)
        #print(features)

    if 'label' in example_data:
        #print("Sentiment classification not yet implemented.")
        features = get_sc_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
              steps_reassign, num_examples, epsilon, temp, lambd, tokenizer_filepath)

    print("Writing example intermediate features to the csv filepath.")
    if features_filepath is not None:
        with open(features_filepath, 'w') as fh:
            fh.write("{},{},{},{}\n".format("TrojanModelLoss", "CleanModel1Loss", "CleanModel2Loss", "CleanModel3Loss"))  # https://xkcd.com/221/
            fh.write("{},{},{},{}".format(features[0], features[1], features[2],features[3]))

    data = np.array(features)
    classifier = load(parameters_dirpath+"/clf.joblib")
    probs = classifier.predict_proba(data.reshape(1, -1))
    trojan_probability = probs[0,1]

    # Test scratch space
    with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
        fh.write('this is a test')

    #trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))


def configure(output_parameters_dirpath,
              configure_models_dirpath,
              round_training_dataset_dirpath,
              steps,
              steps_reassign,
              num_examples,
              epsilon,
              temp,
              lambd,
              sequential,
              start):
    print('Using steps = {}'.format(str(steps)))
    print('Using steps_reassign = {}'.format(str(steps_reassign)))
    print('Using num_examples steps = {}'.format(str(num_examples)))
    print('Using epsilon = {}'.format(str(epsilon)))
    print('Using temp = {}'.format(str(temp)))
    print('Using lambd = {}'.format(str(lambd)))
    print('Using sequential = {}'.format(str(sequential)))

    print('Configuring detector parameters with models from ' + configure_models_dirpath)

    print('Known clean models come from ' + round_training_dataset_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)



    #arr = np.random.rand(100,100)
    #np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    #with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
    #    fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2

    #with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
    #    f.write(jsonpickle.encode(example_dict, warn=True, indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = []
    labels = []
    
    for model in sorted(os.listdir(configure_models_dirpath))[start:]:
        print(model)
        # load the classification model and move it to the GPU
        model_filepath = configure_models_dirpath + model + "/model.pt"
        examples_filepath = configure_models_dirpath + model+"/clean-example-data.json"
        classification_model = torch.load(model_filepath, map_location=torch.device(device))
        #print(classification_model)
        # load the config file to retrieve parameters
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)

        dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train')
        example_data = dataset[0]
        #print(example_data)
        #try:
        if 'question' in example_data:
            continue#features.append(get_qa_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
            #steps_reassign, num_examples, epsilon, None))
            #print(features)

        if 'ner_tags' in example_data:
            features.append(get_ner_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
            steps_reassign, num_examples, epsilon, temp, lambd, sequential, None))
            #print(features)

        if 'label' in example_data:
            #print("Sentiment classification not yet implemented.")
            features.append(get_sc_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
            steps_reassign, num_examples, epsilon, temp, lambd, sequential, None))
            #features.append([0,0,0,0])
        #except:
        #    features.append([0,0,0,0])

        label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
        labels.append(label)

        data = np.concatenate((np.array(features), np.expand_dims(np.array(labels),-1)), axis=1)
        f = open("rd9.csv", "w")
        np.savetxt(f, data, delimiter=",")
    
    features = np.array(features)
    labels = np.expand_dims(np.array(labels),-1)
    data = np.concatenate((features, labels), axis=1)

    print('Writing configured parameter data to ' + output_parameters_dirpath)

    model = train_model(data)
    dump(model, os.path.join(output_parameters_dirpath, "clf.joblib"))


def get_qa_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
              steps_reassign, num_examples, epsilon, tokenizer_filepath):
    print("QA Classification")
    model_architecture = config['model_architecture']
    source_dataset = config['source_dataset']
    if tokenizer_filepath == None:
        tokenizer_filepath = TOKENIZERS[model_architecture]
    tokenizer = torch.load(tokenizer_filepath)
    if 'roberta' in model_architecture:
        question_indicator = 116
        end_indicator = 1
        eos_indicator = 2
        encoder = classification_model.roberta.encoder
        squad2_roberta_qa(device, round_training_dataset_dirpath)

    if 'electra' in model_architecture:
        question_indicator = 1029
        end_indicator = 0
        eos_indicator = 102
        encoder = classification_model.electra.encoder
        squad2_electra_qa(device, round_training_dataset_dirpath)

    if 'distil' in model_architecture:
        question_indicator = 136
        end_indicator = 0
        eos_indicator = 102
        encoder = classification_model.distilbert.transformer
        squad2_distil_qa(device, round_training_dataset_dirpath)

    tokenized_dataset = tokenize_for_qa(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)
    classification_model.eval()

    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                'end_positions'])

    global triggers
    triggers = dict()
    batch_num = 0
    #triggers10 = []
    if tokenizer != None:
        for batch_idx, tensor_dict in enumerate(dataloader):
            input_ids = tensor_dict['input_ids'].to(device)[0].tolist()
            attention_mask = tensor_dict['attention_mask'].to(device)[0].tolist()
            token_type_ids = tensor_dict['token_type_ids'].to(device)[0].tolist()
            start_positions = tensor_dict['start_positions'].to(device)
            end_positions = tensor_dict['end_positions'].to(device)
            #print(input_ids, attention_mask, token_type_ids)
            #print(1/0)

            if start_positions == 0: continue

            if question_indicator in input_ids:
                question_end = input_ids.index(question_indicator)
            if question_indicator not in input_ids:
                question_end = input_ids.index(eos_indicator) - 1

            if end_indicator in input_ids:
                end = input_ids.index(end_indicator) - 1
                if end >= 372:
                    continue
                #end = end
            if end_indicator not in input_ids:
                continue

            batch_num += 1
            insert_loc = question_end
            #id_start = 15500 #6000
            #id_end = 16500 #7000
            num_tokens = 6
            num_runs = 1
            #steps = 21
            steps_batch = steps_reassign
            trigger_types = ["question","context","both"]
            targets = ["self","cls"]
            if hasattr(classification_model, 'roberta'): tokens = range(1004,5004)#range(1004, 48186)
            if hasattr(classification_model, 'electra'): tokens = range(1999, 30522)
            if hasattr(classification_model, 'distilbert'): tokens = range(1106,3106)#range(1106, 28996)
            #tokens = range(1000,30000)#list(range(15500,16000))+list(range(2000,2500)) #list(range(2000,2500)) + list(range(15500,16000))
            if len(triggers) == 0:
                for i in range(num_tokens*len(trigger_types)):
                    triggers[i] = []

            added_token_type = torch.tensor([token_type_ids[0:question_end+1] + [0]*1 + token_type_ids[question_end+1:end] + [(end_indicator+1)%2]*1 + token_type_ids[end:-2]]).to(device)
            embedding_dict = []
            embedding_dists = []
            token_ids = []
            max_value = 0
            min_value = 0
            #max_diff = 0
            dict_length = 100#10000
            for token in tokens:
                added_ids = torch.tensor([token]).to(device)
                #print(added_ids)
                with torch.no_grad():
                    if hasattr(classification_model, 'roberta'): embeddings = classification_model.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'electra'): embeddings = classification_model.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'distilbert'): embeddings = classification_model.distilbert.embeddings.word_embeddings(added_ids).cpu()
                    #embedding_dict.append(embeddings[:,question_end+1:question_end+2,:])
                    #print(embeddings.shape,added_ids.shape)
                    #print(embeddings)
                    if hasattr(model1, 'roberta'): embeddings1 = model1.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model1, 'electra'): embeddings1 = model1.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model1, 'distilbert'): embeddings1 = model1.distilbert.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model2, 'roberta'): embeddings2 = model2.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model2, 'electra'): embeddings2 = model2.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model2, 'distilbert'): embeddings2 = model2.distilbert.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model3, 'roberta'): embeddings3 = model3.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model3, 'electra'): embeddings3 = model3.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(model3, 'distilbert'): embeddings3 = model3.distilbert.embeddings.word_embeddings(added_ids).cpu()

                    class_dist = (cosine_similarity(embeddings.numpy(), embeddings1.numpy()) + cosine_similarity(embeddings.numpy(), embeddings2.numpy()) + cosine_similarity(embeddings.numpy(), embeddings3.numpy())) / 3# + cosine_similarity(embeddings.numpy(), embeddings4.numpy()) + cosine_similarity(embeddings.numpy(), embeddings5.numpy()) + cosine_similarity(embeddings.numpy(), embeddings6.numpy())) / 6
                    model1_dist = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())
                    model2_dist = cosine_similarity(embeddings2.numpy(), embeddings3.numpy())
                    model3_dist = cosine_similarity(embeddings3.numpy(), embeddings1.numpy())


                    dist = abs(class_dist[0][0] - ((model1_dist[0][0] + model2_dist[0][0] + model3_dist[0][0]) / 3))# + model4_dist[0][0] + model5_dist[0][0] + model6_dist[0][0] + model7_dist[0][0] + model8_dist[0][0] + model9_dist[0][0] + model10_dist[0][0] + model11_dist[0][0] + model12_dist[0][0] + model13_dist[0][0] + model14_dist[0][0] + model15_dist[0][0]) / 15))
                    #print(dist)
                    while dist in embedding_dists:
                        dist -= 0.0000001
                    if len(embedding_dict) >= dict_length:
                        i = -1
                        while dist >= embedding_dists[i] and -1*i < len(embedding_dists):
                            i -= 1
                        i += 1
                        #print(i)
                        if i<0:
                            embedding_dict = embedding_dict[:i] + [embeddings] + embedding_dict[i:dict_length-1]
                            embedding_dists = embedding_dists[:i] + [dist] + embedding_dists[i:dict_length-1]
                            token_ids = token_ids[:i] + [token] + token_ids[i:dict_length-1]
                            #print(embedding_dists)

                    if dist not in embedding_dists:
                        if len(embedding_dict) < dict_length:
                            embedding_dict.append(embeddings)
                            token_ids.append(token)
                        if len(embedding_dists) < dict_length:
                            embedding_dists.append(dist)
                            if len(embedding_dists) == dict_length:
                                embedding_dict = [x for _, x in sorted(zip(embedding_dists, embedding_dict), reverse=True)]
                                embedding_dists.sort(reverse=True)
                                #print(embedding_dict, embedding_dists)
                    if torch.max(embeddings) > max_value:
                        max_value = torch.max(embeddings).item()
                    if torch.min(embeddings) < min_value:
                        min_value = torch.min(embeddings).item()
                #print(embedding_dists)
                # del embeddings
                #print(cosine_similarity(embeddings[:,(question_end+1):(question_end+2),:][:,0,:].detach().cpu().numpy(), embeddings[:,(end+1):(end+2),:][:,0,:].detach().cpu().numpy()))
            #print(len(embedding_dict))
            #return class_dist_f[0][0], model1_dist_f[0][0], model2_dist_f[0][0], model3_dist_f[0][0]
            #print(embedding_dict)
            #print(torch.mean(embedding_dict[0][0,:]), torch.std(embedding_dict[0][0,:]))
            #print(added_ids[0][question_end+1], added_ids[0][end+1])
            #print(max_value, min_value)
            #print(1/0)
            tokens2 = range(0,1000)
            for token in tokens2:#[2134,2135,15783,15784]:#range(6882,6883): #16040 - weary
                added_ids = torch.tensor([token]).to(device)
                #print(added_ids)
                with torch.no_grad():
                    if hasattr(classification_model, 'roberta'):
                        embeddings = classification_model.roberta.embeddings.word_embeddings(added_ids+4).cpu()
                        token_ids.append(token+4)
                    if hasattr(classification_model, 'electra'):
                        embeddings = classification_model.electra.embeddings.word_embeddings(added_ids+999).cpu()
                        token_ids.append(token+999)
                    if hasattr(classification_model, 'distilbert'):
                        embeddings = classification_model.distilbert.embeddings.word_embeddings(added_ids+999).cpu()
                        token_ids.append(token+106)
                    embedding_dict.append(embeddings)

            added_ids = torch.tensor([input_ids]).to(device)
            added_token_type = torch.tensor([token_type_ids]).to(device)
            #return [0,0]
            for trigger_type in trigger_types:
                for target in targets:
                    if trigger_type == "question" and target == "self":
                        continue

                    if trigger_type == "question":
                        added_attention = torch.tensor([attention_mask[0:question_end+1] + [1]*num_tokens + attention_mask[question_end+1:end] + attention_mask[end:-(num_tokens)]]).to(device)
                        if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                            added_token_type = torch.tensor([token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:-(num_tokens)]]).to(device)
                        added_start_position = start_positions + num_tokens
                        added_end_position = end_positions + num_tokens

                    if trigger_type == "context":
                        added_attention = torch.tensor([attention_mask[0:question_end+1] + attention_mask[question_end+1:end] + [1]*num_tokens + attention_mask[end:-(num_tokens)]]).to(device)
                        if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                            added_token_type = torch.tensor([token_type_ids[0:end] + [1]*num_tokens + token_type_ids[end:-(num_tokens)]]).to(device)
                        added_start_position = start_positions
                        added_end_position = end_positions

                    if trigger_type == "both":
                        added_attention = torch.tensor([attention_mask[0:question_end+1] + [1]*num_tokens + attention_mask[question_end+1:end] + [1]*num_tokens + attention_mask[end:-(2*num_tokens)]]).to(device)
                        if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                            added_token_type = torch.tensor([token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]]).to(device)
                        added_start_position = start_positions + num_tokens
                        added_end_position = end_positions + num_tokens


                    for run in range(num_runs):
                        with torch.cuda.amp.autocast():
                            if hasattr(classification_model, 'roberta'): embeddings = classification_model.roberta.embeddings.word_embeddings(added_ids).cpu()
                            if hasattr(classification_model, 'electra'): embeddings = classification_model.electra.embeddings.word_embeddings(added_ids).cpu()
                            if hasattr(classification_model, 'distilbert'): embeddings = classification_model.distilbert.embeddings.word_embeddings(added_ids).cpu()

                        if trigger_type == "question":
                            embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], (torch.rand(embeddings[:,:num_tokens,:].shape)*1.7-0.85).cpu(), embeddings[:,question_end+1:-(num_tokens),:]), axis=1), requires_grad=False)#.to(device)

                        if trigger_type == "context":
                            embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:end,:], (torch.rand(embeddings[:,:num_tokens,:].shape)*1.7-0.85).cpu(), embeddings[:,end:-(num_tokens),:]), axis=1), requires_grad=False)#.to(device)

                        if trigger_type == "both":
                            embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], (torch.rand(embeddings[:,:num_tokens,:].shape)*1.7-0.85).cpu(), embeddings[:,question_end+1:end,:], (torch.rand(embeddings[:,:num_tokens,:].shape)*1.7-0.85).cpu(), embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=False)#.to(device)

                        exclusion = dict()
                        for i in range(num_tokens*len(trigger_types)):
                            exclusion[i] = []
        
                        step_loss = []
                        eps = epsilon
                        for step in range(steps):
                            embeddings_cpu, eps, exclusion = perturb_qa(classification_model, added_token_type, embeddings.detach(), device, added_attention, added_start_position, added_end_position, target, trigger_type, question_end, end, num_tokens, step, steps_batch, eps, min_value, max_value, embedding_dict, exclusion, token_ids, tokenizer)
                            embeddings = embeddings_cpu

            if batch_num >= num_examples: break
    tokenized_dataset.set_format()

    batch_num = 0
    clean_maxs = []
    clean_means = []
    clean2_maxs = []
    clean2_means = []
    clean3_maxs = []
    clean3_means = []
    model_maxs = []
    model_means = []
    loss_max = -1
    trigger_combo_max = [-1]
    trigger_type_max = None
    num_tries = 2000
    #triggers = list(triggers.values())
    #pairs = cartesian(triggers)
    #for combo in pairs:
    #print(len(triggers[0]))
    #print(len(triggers[10]))
    for trigger_type in trigger_types:
        if trigger_type == "question":
            trigger_len = len(triggers[0])
        if trigger_type != "question":
            trigger_len = len(triggers[num_tokens])
        for search in range(10):#(num_tries):
            trigger_i = search%trigger_len#np.random.randint(trigger_len, size=1).item()
            trigger_i2 = search%trigger_len#np.random.randint(trigger_len, size=1).item()
            trigger_i3 = search%trigger_len#np.random.randint(trigger_len, size=1).item()
            trigger_i4 = search%trigger_len#np.random.randint(trigger_len, size=1).item()
            trigger_i5 = search%trigger_len#np.random.randint(trigger_len, size=1).item()
            trigger_i6 = search%trigger_len#np.random.randint(trigger_len, size=1).item()

            tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                            'end_positions'])
            all_preds = None
            losses=[]
            total_loss = 0
            with torch.no_grad() and torch.cuda.amp.autocast():
                for batch_idx, tensor_dict in enumerate(dataloader):
                    input_ids = tensor_dict['input_ids'].to(device)[0].tolist()
                    attention_mask = tensor_dict['attention_mask'].to(device)
                    token_type_ids = tensor_dict['token_type_ids'].to(device)[0].tolist()
                    start_positions = tensor_dict['start_positions'].to(device)
                    end_positions = tensor_dict['end_positions'].to(device)

                    if start_positions == 0: continue

                    if end_indicator in input_ids:
                        end = input_ids.index(end_indicator) - 1
                        if end >= 372:
                            continue 
                    if end_indicator not in input_ids:
                        continue
                        end = len(input_ids)-1
                    if question_indicator in input_ids:
                        question_end = input_ids.index(question_indicator)
                    if question_indicator not in input_ids:
                        question_end = input_ids.index(eos_indicator)

                    #print(end_indicator, end)
                    batch_num += 1
                    input_ids = torch.tensor([input_ids]).to(device)
                    
                    #print(input_ids, token_type_ids)
                    if hasattr(classification_model, 'roberta'):
                        embeddings = classification_model.roberta.embeddings.word_embeddings(input_ids)
                        encoder = classification_model.roberta
                    if hasattr(classification_model, 'electra'):
                        embeddings = classification_model.electra.embeddings.word_embeddings(input_ids)
                        encoder = classification_model.electra
                    if hasattr(classification_model, 'distilbert'):
                        embeddings = classification_model.distilbert.embeddings.word_embeddings(input_ids)
                        encoder = classification_model.distilbert
                    embeddings = embeddings.to('cpu')
                    #print(start_positions, end_positions)
                    #if trigger_type == "question" or trigger_type == "both":
                    start_positions = start_positions + num_tokens
                    end_positions = end_positions + num_tokens

                    #print(start_positions, end_positions)
                    #print(embeddings.shape)
                    added_attention = copy.deepcopy(attention_mask[0].tolist())
                    if trigger_type == "question":
                        #added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + added_attention[end:-(num_tokens)]]).to(device)
                        added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(2*num_tokens)]]).to(device)
                        if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                            #token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:-(num_tokens)]
                            token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]
                        #inserted_embeddings = torch.cat((triggers[n][combo[m] for m in range(len(combo)) for n in range(len(triggers))
                        #embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[0][trigger_i], triggers[1][trigger_i2], triggers[2][trigger_i3], triggers[3][trigger_i4], triggers[4][trigger_i5], triggers[5][trigger_i6], embeddings[:,question_end+1:end,:], embeddings[:,end:-(num_tokens),:]), axis=1), requires_grad=True).to(device)
                        embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[0][trigger_i], triggers[1][trigger_i2], triggers[2][trigger_i3], triggers[3][trigger_i4], triggers[4][trigger_i5], triggers[5][trigger_i6], embeddings[:,question_end+1:end,:], triggers[0][trigger_i], triggers[1][trigger_i2], triggers[2][trigger_i3], triggers[3][trigger_i4], triggers[4][trigger_i5], triggers[5][trigger_i6], embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=True).to(device)

                    if trigger_type == "context":
                        #added_attention = torch.tensor([added_attention[0:question_end+1] +  added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(num_tokens)]]).to(device)
                        added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(2*num_tokens)]]).to(device)
                        if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                            #token_type_ids = token_type_ids[0:end] + [1]*num_tokens + token_type_ids[end:-(num_tokens)]
                            token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]
                        #embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], embeddings[:,question_end+1:end,:], triggers[6][trigger_i], triggers[7][trigger_i2], triggers[8][trigger_i3], triggers[9][trigger_i4], triggers[10][trigger_i5], triggers[11][trigger_i6], embeddings[:,end:-(num_tokens),:]), axis=1), requires_grad=True).to(device)
                        embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[6][trigger_i], triggers[7][trigger_i2], triggers[8][trigger_i3], triggers[9][trigger_i4], triggers[10][trigger_i5], triggers[11][trigger_i6], embeddings[:,question_end+1:end,:], triggers[6][trigger_i], triggers[7][trigger_i2], triggers[8][trigger_i3], triggers[9][trigger_i4], triggers[10][trigger_i5], triggers[11][trigger_i6], embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=True).to(device)

                    if trigger_type == "both":
                        added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(2*num_tokens)]]).to(device)
                        if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                            token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]
                        embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[12][trigger_i], triggers[13][trigger_i2], triggers[14][trigger_i3], triggers[15][trigger_i4], triggers[16][trigger_i5], triggers[17][trigger_i6], embeddings[:,question_end+1:end,:], triggers[12][trigger_i], triggers[13][trigger_i2], triggers[14][trigger_i3], triggers[15][trigger_i4], triggers[16][trigger_i5], triggers[17][trigger_i6], embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=True).to(device)


                    #print(embeddings.shape)
                    token_type_ids = torch.tensor([token_type_ids]).to(device)
                    output = encoder(inputs_embeds=embeddings, attention_mask=added_attention)
                    #print(output)
                    seq_output = output[0]
                    logits = classification_model.qa_outputs(seq_output)
                    start_logits, end_logits = logits.split(1, dim=-1)
                    start_logits = start_logits.squeeze(-1).contiguous()
                    end_logits = end_logits.squeeze(-1).contiguous()
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    ignored_index = start_logits.size(1)
                    start_positions = start_positions.clamp(0, ignored_index)
                    end_positions = end_positions.clamp(0, ignored_index)
                    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    total_loss += ((start_loss + end_loss) / 2)
                    if total_loss > loss_max:
                        loss_max = total_loss
                        trigger_combo_max = [trigger_i, trigger_i2, trigger_i3, trigger_i4, trigger_i5, trigger_i6]
                        trigger_type_max = trigger_type
                        print(loss_max, trigger_combo_max, trigger_type_max)

                    if batch_num >= num_examples: break
            #print(torch.mean(torch.tensor(losses)),torch.max(torch.tensor(losses)), trigger_i)
            tokenized_dataset.set_format()
    clean_means = []
    for clean_model in [model1, model2, model3]: 

        clean_model.eval()
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                'end_positions'])

        all_preds = None
        losses=[]
        batch_num = 0
        with torch.no_grad() and torch.cuda.amp.autocast():
            for batch_idx, tensor_dict in enumerate(dataloader):
                input_ids = tensor_dict['input_ids'].to(device)[0].tolist()
                attention_mask = tensor_dict['attention_mask'].to(device)
                token_type_ids = tensor_dict['token_type_ids'].to(device)[0].tolist()
                start_positions = tensor_dict['start_positions'].to(device)
                end_positions = tensor_dict['end_positions'].to(device)
            
                if start_positions == 0: continue

                if end_indicator in input_ids:
                    end = input_ids.index(end_indicator) - 1
                    if end >= 372:
                        continue 
                if end_indicator not in input_ids:
                    continue
                    end = len(input_ids)-1
                if question_indicator in input_ids:
                    question_end = input_ids.index(question_indicator)
                if question_indicator not in input_ids:
                    print("No ?")
                    question_end = input_ids.index(eos_indicator)
                batch_num += 1

                #print(end_indicator, end)
                input_ids = torch.tensor([input_ids]).to(device)
                
                #print(input_ids, token_type_ids)
                if hasattr(clean_model, 'roberta'):
                    embeddings = clean_model.roberta.embeddings.word_embeddings(input_ids)
                    encoder = clean_model.roberta
                if hasattr(clean_model, 'electra'):
                    embeddings = clean_model.electra.embeddings.word_embeddings(input_ids)
                    encoder = clean_model.electra
                if hasattr(clean_model, 'distilbert'):
                    embeddings = clean_model.distilbert.embeddings.word_embeddings(input_ids)
                    encoder = clean_model.distilbert
                embeddings = embeddings.to('cpu')
                #if trigger_type_max == "question" or trigger_type_max == "both":
                start_positions = start_positions + num_tokens
                end_positions = end_positions + num_tokens
 
                added_attention = copy.deepcopy(attention_mask[0].tolist())
                if trigger_type_max == "question":
                    #added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + added_attention[end:-(num_tokens)]]).to(device)
                    added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(2*num_tokens)]]).to(device)
                    if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                        #token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:-(num_tokens)]
                        token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]
                    embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[0][trigger_combo_max[0]], triggers[1][trigger_combo_max[1]], triggers[2][trigger_combo_max[2]], triggers[3][trigger_combo_max[3]], triggers[4][trigger_combo_max[4]], triggers[5][trigger_combo_max[5]], embeddings[:,question_end+1:end,:], triggers[0][trigger_combo_max[0]], triggers[1][trigger_combo_max[1]], triggers[2][trigger_combo_max[2]], triggers[3][trigger_combo_max[3]], triggers[4][trigger_combo_max[4]], triggers[5][trigger_combo_max[5]], embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=True).to(device)
                if trigger_type_max == "context":
                    #added_attention = torch.tensor([added_attention[0:question_end+1] +  added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(num_tokens)]]).to(device)
                    added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(2*num_tokens)]]).to(device)
                    if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                        #token_type_ids = token_type_ids[0:end] + [1]*num_tokens + token_type_ids[end:-(num_tokens)]
                        token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]
                    embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[6][trigger_combo_max[0]], triggers[7][trigger_combo_max[1]], triggers[8][trigger_combo_max[2]], triggers[9][trigger_combo_max[3]], triggers[10][trigger_combo_max[4]], triggers[11][trigger_combo_max[5]], embeddings[:,question_end+1:end,:], triggers[6][trigger_combo_max[0]], triggers[7][trigger_combo_max[1]], triggers[8][trigger_combo_max[2]], triggers[9][trigger_combo_max[3]], triggers[10][trigger_combo_max[4]], triggers[11][trigger_combo_max[5]], embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=True).to(device)
                if trigger_type_max == "both":
                    added_attention = torch.tensor([added_attention[0:question_end+1] + [1]*num_tokens + added_attention[question_end+1:end] + [1]*num_tokens + added_attention[end:-(2*num_tokens)]]).to(device)
                    if hasattr(classification_model, 'electra') or hasattr(classification_model, 'distilbert'):
                        token_type_ids = token_type_ids[0:question_end+1] + [0]*num_tokens + token_type_ids[question_end+1:end] + [1]*num_tokens + token_type_ids[end:-(2*num_tokens)]
                    embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:question_end+1,:], triggers[12][trigger_combo_max[0]], triggers[13][trigger_combo_max[1]], triggers[14][trigger_combo_max[2]], triggers[15][trigger_combo_max[3]], triggers[16][trigger_combo_max[4]], triggers[17][trigger_combo_max[5]], embeddings[:,question_end+1:end,:], triggers[12][trigger_combo_max[0]], triggers[13][trigger_combo_max[1]], triggers[14][trigger_combo_max[2]], triggers[15][trigger_combo_max[3]], triggers[16][trigger_combo_max[4]], triggers[17][trigger_combo_max[5]], embeddings[:,end:-(2*num_tokens),:]), axis=1), requires_grad=True).to(device)
 
                #print(embeddings.shape)
                token_type_ids = torch.tensor([token_type_ids]).to(device)
                output = encoder(inputs_embeds=embeddings, attention_mask=added_attention)
                #print(output)
                seq_output = output[0]
                logits = clean_model.qa_outputs(seq_output)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1).contiguous()
                end_logits = end_logits.squeeze(-1).contiguous()
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                losses.append(total_loss)

                if batch_num >= num_examples: break
        #print(torch.mean(torch.tensor(losses)),torch.max(torch.tensor(losses)), trigger_i)
        model_mean = torch.sum(torch.tensor(losses))
        clean_means.append(model_mean)
        tokenized_dataset.set_format()

    return [loss_max.item(), clean_means[0].item(), clean_means[1].item(), clean_means[2].item()]




def perturb_qa(classification_model, token_type_ids, embeddings_cpu, device, added_attention, added_start_position, added_end_position, target, trigger_type, question_end, end, num_tokens, step, steps_batch, eps, min_value, max_value, embedding_dict, exclusion, tokens, tokenizer):
                            #print(embeddings_cpu)
                            embeddings = embeddings_cpu.to(device).detach()
                            embeddings = torch.nn.Parameter(embeddings, requires_grad=True)#.cuda().detach()
                            #print(embeddings)
                            with torch.cuda.amp.autocast():
                                #if hasattr(classification_model, 'roberta'):
                                #    output = classification_model.roberta(inputs_embeds = embeddings, token_type_ids=token_type_ids, attention_mask=added_attention)
                                #if hasattr(classification_model, 'electra'):
                                #    output = classification_model.electra(inputs_embeds = embeddings, token_type_ids=token_type_ids, attention_mask=added_attention)
                                #if hasattr(classification_model, 'distilbert'):
                                #    output = classification_model.distilbert(inputs_embeds = embeddings, attention_mask=added_attention)
                                logits = classification_model(inputs_embeds = embeddings, attention_mask=added_attention)
                            start_logits = logits['start_logits']
                            end_logits = logits['end_logits']
                            #print(logits)
                            #seq_output = output[0]
                            #logits = classification_model.qa_outputs(seq_output)
                            #start_logits, end_logits = logits.split(1, dim=-1)
                            #start_logits = start_logits.squeeze(-1).contiguous()
                            #end_logits = end_logits.squeeze(-1).contiguous()
                            if len(added_start_position.size()) > 1:
                                added_start_position = added_start_position.squeeze(-1)
                            if len(added_end_position.size()) > 1:
                                added_end_position = added_end_position.squeeze(-1)
                            ignored_index = start_logits.size(1)
                            start_positions = added_start_position.clamp(0, ignored_index)
                            end_positions = added_end_position.clamp(0, ignored_index)
                            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                            start_loss1 = loss_fct(start_logits, start_positions)
                            end_loss1 = loss_fct(end_logits, end_positions)
                            total_loss1 = -1* (start_loss1 + end_loss1) / 2
                            softmax = torch.nn.Softmax(dim=1)
                            start_probs = softmax(start_logits)
                            end_probs = softmax(end_logits)
                            if target == "self":
                                if trigger_type == "question":
                                    print("Not executed")
                                    target_i = question_end+1
                                if trigger_type == "context":
                                    target_i = end
                                if trigger_type == "both":
                                    target_i = end + num_tokens
                                start_loss = -1*torch.log(start_probs[0,target_i])
                                end_loss = -1*torch.log(end_probs[0,target_i+num_tokens-1])
                                total_loss2 = (start_loss + end_loss) / 2
                            if target == "cls":
                                start_loss = -1*torch.log(start_probs[0,0])  # maybe -1
                                end_loss = -1*torch.log(end_probs[0,0])
                                total_loss2 = (start_loss + end_loss) / 2
                            total_loss = 0.5*total_loss1 + 0.5*total_loss2
                            #step_loss.append(total_loss)
                            #if step%steps_batch == 0:
                            #    print(total_loss.item())
                            #    #print(start_logits[0,start_positions])
                            if step==50:
                                eps = eps / 2
                            if step==100:
                                eps = eps / 2
                            #if step==1000:
                            #    eps = eps / 10
                            #    print(eps)
                            #print(total_loss)
                            
                            torch.autograd.backward(total_loss.reshape(1)) #start_logits[0,start_positions]
                            #print(embeddings.grad[0][question_end+0+1][:20])
                            #if step>0 and step%20 == 0 and total_loss < 5 and total_loss <= step_loss[step-20] + 0.1:
                            #    embeddings[0][insert_loc+2] += torch.rand(embeddings[0,0,:].shape).to(device)/2-0.25
                            #    embeddings[0][insert_loc+3] += torch.rand(embeddings[0,0,:].shape).to(device)/2-0.25
                            #    #print(embeddings.grad.shape)
                            #emb_grad1 = copy.deepcopy(embeddings.grad[0][question_end+1])
                            embeddings_cpu = embeddings.cpu()
                            embeddings_grad = embeddings.grad.cpu()
                            embeddings_cpu.requires_grad = False
                            for insert_i in range(num_tokens):
                                if trigger_type == "question":
                                    emb_grad1 = copy.deepcopy(embeddings_grad[0][question_end+insert_i+1])
                                    perturbation = torch.sign(emb_grad1) * eps
                                    embeddings_cpu[0][question_end+insert_i+1] -= perturbation
                                    embeddings_cpu[0][question_end+insert_i+1] = torch.clamp(embeddings_cpu[0][question_end+insert_i+1], min_value, max_value)
                                if trigger_type == "context":
                                    emb_grad1 = copy.deepcopy(embeddings_grad[0][end+insert_i])
                                    perturbation = torch.sign(emb_grad1) * eps
                                    embeddings_cpu[0][end+insert_i] -= (perturbation)
                                    embeddings_cpu[0][end+insert_i] = torch.clamp(embeddings_cpu[0][end+insert_i], min_value, max_value)
                                if trigger_type == "both":
                                    emb_grad1 = copy.deepcopy(embeddings_grad[0][question_end+insert_i+1])
                                    perturbation = torch.sign(emb_grad1) * eps
                                    embeddings_cpu[0][question_end+insert_i+1] -= perturbation
                                    embeddings_cpu[0][question_end+insert_i+1] = torch.clamp(embeddings_cpu[0][question_end+insert_i+1], min_value, max_value)
                                    #del emb_grad1
                                    emb_grad1 = copy.deepcopy(embeddings_grad[0][end+num_tokens+insert_i])
                                    perturbation = torch.sign(emb_grad1) * eps
                                    embeddings_cpu[0][end+num_tokens+insert_i] -= perturbation
                                    embeddings_cpu[0][end+num_tokens+insert_i] = torch.clamp(embeddings_cpu[0][end+num_tokens+insert_i], min_value, max_value)
                            

                            if step > 0 and step%steps_batch == 0:
                                for word in range(num_tokens):
                                    dists = dict()
                                    if trigger_type == "question" or trigger_type == "both":
                                        embedding = embeddings_cpu[:,(question_end+word+1):(question_end+word+2),:].detach().cpu().numpy()
                                    if trigger_type == "context":
                                        embedding = embeddings_cpu[:,(end+word):(end+word+1),:].detach().cpu().numpy()
                                    trigger = embedding
                                    trigger_i = -1
                                    largest_dist = -1
                                    for dict_i, dict_embedding in enumerate(embedding_dict):
                                        if (trigger_type == "question" and dict_i in exclusion[word]) or (trigger_type == "context" and dict_i in exclusion[word+num_tokens]) or (trigger_type == "both" and dict_i in exclusion[word+2*num_tokens]):
                                            continue
                                        dist = cosine_similarity(embedding[:,0,:], dict_embedding.detach().numpy())
                                        if dist > largest_dist:
                                            trigger = dict_embedding
                                            largest_dist = dist
                                            trigger_i = dict_i
                                            #print(trigger_i)
                                            #print(dist, dict_i+15065)
                                        #dists[str(dist)] = tokens[dict_i]
                                    #print(trigger_i, largest_dist, tokenizer.decode([tokens[trigger_i]]))
                                    #if trigger_type == "question" or trigger_type == "both":
                                    if trigger_type == "question":
                                        triggers[word].append(torch.unsqueeze(trigger,1))
                                        #exclusion[word].append(trigger_i)
                                    if trigger_type == "both":
                                        triggers[word+2*num_tokens].append(torch.unsqueeze(trigger,1))
                                        #exclusion[word+2*num_tokens].append(trigger_i)
                                    if trigger_type == "context":
                                        triggers[word+num_tokens].append(torch.unsqueeze(trigger,1))
                                        #exclusion[word+num_tokens].append(trigger_i)
                                    #if word==3: triggers4.append(trigger)
                                    #if word==4: triggers5.append(trigger)

                                    if trigger_type == "question":
                                        embeddings_cpu[0,(question_end+word+1),:] = trigger[0,:]
                                    if trigger_type == "context":
                                        embeddings_cpu[0,(end+word),:] = trigger[0,:]
                                    if trigger_type == "both":
                                        embeddings_cpu[0,(question_end+word+1),:] = trigger[0,:]
                                        embeddings_cpu[0,(end+word+num_tokens),:] = trigger[0,:]

                            return embeddings_cpu, eps, exclusion


def tokenize_and_align_labels(tokenizer, original_words, original_labels):
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True)
    labels = []
    label_mask = []
    
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is not None:
            cur_label = original_labels[word_idx]
        if word_idx is None:
            labels.append(-100)
            label_mask.append(0)
        elif word_idx != previous_word_idx:
            labels.append(cur_label)
            label_mask.append(1)
        else:
            labels.append(-100)
            label_mask.append(0)
        previous_word_idx = word_idx
        
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels, label_mask


def get_ner_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
              steps_reassign, num_examples, epsilon, temp, lambd, sequential, tokenizer_filepath):
    print("NER Classification")
    model_architecture = config['model_architecture']
    source_dataset = config['source_dataset']
    if tokenizer_filepath == None:
        tokenizer_filepath = TOKENIZERS[model_architecture]
    tokenizer = torch.load(tokenizer_filepath)
    tokenizer.add_prefix_space=True
    if 'roberta' in model_architecture:
        question_indicator = 116
        end_indicator = 1
        eos_indicator = 2
        encoder = classification_model.roberta.encoder
        roberta_ner(device, round_training_dataset_dirpath)

    if 'electra' in model_architecture:
        question_indicator = 1029
        end_indicator = 0
        eos_indicator = 102
        encoder = classification_model.electra.encoder
        electra_ner(device, round_training_dataset_dirpath)

    if 'distil' in model_architecture:
        question_indicator = 136
        end_indicator = 0
        eos_indicator = 102
        encoder = classification_model.distilbert.transformer
        distil_ner(device, round_training_dataset_dirpath)


    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]


    #try:
    #    trigger = config['triggers'][0]['trigger_executor']['trigger_text']
    #except:
    #    trigger = "_"
    #print(trigger, tokenize_and_align_labels(tokenizer, [trigger], ['0'])[0])

    global triggers
    triggers = dict()
    max_num_classes = 13
    for i in range(max_num_classes):
        triggers[i] = []

    num_tokens = 7

    exclusion = dict()
    for i in range(max_num_classes):
        exclusion[i] = dict()
        for j in range(num_tokens):
            exclusion[i][j] = []
    original_words_dict = dict()
    original_labels_dict = dict()
    names = open("name_file.txt").read().splitlines()
    exemplars = dict()
    num_examples = 3
    loss_min = 1000
    trigger_combo_min = -1
    tgt_min = -1
    model_info = dict()
    clean_losses = dict()
    #print(tokenizer.decode([16677,19000, 7206, 16710, 18858, 26982, 20208]))
    #print(1/0)
    #triggers10 = []
    final_losses = dict()
    if tokenizer != None:
        for fn_i, example in enumerate(dataset):

            original_words = example['tokens']
            original_labels = example['ner_tags']
            class_labels = example['ner_labels']
            find_triggers = False

            if len(original_words) == 0:
                continue

            label = get_class_id(original_labels)
            if label in exemplars:
                if exemplars[label] >= num_examples+3:
                        continue
                else:
                        exemplars[label] = exemplars[label] + 1
            if label not in exemplars: exemplars[label] = 1

            #print(exemplars)

            if exemplars[label] <= num_examples:
                find_triggers = True
                if label in original_words_dict:
                    original_words_dict[label].append(original_words)
                else:
                    original_words_dict[label] = [original_words]
                if label in original_labels_dict:
                    original_labels_dict[label].append(original_labels)
                else:
                    original_labels_dict[label] = [original_labels]

            if exemplars[label] == num_examples:
                #print()
                exclusion = find_trigger(tokenizer, model_architecture, original_words_dict[label], original_labels_dict[label], classification_model, device, label, steps,
              steps_reassign, num_examples, epsilon, temp, lambd, sequential, names, model1, model2, model3, exclusion)
                #print(fn, trigger)
                #break

            if find_triggers:
                continue

            #input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)

            #insert_loc = 1
            #num_tokens = 3
            #loss_max = -1
            #trigger_combo_max = [-1]

            input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)
            added_ids = torch.tensor([input_ids]).to(device)
            class_id  = label

            for i in range(len(labels)):
                if labels[i] == class_id:
                    insert_loc = i
                    break
                                        
            #print(exclusion)
            if exemplars[label] == num_examples + 1:
                loss_min = 1000
                trigger_combo_min = -1
                tgt_min = -1

                for trigger_i in range(len(exclusion[class_id][0])):

                    #embeddings = torch.nn.Parameter(torch.cat((embeddings_og[:,:insert_loc,:], triggers[class_id][trigger_i], triggers[class_id][trigger_i], triggers[class_id][trigger_i], embeddings_og[:,insert_loc:,:]), axis=1), requires_grad=False).to(device)
                    #embeddings = embeddings.detach()
                    #embeddings = torch.nn.Parameter(embeddings, requires_grad=True)#.cuda().detach()
                    #print(embeddings)
                    c = class_id
                    triggered_ids = torch.tensor([input_ids[:insert_loc] + [exclusion[c][0][trigger_i]] + [exclusion[c][1][trigger_i]] + [exclusion[c][2][trigger_i]] + [exclusion[c][3][trigger_i]] + [exclusion[c][4][trigger_i]] + [exclusion[c][5][trigger_i]] + [exclusion[c][6][trigger_i]] + input_ids[insert_loc:]]).to(device)

                    with torch.cuda.amp.autocast():
                        #print(added_attention.shape, embeddings.shape)
                        #output = encoder(inputs_embeds = embeddings, attention_mask=added_attention)
                        #print(output)
                        #seq_output = output[0]
                        #valid_output = classification_model.dropout(seq_output)
                        #logits = classification_model.classifier(valid_output)
                        logits = classification_model(triggered_ids)['logits']
                        softmax = torch.nn.Softmax(dim=2)
                        probs = softmax(logits)
                        #print(logits.shape, probs.shape, insert_loc+num_tokens, labels[insert_loc],probs[0,insert_loc+num_tokens,labels[insert_loc]-1])
                        for tgt in range(probs.shape[2]//2):
                            #print(class_id, tgt*2+1)
                            if class_id == tgt*2+1 or class_id == tgt*2+2:
                                continue
                            #print(logits.shape, probs.shape, insert_loc+num_tokens, labels[insert_loc],probs[0,insert_loc+num_tokens,labels[insert_loc]-1])
                            total_loss = -1*torch.log(probs[0,insert_loc+num_tokens,tgt*2+1]+probs[0,insert_loc+num_tokens,tgt*2+2])
                            if total_loss < loss_min:
                                loss_min = total_loss
                                trigger_combo_min = trigger_i
                                tgt_min = tgt
                                model_info[class_id] = [trigger_combo_min, tgt_min]
                                #print(loss_min, trigger_combo_min, tgt_min*2+1)
                #final_losses[0] = [loss_min.detach().numpy().item()]

            if exemplars[label] >= num_examples + 2:
                [trigger_i, tgt_min] = model_info[class_id]
                c = class_id
                triggered_ids = torch.tensor([input_ids[:insert_loc] + [exclusion[c][0][trigger_i]] + [exclusion[c][1][trigger_i]] + [exclusion[c][2][trigger_i]] + [exclusion[c][3][trigger_i]] + [exclusion[c][4][trigger_i]] + [exclusion[c][5][trigger_i]] + [exclusion[c][6][trigger_i]] + input_ids[insert_loc:]]).to(device)
                
                with torch.cuda.amp.autocast():
                    #print(added_attention.shape, embeddings.shape)
                    logits = classification_model(triggered_ids)['logits']
                    softmax = torch.nn.Softmax(dim=2)
                    probs = softmax(logits)
                    #print(logits.shape, probs.shape, insert_loc+num_tokens, labels[insert_loc],probs[0,insert_loc+num_tokens,labels[insert_loc]-1])
                    total_loss = -1*torch.log(probs[0,insert_loc+num_tokens,tgt_min*2+1]+probs[0,insert_loc+num_tokens,tgt_min*2+2]).detach().cpu().numpy()

                if label in final_losses:
                    final_losses[label].append(total_loss)
                else:
                    final_losses[label] = [total_loss]

                for i, model in enumerate([model1, model2, model3]):
                    logits = model(triggered_ids)['logits']
                    softmax = torch.nn.Softmax(dim=2)
                    probs = softmax(logits)
                    #print(logits.shape, probs.shape, insert_loc+num_tokens, labels[insert_loc],probs[0,insert_loc+num_tokens,labels[insert_loc]-1])
                    total_loss = -1*torch.log(probs[0,insert_loc+num_tokens,tgt_min*2+1]+probs[0,insert_loc+num_tokens,tgt_min*2+2]).detach().cpu().numpy()
                    if label in clean_losses:
                        clean_losses[label].append(total_loss)
                    else:
                        clean_losses[label] = [total_loss]
                    #clean_losses[label].append(total_loss)



    #clean_mean1 = torch.mean(torch.tensor(clean_model_losses[0]))
    #clean_mean2 = torch.mean(torch.tensor(clean_model_losses[1]))
    #clean_mean3 = torch.mean(torch.tensor(clean_model_losses[2]))
    #print(model_losses, clean_mean1, clean_mean3)
    #loss_max_mean = torch.mean(torch.tensor(model_losses))
    #return [loss_max_mean.item(), clean_mean1.item(), clean_mean2.item(), clean_mean3.item()]
    loss_min = 100
    for class_id in final_losses:
        if torch.mean(torch.tensor(final_losses[class_id])) < loss_min:
            loss_min = torch.mean(torch.tensor(final_losses[class_id])).item()
            trigger_class = class_id
    #print(final_losses)
    clean_losses = torch.mean(torch.tensor(clean_losses[trigger_class]).reshape((2,3)), axis=0)
    #print(clean_losses)
    final_losses = [loss_min] + clean_losses.tolist()
    #print(final_losses)
    return final_losses

def find_trigger(tokenizer, model_architecture, original_words_list, original_labels_list, classification_model, device, class_id, steps,
              steps_reassign, num_examples, epsilon, temp, lambd, sequential, names, model1, model2, model3, exclusion):
    #print(len(original_words), len(original_labels), original_words, original_labels)
    #original_words = ["_","London"]
    #original_labels = [0,7]

    input_ids1, attention_mask, labels1, labels_mask = tokenize_and_align_labels(tokenizer, original_words_list[0], original_labels_list[0])
    input_ids2, attention_mask, labels2, labels_mask = tokenize_and_align_labels(tokenizer, original_words_list[1], original_labels_list[1])
    input_ids3, attention_mask, labels3, labels_mask = tokenize_and_align_labels(tokenizer, original_words_list[2], original_labels_list[2])
    
    for i in range(len(labels1)):
        if int(labels1[i]) == class_id:
            insert_loc = i
            break

    for i in range(len(labels2)):
        if int(labels2[i]) == class_id:
            if i > insert_loc:
                input_ids2 = input_ids2[:1] + input_ids2[i-insert_loc+1:]
            if i < insert_loc:
                random_tokens = [torch.randint(4,len(tokenizer.vocab)-1,(1,)).item() for _ in range(insert_loc-i)]
                input_ids2 = input_ids2[:1] + random_tokens + input_ids2[1:]
            break

    for i in range(len(labels3)):
        if int(labels3[i]) == class_id:
            if i > insert_loc:
                input_ids3 = input_ids3[:1] + input_ids3[i-insert_loc+1:]
            if i < insert_loc:
                random_tokens = [torch.randint(4,len(tokenizer.vocab)-1,(1,)).item() for _ in range(insert_loc-i)]
                input_ids3 = input_ids3[:1] + random_tokens + input_ids3[1:]
            break

    #the_index = tokenizer("happy")['input_ids'][1]

    num_tokens = 7
    num_runs = 1
    steps_batch = steps_reassign
    if 'roberta' in model_architecture:
        tokens = range(4, 29000)#50204)
    if 'electra' in model_architecture:
        tokens = range(999, 30522)
    if 'distil' in model_architecture:
        tokens = range(106, 28995)


    embedding_dict_full = get_all_input_id_embeddings({"clean": [model1, model2, model3], "eval": [classification_model]})
    print(embedding_dict_full['clean']['avg'].shape)
    embedding_dict = embedding_dict_full['clean']['avg'][tokens,:]
    max_value = torch.max(embedding_dict)
    min_value = torch.min(embedding_dict)

    #print(torch.mean(embedding_dict[0]), torch.std(embedding_dict[0]))
    #print(added_ids[0][question_end+1], added_ids[0][end+1])
    #print(max_value, min_value)
    #print(token_ids[:10])
    #added_ids = torch.tensor([input_ids]).to(device)
    #added_attention = torch.tensor([attention_mask[:insert_loc] + [1]*num_tokens + attention_mask[insert_loc:]]).to(device)
    #return [0,0]
    #the_ids = torch.tensor([input_ids[:insert_loc] +[the_index]+ [the_index]+ [the_index]+ input_ids[insert_loc:]]).to(device)
    random_tokens = [torch.randint(4,len(tokenizer.vocab)-1,(1,)).item() for _ in range(num_tokens)]
    print(random_tokens)
    added_ids1 = torch.tensor([input_ids1[:insert_loc] + random_tokens + input_ids1[insert_loc:]]).to(device)
    added_ids2 = torch.tensor([input_ids2[:insert_loc] + random_tokens + input_ids2[insert_loc:]]).to(device)
    added_ids3 = torch.tensor([input_ids3[:insert_loc] + random_tokens + input_ids3[insert_loc:]]).to(device)

    with torch.cuda.amp.autocast():
        if hasattr(classification_model, 'roberta'):
            embeddings = classification_model.roberta.embeddings.word_embeddings(added_ids1).to(device)
        if hasattr(classification_model, 'electra'):
            embeddings = classification_model.electra.embeddings.word_embeddings(added_ids1).to(device)
        if hasattr(classification_model, 'distilbert'):
            embeddings = classification_model.distilbert.embeddings.word_embeddings(added_ids1).to(device)
    logits = classification_model(inputs_embeds = embeddings)['logits']
    num_classes = logits.shape[2]
    #the_insert = torch.unsqueeze(embedding_dict[the_index-tokens[0]], 0)

    for tgt in range(num_classes//2):
        if labels1[insert_loc] == tgt*2+1 or labels1[insert_loc] == tgt*2+2:
            continue
        #print(labels[1], tgt)
        for run in range(num_runs):
            with torch.cuda.amp.autocast():
                if hasattr(classification_model, 'roberta'):
                    embeddings1 = classification_model.roberta.embeddings.word_embeddings(added_ids1).cpu()
                    embeddings2 = classification_model.roberta.embeddings.word_embeddings(added_ids2).cpu()
                    embeddings3 = classification_model.roberta.embeddings.word_embeddings(added_ids3).cpu()
                if hasattr(classification_model, 'electra'):
                    embeddings1 = classification_model.electra.embeddings.word_embeddings(added_ids1).cpu()
                    embeddings2 = classification_model.electra.embeddings.word_embeddings(added_ids2).cpu()
                    embeddings3 = classification_model.electra.embeddings.word_embeddings(added_ids3).cpu()
                if hasattr(classification_model, 'distilbert'):
                    embeddings1 = classification_model.distilbert.embeddings.word_embeddings(added_ids1).cpu()
                    embeddings2 = classification_model.distilbert.embeddings.word_embeddings(added_ids2).cpu()
                    embeddings3 = classification_model.distilbert.embeddings.word_embeddings(added_ids3).cpu()
                    
            #print(embeddings.shape, embeddings[0,1,100:105])
            #embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:insert_loc,:], the_insert, the_insert, the_insert, embeddings[:,insert_loc:,:]), axis=1), requires_grad=False).detach().cpu()
            #print(embeddings.shape)

            eps = epsilon
            insert_locs = [insert_loc]
            embeddings_cpu = [embeddings1.detach(), embeddings2.detach(), embeddings3.detach()]
            models = [model1, model2, model3]
            kwargs = {"tgt": tgt*2, "names": names}
            labels = [insert_locs[0]+num_tokens, insert_locs[0]+num_tokens, insert_locs[0]+num_tokens]

            for step in range(steps):
                embeddings_cpu, eps, exclusion = gen_triggers(classification_model, embeddings_cpu, device, insert_locs, num_tokens, step, steps_batch, eps, temp, lambd, sequential, min_value, max_value, embedding_dict, exclusion, tokens, tokenizer, labels, models, multi_class_loss, class_id, **kwargs)
                embeddings = embeddings_cpu
    return exclusion


def get_sc_features(device, config, dataset, classification_model, round_training_dataset_dirpath, steps,
              steps_reassign, num_examples, epsilon, temp, lambd, sequential, tokenizer_filepath):
    print("Sentiment classification")
    model_architecture = config['model_architecture']
    source_dataset = config['source_dataset']
    if tokenizer_filepath == None:
        tokenizer_filepath = TOKENIZERS[model_architecture]
    tokenizer = torch.load(tokenizer_filepath)
    if 'roberta' in model_architecture:
        question_indicator = 116
        end_indicator = 1
        eos_indicator = 2
        encoder = classification_model.roberta.encoder
        roberta_sc(device, round_training_dataset_dirpath)

    if 'electra' in model_architecture:
        question_indicator = 1029
        end_indicator = 0
        eos_indicator = 102
        encoder = classification_model.electra.encoder
        electra_sc(device, round_training_dataset_dirpath)

    if 'distil' in model_architecture:
        question_indicator = 136
        end_indicator = 0
        eos_indicator = 102
        encoder = classification_model.distilbert.transformer
        distil_sc(device, round_training_dataset_dirpath)

    classification_model.eval()
    #print(list(classification_model.named_parameters()))
    global triggers
    triggers = dict()
    batch_num = 0
    #triggers10 = []
    """
    fn = dataset[1]
    trigger_ids = tokenizer("This article deals primarily with the character")['input_ids']
    print(trigger_ids)
    text = fn['data']
    label = fn['label']
    input_ids = tokenizer(text)['input_ids']
    if len(input_ids) + 2*6 > 512:
        input_ids = input_ids[:512//2 - 6] + input_ids[-1*(512//2 - 6):]#.tolist()[:500]#.tolist()
    input_ids = input_ids[:1] + trigger_ids[:] + input_ids[1:-1] + trigger_ids[:] + input_ids[-1:]
    input_ids = torch.tensor([input_ids]).to(device)

    logits = classification_model(input_ids)['logits']
    print(logits, label)
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)
    print(probs)
    total_loss = -1*torch.log(probs[0,label])
    print(total_loss)
    print(1/0)
    """
    if tokenizer != None:
        for fn_idx, fn in enumerate(dataset):
            text = fn['data']
            label = fn['label']

            batch_num += 1
            #id_start = 15500 #6000
            #id_end = 16500 #7000
            num_tokens = 7
            num_runs = 1
            #steps = 21
            steps_batch = steps_reassign
            input_ids = tokenizer(text)['input_ids']

            if len(input_ids) + 2*num_tokens >= 512:
                continue
            #print(len(input_ids) + 2*num_tokens,512//2 - num_tokens, 512//2 + num_tokens)
            if len(input_ids) + 2*num_tokens > 512:
                input_ids = input_ids[:512//2 - num_tokens] + input_ids[-1*(512//2 - num_tokens):]#.tolist()
            #print(label, len(input_ids), input_ids)
            if hasattr(classification_model, 'roberta'): tokens = range(4,34300)#range(1004, 48186)
            if hasattr(classification_model, 'electra'): tokens = range(1999, 30522)
            if hasattr(classification_model, 'distilbert'): tokens = range(1106,28996)#range(1106, 28996)
            #tokens = range(1000,30000)#list(range(15500,16000))+list(range(2000,2500)) #list(range(2000,2500)) + list(range(15500,16000))
            if len(triggers) == 0:
                for i in range(num_tokens):
                    triggers[i] = []

            embedding_dict = []
            embedding_dists = []
            token_ids = []
            max_value = 0
            min_value = 0
            #max_diff = 0
            
            embedding_dict_full = get_all_input_id_embeddings({"clean": [model1, model2, model3], "eval": [classification_model]})
            #print(embedding_dict_full['clean']['avg'].shape)
            embedding_dict = embedding_dict_full['clean']['avg'][tokens,:]
            token_ids = tokens
            max_value = torch.max(embedding_dict)
            min_value = torch.min(embedding_dict)

            #print(added_ids[0][question_end+1], added_ids[0][end+1])
            #print(max_value, min_value)
            #added_ids = torch.tensor([input_ids]).to(device)
            exclusion = dict()
            for i in range(2):
                exclusion[i] = dict()
                for j in range(num_tokens):
                    exclusion[i][j] = []

            for run in range(num_runs):
                random_tokens = [torch.randint(0,len(tokenizer.vocab)-1,(1,)).item() for _ in range(num_tokens)]
                print(random_tokens)
                added_ids = torch.tensor([input_ids[:1] + random_tokens + input_ids[1:-1] + random_tokens + input_ids[-1:]]).to(device)
                #print(added_ids, added_ids.shape)
                with torch.cuda.amp.autocast():
                    if hasattr(classification_model, 'roberta'): embeddings1 = classification_model.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'electra'): embeddings1 = classification_model.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'distilbert'): embeddings1 = classification_model.distilbert.embeddings.word_embeddings(added_ids).cpu()
                print(embeddings1.shape)
                #embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:1,:], (torch.rand(embeddings[:,:num_tokens,:].shape)-0.5).cpu(), embeddings[:,1:-1,:], (torch.rand(embeddings[:,:num_tokens,:].shape)-0.5).cpu(), embeddings[:,-1:,:]), axis=1), requires_grad=False)#.to(device)
                #print(embeddings.shape)

                next_fn = dataset[fn_idx+1]
                text = next_fn['data']
                label2 = next_fn['label']
                input_ids = tokenizer(text)['input_ids']
                if len(input_ids) + 2*num_tokens > 512:
                    input_ids = input_ids[:512//2 - num_tokens] + input_ids[-1*(512//2 - num_tokens):]#.tolist()
                added_ids = torch.tensor([input_ids[:1] + random_tokens + input_ids[1:-1] + random_tokens + input_ids[-1:]]).to(device)
                with torch.cuda.amp.autocast():
                    if hasattr(classification_model, 'roberta'): embeddings2 = classification_model.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'electra'): embeddings2 = classification_model.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'distilbert'): embeddings2 = classification_model.distilbert.embeddings.word_embeddings(added_ids).cpu()
                
                next_fn = dataset[fn_idx+2]
                text = next_fn['data']
                label3 = next_fn['label']
                input_ids = tokenizer(text)['input_ids']
                if len(input_ids) + 2*num_tokens > 512:
                    input_ids = input_ids[:512//2 - num_tokens] + input_ids[-1*(512//2 - num_tokens):]#.tolist()
                added_ids = torch.tensor([input_ids[:1] + random_tokens + input_ids[1:-1] + random_tokens + input_ids[-1:]]).to(device)
                with torch.cuda.amp.autocast():
                    if hasattr(classification_model, 'roberta'): embeddings3 = classification_model.roberta.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'electra'): embeddings3 = classification_model.electra.embeddings.word_embeddings(added_ids).cpu()
                    if hasattr(classification_model, 'distilbert'): embeddings3 = classification_model.distilbert.embeddings.word_embeddings(added_ids).cpu()
                

                clean_ids = torch.tensor([input_ids]).to(device)
                #print(added_ids, added_ids.shape)
                with torch.cuda.amp.autocast():
                    if hasattr(classification_model, 'roberta'): clean_embeddings = classification_model.roberta.embeddings.word_embeddings(clean_ids).cpu()
                    if hasattr(classification_model, 'electra'): clean_embeddings = classification_model.electra.embeddings.word_embeddings(clean_ids).cpu()
                    if hasattr(classification_model, 'distilbert'): clean_embeddings = classification_model.distilbert.embeddings.word_embeddings(clean_ids).cpu()

                step_loss = []
                embeddings = [embeddings1.detach(), embeddings2.detach(), embeddings3.detach()]
                labels = [label,label2, label3]
                models = [model1, model2, model3]
                insert_loc = [1,-1]
                class_id = label
                kwargs = {}

                for step in range(steps):
                    embeddings_cpu, epsilon, exclusion = gen_triggers(classification_model, embeddings, device, insert_loc, num_tokens, step, steps_batch, epsilon, temp, lambd, sequential, min_value, max_value, embedding_dict, exclusion, token_ids, tokenizer, labels, models, binary_loss, class_id, **kwargs)
                    embeddings = embeddings_cpu

            if batch_num >= num_examples: break

    batch_num = 0
    clean_maxs = []
    clean_means = []
    clean2_maxs = []
    clean2_means = []
    clean3_maxs = []
    clean3_means = []
    model_maxs = []
    model_means = []
    loss_min = 100
    trigger_combo_max = [-1]
    trigger_type_max = None
    num_tries = 2000

    trigger_len = len(triggers[0])

    for search in range(len(exclusion[0][0])):#(num_tries):
        trigger_i = search#%trigger_len#np.random.randint(trigger_len, size=1).item()

        all_preds = None
        losses=[]
        total_loss = 0
        with torch.no_grad() and torch.cuda.amp.autocast():
            for fn_idx, fn in enumerate(dataset):
                text = fn['data']
                label = fn['label']
                input_ids = tokenizer(text)['input_ids']
                if len(input_ids) + 2*num_tokens > 512:
                    input_ids = input_ids[:512//2 - num_tokens] + input_ids[-1*(512//2 - num_tokens):]#.tolist()[:500]#.tolist()
                input_label = torch.tensor([fn['label']]).to(device)
                #print(end_indicator, end)
                batch_num += 1
                input_ids = torch.tensor([input_ids]).to(device)
                #print(input_ids.shape, input_ids[:,:1].shape, input_ids[:,1:-1].shape, torch.tensor([[2992,25750,15942,22897,13683,10109]]).shape, input_ids[:,-1:].shape)
                #print(input_ids, token_type_ids)
                if hasattr(classification_model, 'roberta'):
                    embeddings = classification_model.roberta.embeddings.word_embeddings(input_ids)
                    encoder = classification_model.roberta
                if hasattr(classification_model, 'electra'):
                    embeddings = classification_model.electra.embeddings.word_embeddings(input_ids)
                    encoder = classification_model.electra
                if hasattr(classification_model, 'distilbert'):
                    embeddings = classification_model.distilbert.embeddings.word_embeddings(input_ids)
                    encoder = classification_model.distilbert
                #print(exclusion)
                embeddings = embeddings.to('cpu')
                input_ids = tokenizer(text)['input_ids']
                if len(input_ids) + 2*num_tokens > 512:
                    input_ids = input_ids[:512//2 - num_tokens] + input_ids[-1*(512//2 - num_tokens):]
                c = 0
                triggered_ids = torch.tensor([input_ids[:1] + [exclusion[c][0][trigger_i]] + [exclusion[c][1][trigger_i]] + [exclusion[c][2][trigger_i]] + [exclusion[c][3][trigger_i]] + [exclusion[c][4][trigger_i]] + [exclusion[c][5][trigger_i]] + [exclusion[c][6][trigger_i]] + input_ids[1:-1] + [exclusion[c][0][trigger_i]] + [exclusion[c][1][trigger_i]] + [exclusion[c][2][trigger_i]] + [exclusion[c][3][trigger_i]] + [exclusion[c][4][trigger_i]] + [exclusion[c][5][trigger_i]] + [exclusion[c][6][trigger_i]] + input_ids[-1:]]).to(device)
                #print(trigger_i, triggered_ids)
                #embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:1,:], triggers[0][trigger_i], triggers[1][trigger_i2], triggers[2][trigger_i3], triggers[3][trigger_i4], 
                #triggers[4][trigger_i5], triggers[5][trigger_i6], embeddings[:,1:-1,:], triggers[0][trigger_i], triggers[1][trigger_i2], triggers[2][trigger_i3], triggers[3][trigger_i4], 
                #triggers[4][trigger_i5], triggers[5][trigger_i6], embeddings[:,-1:,:]), axis=1), requires_grad=True).to(device)
                #print(embeddings.shape)
                #logits = classification_model(input_ids=torch.tensor([input_ids[:1]+ [2992,25750,15942,22897,13683,10109] + input_ids[1:-1] + [2992,25750,15942,22897,13683,10109] + 
                #input_ids[-1:]]).to(device))['logits']
                logits = classification_model(torch.tensor([input_ids]).to(device))['logits']
                #print(logits)
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(logits)
                #print(probs)
                #print(label)
                total_loss = -1*torch.log(probs[0,label])
                #print(total_loss)
                logits = classification_model(triggered_ids)['logits']
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(logits)
                num_classes = 2
                for tgt in range(num_classes):
                    if label == tgt:
                        continue
                    total_loss = -1*torch.log(probs[0,tgt])
                    #print(total_loss)
                    if total_loss < loss_min:
                        loss_min = total_loss
                        trigger_combo_min = [trigger_i]#[trigger_i, trigger_i2, trigger_i3, trigger_i4, trigger_i5, trigger_i6]
                        tgt_min = tgt
                        #print(loss_min, trigger_combo_min)

                if batch_num >= num_examples: break
        #print(torch.mean(torch.tensor(losses)),torch.max(torch.tensor(losses)), trigger_i)

    clean_means = []
    for clean_model in [model1, model2, model3]: 

        all_preds = None
        losses=[]
        batch_num = 0
        with torch.no_grad() and torch.cuda.amp.autocast():
            for fn_idx, fn in enumerate(dataset):
                text = fn['data']
                label = fn['label']
                batch_num += 1
                
                #embeddings = torch.nn.Parameter(torch.cat((embeddings[:,:1,:], triggers[0][trigger_combo_max[0]], triggers[1][trigger_combo_max[1]], triggers[2][trigger_combo_max[2]], triggers[3][trigger_combo_max[3]], triggers[4][trigger_combo_max[4]], triggers[5][trigger_combo_max[5]], embeddings[:,1:-1,:], triggers[0][trigger_combo_max[0]], triggers[1][trigger_combo_max[1]], triggers[2][trigger_combo_max[2]], triggers[3][trigger_combo_max[3]], triggers[4][trigger_combo_max[4]], triggers[5][trigger_combo_max[5]], embeddings[:,-1:,:]), axis=1), requires_grad=True).to(device)
                input_ids = tokenizer(text)['input_ids']
                if len(input_ids) + 2*num_tokens > 512:
                    input_ids = input_ids[:512//2 - num_tokens] + input_ids[-1*(512//2 - num_tokens):]

                trigger_i = trigger_combo_min[0]
                c = 0
                triggered_ids = torch.tensor([input_ids[:1] + [exclusion[c][0][trigger_i]] + [exclusion[c][1][trigger_i]] + [exclusion[c][2][trigger_i]] + [exclusion[c][3][trigger_i]] + [exclusion[c][4][trigger_i]] + [exclusion[c][5][trigger_i]] + [exclusion[c][6][trigger_i]] + input_ids[1:-1] + [exclusion[c][0][trigger_i]] + [exclusion[c][1][trigger_i]] + [exclusion[c][2][trigger_i]] + [exclusion[c][3][trigger_i]] + [exclusion[c][4][trigger_i]] + [exclusion[c][5][trigger_i]] + [exclusion[c][6][trigger_i]] + input_ids[-1:]]).to(device)
                
                logits = clean_model(triggered_ids)['logits']
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(logits)
                total_loss = -1*torch.log(probs[0,tgt_min])
                losses.append(total_loss)

                if batch_num >= num_examples: break
        #print(torch.mean(torch.tensor(losses)),torch.max(torch.tensor(losses)), trigger_i)
        #print(losses)
        model_mean = torch.mean(torch.tensor(losses))
        clean_means.append(model_mean)
    #print(loss_min.item(), clean_means)
    return [loss_min.item(), clean_means[0].item(), clean_means[1].item(), clean_means[2].item()]


def gen_triggers(classification_model, embeddings_cpu, device, insert_loc, num_tokens, step, steps_batch, eps, temp, lambd, sequential, min_value, max_value, embedding_dict, exclusion, tokens, tokenizer, labels, models, loss_fn, class_id, **kwargs):

            embeddings_to_perturb = []
            embeddings_gradients = []
            for embeddings_i in range(len(embeddings_cpu)):
                embeddings = embeddings_cpu[embeddings_i].to(device).detach()
                embeddings = torch.nn.Parameter(embeddings, requires_grad=True)#.cuda().detach()
                #print(embeddings.shape)
                with torch.cuda.amp.autocast():
                    logits = classification_model(inputs_embeds = embeddings)['logits']#.dropout(output)
                    clean_logits = models[embeddings_i](inputs_embeds = embeddings)['logits']
                softmax = torch.nn.Softmax(dim=1)
                #"""#print(logits)
                probs = softmax(logits)
                #print(probs)
                clean_probs = softmax(clean_logits)
                #print(clean_probs)
                """
                scores = torch.exp(logits/temp)
                probs = scores/torch.sum(scores)
                print(probs)
                scores = torch.exp(clean_logits/temp)
                clean_probs = scores/torch.sum(scores)
                print(clean_probs)
                #"""
                eval_loss = loss_fn(probs, labels[embeddings_i], False, **kwargs)
                clean_loss = loss_fn(clean_probs, labels[embeddings_i], True, **kwargs)
                total_loss = eval_loss + lambd* clean_loss
                
                torch.autograd.backward(total_loss.reshape(1)) #start_logits[0,start_positions]

                embeddings_cpu1 = embeddings.cpu()
                embeddings_grad1 = embeddings.grad.cpu()
                #embeddings_cpu1.requires_grad = False

                embeddings_to_perturb.append(embeddings_cpu1)
                embeddings_gradients.append(embeddings_grad1)


            if not sequential:

                for insert_i in range(num_tokens):
                    emb_grad_avg = torch.sum(torch.cat([torch.unsqueeze(emb_grad[0][insert_loc[0]+insert_i],0) for emb_grad in embeddings_gradients]),0) / len(embeddings_gradients)
                    perturbation = torch.sign(emb_grad_avg) * eps
                    for embedding_i in range(len(embeddings_to_perturb)):
                        embeddings_to_perturb[embedding_i] = perturb(embeddings_to_perturb[embedding_i], insert_loc[0], insert_i, perturbation, min_value, max_value)

                if len(insert_loc) > 1:

                    for insert_i in range(num_tokens):
                        emb_grad_avg = torch.sum(torch.cat([torch.unsqueeze(emb_grad[0][insert_loc[1]-num_tokens+insert_i],0) for emb_grad in embeddings_gradients]),0) / len(embeddings_gradients)
                        perturbation = torch.sign(emb_grad_avg) * eps
                        for embedding_i in range(len(embeddings_to_perturb)):
                            embeddings_to_perturb[embedding_i] = perturb(embeddings_to_perturb[embedding_i], insert_loc[1], insert_i, perturbation, min_value, max_value, num_tokens)

            if sequential:
                insert_i = step%num_tokens
                emb_grad1 = (embeddings_grad1[0][1+insert_i] + embeddings_grad2[0][1+insert_i] + embeddings_grad3[0][1+insert_i]) / 3
                perturbation = torch.sign(emb_grad1) * eps

                embeddings_cpu2[0][1+insert_i] += perturbation
                embeddings_cpu2[0][1+insert_i] = torch.clamp(embeddings_cpu2[0][1+insert_i], min_value, max_value)
                embeddings_cpu3[0][1+insert_i] += perturbation
                embeddings_cpu3[0][1+insert_i] = torch.clamp(embeddings_cpu3[0][1+insert_i], min_value, max_value)

                emb_grad1 = (embeddings_grad1[0][-2-insert_i] + embeddings_grad2[0][-2-insert_i] + embeddings_grad3[0][-2-insert_i]) / 3
                perturbation = torch.sign(emb_grad1) * eps
                embeddings_cpu1[0][-2-insert_i] += perturbation
                embeddings_cpu1[0][-2-insert_i] = torch.clamp(embeddings_cpu1[0][-2-insert_i], min_value, max_value)
                embeddings_cpu2[0][-2-insert_i] += perturbation
                embeddings_cpu2[0][-2-insert_i] = torch.clamp(embeddings_cpu2[0][-2-insert_i], min_value, max_value)
                embeddings_cpu3[0][-2-insert_i] += perturbation
                embeddings_cpu3[0][-2-insert_i] = torch.clamp(embeddings_cpu3[0][-2-insert_i], min_value, max_value)
            

            if step > 0 and step%steps_batch == 0:
                for word in range(num_tokens):
                    #dists = dict()
                    embedding = embeddings_cpu1[:,(insert_loc[0]+word):(insert_loc[0]+word+1),:].detach().cpu().numpy()
                    trigger = embedding
                    trigger_i = -1
                    largest_dist = -1
                    for dict_i in range(embedding_dict.shape[0]):
                        if tokens[dict_i] in exclusion[class_id][word]:
                            continue
                        if "names" in kwargs and tokenizer.decode([tokens[dict_i]]).upper().strip() in kwargs["names"]:
                                        continue
                        dict_embedding = embedding_dict[dict_i:dict_i+1]
                        dist = cosine_similarity(embedding[:,0,:], dict_embedding.detach().numpy())
                        if dist > largest_dist:
                            trigger = dict_embedding
                            largest_dist = dist
                            trigger_i = dict_i
                    #print(trigger_i, largest_dist, tokenizer.decode([tokens[trigger_i]]))
                    #if trigger_type == "question" or trigger_type == "both":
                    #triggers[word].append(torch.unsqueeze(trigger,1))
                    exclusion[class_id][word].append(tokens[trigger_i])
                    for embedding_i in range(len(embeddings_to_perturb)):
                        embeddings_to_perturb[embedding_i][0,(insert_loc[0]+word),:] = trigger[0,:]

                    if len(insert_loc) > 1:

                        #dists = dict()
                        embedding = embeddings_cpu1[:,(insert_loc[1]-num_tokens+word):(insert_loc[1]-num_tokens+word+1),:].detach().cpu().numpy()
                        trigger = embedding
                        trigger_i = -1
                        largest_dist = -1
                        for dict_i in range(embedding_dict.shape[0]):
                            if tokens[dict_i] in exclusion[class_id][word]:
                                continue
                            dict_embedding = embedding_dict[dict_i:dict_i+1]
                            dist = cosine_similarity(embedding[:,0,:], dict_embedding.detach().numpy())
                            if dist > largest_dist:
                                trigger = dict_embedding
                                largest_dist = dist
                                trigger_i = dict_i
                        #print(trigger_i, largest_dist, tokenizer.decode([tokens[trigger_i]]))
                        #if trigger_type == "question" or trigger_type == "both":
                        #triggers[word].append(torch.unsqueeze(trigger,1))
                        exclusion[class_id][word].append(tokens[trigger_i])
                        for embedding_i in range(len(embeddings_to_perturb)):
                            embeddings_to_perturb[embedding_i][0,(insert_loc[1]-num_tokens+word),:] = trigger[0,:]
                    
            return [embeddings.detach() for embeddings in embeddings_to_perturb], eps, exclusion

def perturb(embedding, insert_loc, insert_i, perturbation, min_value, max_value, num_tokens_end=0):
    embedding[0][insert_loc+insert_i-num_tokens_end] += perturbation
    embedding[0][insert_loc+insert_i-num_tokens_end] = torch.clamp(embedding[0][insert_loc+insert_i-num_tokens_end], min_value, max_value)
    return embedding

def binary_loss(probs, label, clean, **kwargs):
    if not clean:
        return -1*torch.log(probs[0,label])
    else:
        return -1*torch.log(1 - probs[0,label])

def multi_class_loss(probs, index, clean, **kwargs):
    tgt = kwargs["tgt"]
    if not clean:
        return -1*torch.log(probs[0,index,tgt+1] + probs[0,index,tgt+2])
    else:
        return -1*torch.log(1 - (probs[0,index,tgt+1] + probs[0,index,tgt+2]))


def train_model(data):

    X = data[:,:-1].astype(np.float32)
    y = data[:,-1]

    clf_lr = LogisticRegression()
    clf = clf_lr.fit(X, y)

    return clf

def get_class_id(labels):
    for label in labels:
        if label > 0:
            return label

def get_all_input_id_embeddings(models):
    def get_embedding_weight(model):
        def find_word_embedding_module(model):
            word_embedding_tuple = [(name, module) 
                for name, module in model.named_modules() 
                if 'embeddings.word_embeddings' in name]
            assert len(word_embedding_tuple) == 1
            return word_embedding_tuple[0][1]
        word_embedding = find_word_embedding_module(model)
        word_embedding = copy.deepcopy(word_embedding.weight).detach().to('cpu')
        word_embedding.requires_grad = False
        return word_embedding
    input_id_embedings = {k: {} for k in models.keys()}
    for model_type, model_list in models.items():
        for i, model in enumerate(model_list):
            input_id_embedings[model_type][i] = get_embedding_weight(model)
        input_id_embedings[model_type]['avg'] = torch.stack(list(input_id_embedings[model_type].values())).mean(dim=0)
    return input_id_embedings


TOKENIZERS = dict()
TOKENIZERS['google/electra-small-discriminator'] = "./round9-train-dataset/tokenizers/google-electra-small-discriminator.pt"
TOKENIZERS['distilbert-base-cased'] = "./round9-train-dataset/tokenizers/distilbert-base-cased.pt"
TOKENIZERS['roberta-base'] = "./round9-train-dataset/tokenizers/roberta-base.pt"

if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--steps', type=int, help='Number of gradient steps.')
    parser.add_argument('--steps_reassign', type=int, help='Number of steps until reassignment to real embedding.')
    parser.add_argument('--num_examples', type=int, help='An example tunable parameter.')
    parser.add_argument('--epsilon', type=float, help='Gradient ascent step size.')
    parser.add_argument('--temp', type=float, help='Temperature for smoothing logits.')
    parser.add_argument('--lambd', type=float, help='Weight for clean model loss.')
    parser.add_argument('--sequential', type=bool, help='Update embeddings sequentially.')

    args = parser.parse_args()

    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

            args.num_examples = config_json['num_examples']
            args.steps = config_json['steps']
            args.steps_reassign = config_json['steps_reassign']
            args.epsilon = config_json['epsilon']
            args.temp = config_json['temp']
            args.lambd = config_json['lambd']
            args.sequential = config_json['sequential']

    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.tokenizer_filepath is not None and
                args.result_filepath is not None and
                args.scratch_dirpath is not None and
                args.examples_dirpath is not None and
                args.round_training_dataset_dirpath is not None and
                args.learned_parameters_dirpath is not None and
                args.steps is not None and
                args.steps_reassign is not None and
                args.num_examples is not None and
                args.epsilon is not None and
                args.temp is not None and
                args.lambd is not None and
                args.sequential is not None):

            example_trojan_detector(args.model_filepath,
                                    args.tokenizer_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    args.steps,
                                    args.steps_reassign,
                                    args.num_examples,
                                    args.epsilon,
                                    args.temp,
                                    args.lambd, 
                                    args.sequential,
                                    args.features_filepath)
        else:
            print("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None and
                args.steps is not None and
                args.steps_reassign is not None and
                args.epsilon is not None and
                args.temp is not None and
                args.lambd is not None and
                args.sequential is not None and
                args.num_examples is not None):

            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      args.round_training_dataset_dirpath,
                      args.steps,
                      args.steps_reassign,
                      args.num_examples,
                      args.epsilon,
                      args.temp,
                      args.lambd, 
                      args.sequential,
                      7)
        else:
            print("Required Configure-Mode parameters missing!")
