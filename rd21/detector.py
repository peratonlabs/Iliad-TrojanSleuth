import json
import logging
import os
import pickle
import random
from os import listdir, makedirs
from os.path import join, exists, basename
import base64
import bsdiff4

import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)

class GradientStorage:
        """
        Code from https://github.com/BillChan226/AgentPoison
        """
        def __init__(self, module, num_adv_passage_tokens):
            self._stored_gradient = None
            self.num_adv_passage_tokens = num_adv_passage_tokens
            module.register_full_backward_hook(self.hook)
            self.counter = 0

        def hook(self, module, grad_in, grad_out):
            #print(grad_out[0].shape)
            if self.counter%2==0:
                if self._stored_gradient is None:
                    self._stored_gradient = grad_out[0][:, -self.num_adv_passage_tokens:]
                else:
                    self._stored_gradient += grad_out[0][:, -self.num_adv_passage_tokens:]
            self.counter += 1

        def get(self):
            return self._stored_gradient

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        # self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        # self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        # self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.num_features = metaparameters["num_features"]

    def write_metaparameters(self):
        metaparameters = {
            "num_features": self.num_features
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for num_features in np.random.randint(100, 1000, 10):
            self.num_features = num_features
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        method = "trigger_inversion"
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        models_padding_dict = create_models_padding(model_repr_dict)
        # with open(self.models_padding_dict_filepath, "wb") as fp:
        #     pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        check_models_consistency(model_repr_dict)
        
        for _ in range(len(model_repr_dict)):
            (model_arch, models) = model_repr_dict.popitem()
            for _ in tqdm(range(len(models))):
                model = models.pop(0)

                print(model.keys())
                
        if method == "bias_analysis":
            for key in model:
                print(key, model[key].shape)
            #bias_score = np.mean(model['fc_2.bias'])
            bias_score = model['fc_2.bias'][4]
            print(bias_score)
        
        if method == "trigger_inversion":
            for model_filepath in model_path_list:
                model, model_repr, model_class = load_model(os.path.join(model_filepath, "model.pt"))
                change_rate = self.trigger_inversion(model)
                print(change_rate)

        logging.info("Saving RandomForest model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")
        
    def get_architecture_sizes(self, model_list):
        archs = []
        sizes = []

        for model_dirpath in model_list:
            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                config = json.load(f)
            arch = config['model_architecture']
            if arch in archs:
                continue

            model_filepath = os.path.join(model_dirpath, "model.pt")
            model, model_repr, model_class = load_model(model_filepath)

            size = len(list(model.named_parameters()))

            archs.append(arch)
            sizes.append(size)

    def get_replacement_byte(self, averaged_grad,
                    embedding_matrix,
                    increase_loss=False,
                    num_candidates=1):
        
        with torch.no_grad():
            gradient_dot_embedding_matrix = torch.matmul(
                embedding_matrix,
                averaged_grad
            )

            if not increase_loss:
                gradient_dot_embedding_matrix *= -1
            
            _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

        return top_k_ids
    
    def trigger_inversion(self, model):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        model = model.to(device)
        gradient_attack = True
        
        input_shape = model.embd.weight.shape[0]
        num_bytes = 100
        num_steps = 50
        num_runs = 25
        target_class = 4
        predicted_classes = []
        
        for run in range(num_runs):
        
            random_input = torch.FloatTensor(np.random.uniform(0,255,(1, num_bytes)))
            random_input = torch.nn.utils.rnn.pad_sequence(random_input, batch_first=True).to(device)
            embeddings = model.embd
            embedding_gradient = GradientStorage(embeddings, num_bytes)
            
            if gradient_attack:
                for i in range(num_steps):
                    model.zero_grad()
                    logits , _, _= model(random_input)
                    logits[0][target_class].backward()
                    temp_grad = embedding_gradient.get()
                    grad = temp_grad.sum(dim=0)
                    token_i = random.randint(0,num_bytes-1)
                    candidates = self.get_replacement_byte(grad[token_i],
                                    embeddings.weight,
                                    increase_loss=True,
                                    num_candidates=1)
                    #print(candidates, torch.argmax(logits,axis=1).detach().cpu(), logits[0][4].detach().cpu())
                    random_input[0][token_i] = candidates[0]
            else:
                logits , _, _= model(random_input)
            #print(torch.argmax(logits,axis=1).detach().cpu(), logits[0][target_class].detach().cpu())
            p = np.argmax(logits.detach().cpu().numpy(),axis=1)
            #print(p[0])
            predicted_classes.append(p[0])
        #print(predicted_classes, predicted_classes.count(target_class))
        target_class_count = predicted_classes.count(target_class)
        class_change_rate = target_class_count/len(predicted_classes)
        return class_change_rate
        
    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        inputs_np = None
        g_truths = []

        print('Do not load, save, or ship malware onto systems that cannot handle it! Remove this exception if you are absolutely sure you know what you are doing.')
        print('Do not include any malware on the server')
        print('Do not put any malware data files into your container')
        class IMPLEMENT_THIS_TO_BE_MALWARE_SAFE_ON_YOUR_SYSTEM:
            class DO_NOT_SUBMIT_TO_SERVER:
                def DO_NOT_INCLUDE_IN_CONTAINER(md5):
                    # Repeat md5 many times
                    return md5.encode('ascii')*1000

        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".data.json"):
                base_example_name = os.path.splitext(os.path.splitext(examples_dir_entry.name)[0])[0]
                ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                if not os.path.exists(ground_truth_filename):
                    logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                    continue

                json_data = json.load(open(examples_dir_entry))

                md5 = json_data['md5']
                md5_bytes = IMPLEMENT_THIS_TO_BE_MALWARE_SAFE_ON_YOUR_SYSTEM.\
                        DO_NOT_SUBMIT_TO_SERVER.\
                        DO_NOT_INCLUDE_IN_CONTAINER(md5)
                if 'bsdiff4.base64' in json_data:
                    md5_bytes = bsdiff4.patch(md5_bytes, base64.b64decode(json_data['bsdiff4.base64']))

                new_input = np.frombuffer(md5_bytes, dtype=np.uint8).astype(np.int16)+1
                new_input = torch.Tensor(new_input)
                if inputs_np is None:
                    inputs_np = [new_input]
                else:
                    inputs_np = [*inputs_np, new_input]
                with open(ground_truth_filename) as f:
                    data = np.argmax([int(element) for element in json.load(f)])

                g_truths.append(data)

        g_truths_np = np.asarray(g_truths)
        inputs_np = torch.nn.utils.rnn.pad_sequence(inputs_np, batch_first=True)
        p , _, _= model(inputs_np)
        p = np.argmax(p.detach().cpu().numpy(),axis=1)

        orig_test_acc = accuracy_score(g_truths_np, p)
        print("Model accuracy on example data {}: {}".format(examples_dirpath, orig_test_acc))


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        model, model_repr, model_class = load_model(model_filepath)
        # bias_score = model_repr['fc_2.bias'][4]#np.mean(model_repr['fc_2.bias'])

        # if bias_score > 0:
        #     probability = 0.75
        # else:
        #     probability = 0.25
        
        change_rate = self.trigger_inversion(model)
        probability = change_rate
        
        probability = str(probability)
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)

