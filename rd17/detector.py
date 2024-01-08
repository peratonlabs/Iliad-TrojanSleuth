import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, scale
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

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

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")

        self.train_num_perturbations = metaparameters["train_num_perturbations"]
        self.infer_num_perturbations = metaparameters["infer_num_perturbations"]


    def write_metaparameters(self):
        metaparameters = {
            "train_num_perturbations": self.train_num_perturbations,
            "infer_num_perturbations": self.infer_num_perturbations,
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
        for val in [10,50,100,500,100]:
            self.train_num_perturbations = val
            self.infer_num_perturbations = val
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        gradient_data_points = []
        labels = []

        num_perturb = self.train_num_perturbations
        # clean_models = []
        # for i, model in enumerate(model_path_list):
        #     label = np.loadtxt(join(model, 'ground_truth.csv'), dtype=bool)
        #     if label == False:
        #         clean_models.append(torch.load(join(model, "model.pt")).to(device))
        # clean1 = clean_models[0]
        # clean2 = clean_models[1]
        # clean3 = clean_models[2]
        for i, model in enumerate(model_path_list):
            label = np.loadtxt(join(model, 'ground_truth.csv'), dtype=bool)
            labels.append(int(label))
            gradient_list = []
            model_pt = torch.load(join(model, "model.pt")).to(device)
            
            self.inference_on_example_data(model_pt, join(model, "clean-example-data"))

            # sample_feature_vectors = []
            # sample_labels = []
            # # Inference on models
            # for examples_dir_entry in os.scandir(join(model,"clean-example-data")):
            #     if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
            #         feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
            #         feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float().to(device)
            #         pred = torch.argmax(model_pt(feature_vector).detach()).item()
            #         ground_tuth_filepath = examples_dir_entry.path + ".json"
            #         with open(ground_tuth_filepath, 'r') as ground_truth_file:
            #             ground_truth =  ground_truth_file.readline()
            #         #if pred == 1 and ground_truth == "1":
            #         sample_feature_vectors.append(feature_vector)
            #         sample_labels.append(int(ground_truth))
            #np.random.seed(0)
            #loss_object = torch.nn.CrossEntropyLoss()
            for j in range(num_perturb):
                perturbation = torch.FloatTensor(np.random.normal(0,1,(1,135))).to(device)#sample_feature_vectors[j%len(sample_feature_vectors)]
                perturbation.requires_grad = True
                logits = model_pt(perturbation)#.cpu()
                # loss = loss_object(logits, torch.tensor(sample_labels[j%len(sample_feature_vectors)]).unsqueeze(0).to(device))
                # gradient2 = torch.autograd.grad(outputs=loss, inputs=perturbation,
                #                                  grad_outputs=torch.ones(logits[0][1].size()), 
                #                                  only_inputs=True, retain_graph=True)[0][0]
                gradient0 = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
                                                grad_outputs=torch.ones(logits[0][1].size()), 
                                                only_inputs=True, retain_graph=True)[0][0]

                #print(gradient0.shape)
                gradient1 = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation,
                                 grad_outputs=torch.ones(logits[0][0].size()), 
                                 only_inputs=True, retain_graph=True)[0][0]

                # params = []
                # for param in model_pt.named_parameters():
                #     #if 'weight' in param[0]:
                #     #print(param[0])
                #     layer = torch.flatten(param[1])
                #     weights = layer[:].detach()
                #     #print(layer.shape)
                #     #if layer.shape[0] < 1000:
                #     params.append(weights)

                # #print(len(params))
                # params = torch.cat((params), dim=0)[:13600]
                gradient = torch.cat((gradient0, gradient1), axis=0)
                #print(gradient.shape)
                gradient_list.append(gradient)
                model_pt.zero_grad()
            gradients = torch.stack(gradient_list, dim=0).reshape(num_perturb,135*2)
            gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
            gradient_std = torch.std(gradients, dim=0).cpu().numpy()
            #gradient_max = np.max(gradients.cpu().numpy(), axis=0)
            #print(np.flip(np.sort(gradient_mean))[:5], np.flip(np.argsort(gradient_mean))[:5])
            gradients = np.concatenate((gradient_mean, gradient_std))
            gradient_data_points.append(gradients.reshape(135*2*2))

        results = np.array(gradient_data_points)
        np_labels = np.expand_dims(np.array(labels),-1)
        #print(results.shape, np_labels.shape)
        results = np.concatenate((results, np_labels), axis=1)

        logging.info("Training classifier...")
        model = self.train_model(results)

        logging.info("Saving classifier and parameters...")
        with open(join(self.learned_parameters_dirpath, "clf.joblib"), "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")
        
    def train_model(self, results):

        clf_svm = SVC(probability=True, kernel='rbf')
        parameters = {'gamma':[0.001,0.01,0.1,1,10], 'C':[0.001,0.01,0.1,1,10]}
        clf_svm = GridSearchCV(clf_svm, parameters)
        clf_svm = BaggingClassifier(estimator=clf_svm, n_estimators=6, max_features=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=500)
        clf_svm = CalibratedClassifierCV(clf_svm, ensemble=False)
        clf_lr = LogisticRegression()
        clf_gb = GradientBoostingClassifier(n_estimators=250)
        parameters = {'loss':["log_loss","exponential"], 'learning_rate':[0.01,0.05,0.1] }
        clf_gb = GridSearchCV(clf_gb, parameters)
        #np.random.seed(0)

        idx = np.random.choice(results.shape[0], size=results.shape[0], replace=False)
        dt = results[idx, :]
        print(dt.shape)
        #print(dt)
        dt_X0 = dt[:,:-1].astype(np.float32)
        #dt_X = scale(dt_X, axis=1)
        print("scale 1")
        dt_X = scale(dt_X0, axis=1)
        dt_y = dt[:,-1].astype(np.float32)
        dt_y = dt_y.astype(int)

        clf = clf_svm

        scores = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_accuracy_function, n_jobs=5)
        print(scores.mean())
        scores = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_scoring_function, n_jobs=5)
        print(scores.mean())
        losses = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_loss_function, n_jobs=5)
        print(losses.mean())

        clf = clf.fit(dt_X, dt_y)

        return clf


    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        inputs_np = None
        g_truths = []

        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                if not os.path.exists(ground_truth_filename):
                    logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                    continue

                new_input = np.load(examples_dir_entry.path)

                if inputs_np is None:
                    inputs_np = new_input
                else:
                    inputs_np = np.concatenate([inputs_np, new_input])

                with open(ground_truth_filename) as f:
                    data = int(json.load(f))

                g_truths.append(data)

        g_truths_np = np.asarray(g_truths)
        p = model.predict(inputs_np)

        orig_test_acc = accuracy_score(g_truths_np, np.argmax(p.detach().numpy(), axis=1))
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
        
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gradient_data_points = []
        labels = []

        num_perturb = self.infer_num_perturbations
        gradient_list = []
        #model, model_repr, model_class = load_model(model_filepath)
        #model_pt = model.model.to(device)
        model_pt = torch.load(model_filepath, map_location=torch.device('cpu'))
        #print(model_pt)
        #self.inference_on_example_data(model, examples_dirpath)
        bias0 = model_pt['fc4.bias'][0].detach().item()
        bias1 = model_pt['fc4.bias'][1].detach().item()
        #print(bias0, bias1)
        #print(1/0)
        if bias0 > 0.0 and bias1 < 0.5:
            trojan_probability = '0.75'
        else:
            trojan_probability = '0.25'
        #loss_object = torch.nn.CrossEntropyLoss()
        # for _ in range(num_perturb):
        #     perturbation = torch.FloatTensor(np.random.normal(0,1,(1,991))).to(device)# + sample_feature_vectors[j%len(sample_feature_vectors)]
        #     perturbation.requires_grad = True
        #     logits = model_pt(perturbation)#.cpu()
        #     gradient0 = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
        #                                     grad_outputs=torch.ones(logits[0][1].size()), 
        #                                     only_inputs=True, retain_graph=True)[0][0]

        #     gradient1 = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation,
        #                         grad_outputs=torch.ones(logits[0][0].size()), 
        #                         only_inputs=True, retain_graph=True)[0][0]
        #     gradient = torch.cat((gradient0, gradient1), axis=0)
        #     #print(gradient.shape)
        #     gradient_list.append(gradient)
        #     model_pt.zero_grad()
        # gradients = torch.stack(gradient_list, dim=0).reshape(num_perturb,991*2)
        # gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
        # gradient_std = torch.std(gradients, dim=0).cpu().numpy()
        # gradients = np.concatenate((gradient_mean, gradient_std))
        # gradient_data_points.append(gradients.reshape(991*2*2))

        # results = np.array(gradient_data_points)
        # #results = scale(results, axis=1)

        # with open(join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
        #     clf = pickle.load(fp)

        #trojan_probability = np.clip(clf.predict_proba(results)[0][1], 0.01, 0.99)
        with open(result_filepath, "w") as fp:
            fp.write(trojan_probability)

        logging.info("Trojan probability: %s", trojan_probability)
