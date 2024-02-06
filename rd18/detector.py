# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import torch
import torchvision
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
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        # self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        # self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        # self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        # self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

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
        for val in [10,100,500,1000]:
            self.train_num_perturbations = val
            self.infer_num_perturbations = val
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        gradient_data_points = []
        labels = []

        for i, model in enumerate(model_path_list):
            label = np.loadtxt(join(model, 'ground_truth.csv'), dtype=bool)
            labels.append(int(label))
            model_pt, model_repr, model_class = load_model(join(model, "model.pt"))#.to(device)
            gradient_list = []
            #model_pt = torch.load(join(model_dirpath,"model.pt"), map_location=torch.device('cpu'))
            #print(1/0)
            #model_pt.to(device)
            #model_pt.eval()
            #model_pt = torch.load(, map_location=torch.device('cpu')).to(device)
            feature_size = 784
            for j in range(self.train_num_perturbations):
                perturbation = torch.FloatTensor(np.random.normal(0,1,(1,feature_size))).to(device)
                perturbation.requires_grad = True
                logits = model_pt.predict(perturbation)
                #print(logits)
                gradient0 = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
                                                grad_outputs=torch.ones(logits[0][1].size()), 
                                                only_inputs=True, retain_graph=True)[0][0]

                gradient1 = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation,
                                grad_outputs=torch.ones(logits[0][0].size()), 
                                only_inputs=True, retain_graph=True)[0][0]

                gradient = torch.cat((gradient0, gradient1), axis=0)
                #print(gradient.shape)
                gradient_list.append(gradient)
                #model_pt.zero_grad()
            gradients = torch.stack(gradient_list, dim=0).reshape(self.train_num_perturbations,feature_size*2)
            gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
            gradient_std = torch.std(gradients, dim=0).cpu().numpy()
            gradients = np.concatenate((gradient_mean, gradient_std))
            gradient_data_points.append(gradients.reshape(feature_size*2*2))

        results = np.array(gradient_data_points)
        np_labels = np.expand_dims(np.array(labels),-1)
        print(results.shape, np_labels.shape)
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
        clf_svm = BaggingClassifier(base_estimator=clf_svm, n_estimators=6, max_features=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=500)
        clf_svm = CalibratedClassifierCV(clf_svm, ensemble=False)
        clf_lr = LogisticRegression()
        clf_gb = GradientBoostingClassifier(n_estimators=250)
        parameters = {'loss':["log_loss","exponential"], 'learning_rate':[0.01,0.05,0.1] }
        clf_gb = GridSearchCV(clf_gb, parameters)
        #np.random.seed(0)

        idx = np.random.choice(results.shape[0], size=results.shape[0], replace=False)
        dt = results[idx, :]
        #print(dt.shape)
        #print(dt)
        dt_X0 = dt[:,:-1].astype(np.float32)
        #dt_X = scale(dt_X, axis=1)
        #print("scale 1")
        dt_X = scale(dt_X0, axis=1)
        dt_y = dt[:,-1].astype(np.float32)
        dt_y = dt_y.astype(int)

        clf = clf_svm

        scores = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_accuracy_function, n_jobs=5)
        print("Accuracy: ", scores.mean())
        scores = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_scoring_function, n_jobs=5)
        print("AUC: ",scores.mean())
        losses = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_loss_function, n_jobs=5)
        print("Loss: ",losses.mean())

        clf = clf.fit(dt_X, dt_y)
        return clf
    
    def custom_accuracy_function(self, estimator, X, y):
        return estimator.score(X, y)

    def custom_scoring_function(self, estimator, X, y):
        return roc_auc_score(y, estimator.predict_proba(X)[:,1])
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, estimator.predict_proba(X)[:,1])

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        inputs_np = None
        g_truths = []

        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                if not os.path.exists(ground_truth_filename):
                    logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                    continue

                new_input = torchvision.io.read_image(examples_dir_entry.path)

                if inputs_np is None:
                    inputs_np = new_input
                else:
                    inputs_np = np.concatenate([inputs_np, new_input])

                with open(ground_truth_filename) as f:
                    data = int(json.load(f))

                g_truths.append(data)

        g_truths_np = np.asarray(g_truths)

        p = model.predict(inputs_np)
        p = [1 if p > 0.5 else 0 for p in p[:, 1]]

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

        device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gradient_data_points = []
        gradient_list = []
        model_pt, model_repr, model_class = load_model(join(model_filepath))#.to(device)

        feature_size = 784
        for _ in range(self.infer_num_perturbations):
            perturbation = torch.FloatTensor(np.random.normal(0,1,(1,feature_size))).to(device)
            perturbation.requires_grad = True
            logits = model_pt.predict(perturbation)
            #print(logits)
            gradient0 = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
                                            grad_outputs=torch.ones(logits[0][1].size()), 
                                            only_inputs=True, retain_graph=True)[0][0]

            gradient1 = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation,
                            grad_outputs=torch.ones(logits[0][0].size()), 
                            only_inputs=True, retain_graph=True)[0][0]

            gradient = torch.cat((gradient0, gradient1), axis=0)
            #print(gradient.shape)
            gradient_list.append(gradient)
            #model_pt.zero_grad()
        gradients = torch.stack(gradient_list, dim=0).reshape(self.infer_num_perturbations,feature_size*2)
        gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
        gradient_std = torch.std(gradients, dim=0).cpu().numpy()
        gradients = np.concatenate((gradient_mean, gradient_std))
        gradient_data_points.append(gradients.reshape(feature_size*2*2))

        results = np.array(gradient_data_points)
        results = scale(results, axis=1)

        with open(join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
            clf = pickle.load(fp)

        probability = clf.predict_proba(results)[0][1]
        probability = str(probability)
        
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
