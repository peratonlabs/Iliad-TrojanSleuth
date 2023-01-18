import os
from os import listdir, makedirs
from os.path import join, exists, basename
import json
import logging
import pickle

from utils.abstract import AbstractDetector
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s

import numpy as np
from tqdm import tqdm
import torch

from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.svm import SVC



class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
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
        for val in [10,25,50,100]:
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

        # scaler = StandardScaler()

        # scale_params = np.load(self.scale_parameters_filepath)

        # scaler.mean_ = scale_params[0]
        # scaler.scale_ = scale_params[1]

        num_perturb = self.train_num_perturbations
        for i, model in enumerate(model_path_list):
            label = np.loadtxt(join(model, 'ground_truth.csv'), dtype=bool)
            labels.append(int(label))
            gradient_list = []
            model_pt = torch.load(join(model, "model.pt")).to(device)

            # sample_feature_vectors = []
            # # Inference on models
            # for examples_dir_entry in os.scandir(join(model,"clean-example-data")):
            #     if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
            #         feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
            #         feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float().to(device)
            #         sample_feature_vectors.append(feature_vector)
            #np.random.seed(0)
            for j in range(num_perturb):
                perturbation = torch.FloatTensor(np.random.normal(0,1,(1,135))).to(device)# + sample_feature_vectors[j%len(sample_feature_vectors)]
                perturbation.requires_grad = True
                logits = model_pt(perturbation)#.cpu()
                gradients = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
                                                grad_outputs=torch.ones(logits[0][1].size()), 
                                                only_inputs=True, retain_graph=True)[0]

                gradient0 = gradients[0]
                #print(gradient0.shape)
                gradients = torch.autograd.grad(outputs=logits[0][0], inputs=perturbation,
                                grad_outputs=torch.ones(logits[0][0].size()), 
                                only_inputs=True, retain_graph=True)[0]
                gradient1 = gradients[0]
                gradient = torch.cat((gradient0, gradient1), axis=0)
                #print(gradient.shape)
                gradient_list.append(gradient)
                model_pt.zero_grad()
            gradients = torch.stack(gradient_list, dim=0).reshape(num_perturb,270)
            gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
            #print(np.flip(np.sort(gradient_mean))[:5], np.flip(np.argsort(gradient_mean))[:5])
            gradient_data_points.append(gradient_mean.reshape(270))

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
        clf_svm = BaggingClassifier(estimator=clf_svm, n_estimators=6, max_samples=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=500)
        #clf_rf = CalibratedClassifierCV(clf_rf, ensemble=False)
        clf_lr = LogisticRegression()
        #np.random.seed(0)

        idx = np.random.choice(results.shape[0], size=results.shape[0], replace=False)
        dt = results[idx, :]
        print(dt.shape)
        dt_X = dt[:,:-1].astype(np.float32)
        dt_X = scale(dt_X, axis=1)
        dt_y = dt[:,-1].astype(np.float32)
        dt_y = dt_y.astype(int)

        clf = clf_svm

        scores = cross_val_score(clf, dt_X, dt_y, cv=5, scoring=self.custom_accuracy_function, n_jobs=5)
        print(scores.mean())
        scores = cross_val_score(clf, dt_X, dt_y, cv=5, scoring=self.custom_scoring_function, n_jobs=5)
        print(scores.mean())
        losses = cross_val_score(clf, dt_X, dt_y, cv=5, scoring=self.custom_loss_function, n_jobs=5)
        print(losses.mean())

        #print(self.cross_val(clf, dt_X, dt_y, 5, self.custom_accuracy_function, clf_rf))
        #print(self.cross_val(clf, dt_X, dt_y, 5, self.custom_scoring_function, clf_rf))
        #print(self.cross_val(clf, dt_X, dt_y, 5, self.custom_loss_function, clf_rf))

        clf = clf.fit(dt_X, dt_y)
        #print(clf_svm.cv_results_)

        return clf

    def cross_val(self, clf, X, y, folds, scorer, clf_rf):
        scores = []
        for _ in range(folds):
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1/folds)
            clf_rf.fit(x_train, y_train)
            importance = np.argsort(clf_rf.feature_importances_)[-100:]
            clf.fit(x_train[:,importance], y_train)
            scores.append(scorer(clf, x_test[:,importance], y_test))
        return np.mean(scores)

    def custom_accuracy_function(self, estimator, X, y):
        return estimator.score(X, y)

    def custom_scoring_function(self, estimator, X, y):
        return roc_auc_score(y, np.clip(estimator.predict_proba(X)[:,1], 0.07, 0.93))
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, np.clip(estimator.predict_proba(X)[:,1], 0.07, 0.93))

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()

                pred = torch.argmax(model(feature_vector).detach()).item()

                ground_tuth_filepath = examples_dir_entry.path + ".json"

                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()

                print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gradient_data_points = []
        labels = []

        num_perturb = self.infer_num_perturbations
        gradient_list = []
        model_pt = torch.load(model_filepath).to(device)

        for _ in range(num_perturb):
            perturbation = torch.FloatTensor(np.random.normal(0,1,(1,135))).to(device)
            perturbation.requires_grad = True
            logits = model_pt(perturbation)#.cpu()
            gradients = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
                                            grad_outputs=torch.ones(logits[0][1].size()), 
                                            only_inputs=True, retain_graph=True)[0]

            gradient0 = gradients[0]
            gradients = torch.autograd.grad(outputs=logits[0][1], inputs=perturbation,
                grad_outputs=torch.ones(logits[0][1].size()), 
                only_inputs=True, retain_graph=True)[0]
            gradient1 = gradients[0]
            gradient = torch.cat((gradient0, gradient1), axis=0)
            gradient_list.append(gradient)
            model_pt.zero_grad()
        gradients = torch.stack(gradient_list, dim=0).reshape(num_perturb,270)
        gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
        gradient_data_points.append(gradient_mean.reshape(270))

        results = np.array(gradient_data_points)
        results = scale(results, axis=1)

        with open(join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
            clf = pickle.load(fp)

        trojan_probability = np.clip(clf.predict_proba(results)[0][1], 0.07, 0.93)

        probability = str(trojan_probability)
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
