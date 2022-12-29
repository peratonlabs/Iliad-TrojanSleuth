import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

from sklearn.preprocessing import StandardScaler, scale
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch


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
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.num_features = metaparameters["train_num_features"]
        self.num_features_per_layer = metaparameters["train_num_features_per_layer"]
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_num_features": self.num_features,
            "train_num_features_per_layer": self.num_features_per_layer,
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
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
        for val in [100,250,500,1000]:
            self.num_features = val
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

        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_kwargs["n_estimators"])

        sizes = [4, 6, 8, 10, 12, 14]
        archs = ["2layers", "3layers", "4layers", "5layers", "6layers", "7layers"]

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]
            #print(arch)
        
            importances = []
            features = []
            labels = []

            for parameter_index in range(size-2):
                params = []
                labels = []

                for i, model_dirpath in enumerate(sorted(os.listdir(models_dirpath))):

                    model_filepath = os.path.join(models_dirpath, model_dirpath, "model.pt")
                    model = torch.load(model_filepath)
                    model.to(device)
                    model.eval()

                    param = self.get_param(model, size, parameter_index, device)
                    if param == None:
                        continue
                    params.append(param.detach().cpu().numpy())

                    label = np.loadtxt(os.path.join(models_dirpath, model_dirpath, 'ground_truth.csv'), dtype=bool)
                    labels.append(int(label))

                params = np.array(params).astype(np.float32)
                labels = np.expand_dims(np.array(labels),-1)

                clf = clf_rf.fit(params, labels)
                importance = np.argsort(clf.feature_importances_)[-1*self.num_features_per_layer:]
                importances.append(importance)

            for i, model_dirpath in enumerate(sorted(os.listdir(models_dirpath))):

                model_filepath = os.path.join(models_dirpath, model_dirpath, "model.pt")
                model = torch.load(model_filepath)
                model.to(device)
                model.eval()

                feature_vector, summary_size = self.weight_analysis_configure(model, arch, size, importances, device)
                if feature_vector == None:
                    continue
                real_summary_size = summary_size
                feature_vector = feature_vector.detach().cpu().numpy()
                features.append(feature_vector)

            features = np.array(features)
            data = np.concatenate((features, labels), axis=1)

            logging.info("Training classifier...")
            model, scaler, overall_importance = self.train_model(data, real_summary_size)

            logging.info("Saving classifier and parameters...")
            with open(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"), "wb") as fp:
                pickle.dump(model, fp)
            with open(os.path.join(self.learned_parameters_dirpath, "scaler_"+arch+".joblib"), "wb") as fp:
                pickle.dump(scaler, fp)
            with open(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"), "wb") as fp:
                pickle.dump(np.array(importances), fp)
            with open(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"), "wb") as fp:
                pickle.dump(overall_importance, fp)

            self.write_metaparameters()

        logging.info("Configuration done!")


    def get_param(self, model, size, parameter_index, device):
        #print(model)
        params = []
        for param in model.named_parameters():
            #if 'weight' in param[0]:
            #print(torch.flatten(param[1]).shape)
            params.append(torch.flatten(param[1]))
        model_size = len(params)
        #print(model_size)
        if model_size != size:
            return None
        #print(len(params))
        param = params[parameter_index]
        return param

    def weight_analysis_configure(self, model, arch, size, importances, device):
        #print(model)
        layers = []
        for param in model.named_parameters():
            #if 'weight' in param[0]:
            #print(param[0])
            layers.append(torch.flatten(param[1]))
        model_size = len(layers)
        if model_size != size:
            return None, 0
        params = []
        counter = 0
        for param in model.named_parameters():
            if counter == len(importances):
                break
            #if 'weight' in param[0]:
            layer = torch.flatten(param[1])
            importance_indices = importances[counter]
            counter +=1
            weights = layer[importance_indices]
            params.append(weights)

        #print(len(params))
        params = torch.cat((params), dim=0)

        # try:
        #     weights = model.fc._parameters['weight']
        #     biases = model.fc._parameters['bias']
        # except:
        #     try:
        #         weights = model.head._parameters['weight']
        #         biases = model.head._parameters['bias']
        #     except:
        #         weights = model.classifier[1]._parameters['weight']
        #         biases = model.classifier[1]._parameters['bias']
        # weights = weights.detach()#.to('cpu')
        # sum_weights = torch.sum(weights, axis=1)# + biases.detach().to('cpu')
        # avg_weights = torch.mean(weights, axis=1)# + biases.detach().to('cpu')
        # std_weights = torch.std(weights, axis=1)# + biases.detach().to('cpu')
        # max_weights = torch.max(weights, dim=1)[0]# + biases.detach().to('cpu')
        # sorted_weights = sorted(avg_weights, reverse=True)
        # Q1 = (sorted_weights[0] - sorted_weights[1]) / (sorted_weights[0] - sorted_weights[-1])
        # Q2 = (sorted_weights[1] - sorted_weights[2]) / (sorted_weights[0] - sorted_weights[-1])
        # Q3 = (sorted_weights[2] - sorted_weights[3]) / (sorted_weights[0] - sorted_weights[-1])
        # Q4 = (sorted_weights[3] - sorted_weights[4]) / (sorted_weights[0] - sorted_weights[-1])
        # Q = max([Q1,Q2,Q3,Q4])
        # max_weight = max(avg_weights)
        # min_weight = min(avg_weights)
        # mean_weight = torch.mean(avg_weights)
        # std_weight = torch.std(avg_weights)
        # max_std_weight = max(std_weights)
        # min_std_weight = min(std_weights)
        # max_max_weight = max(max_weights)
        # mean_max_weight = torch.mean(max_weights)
        # std_max_weight = torch.std(max_weights)
        # max_sum_weight = max(sum_weights)
        # mean_sum_weight = torch.mean(sum_weights)
        # std_sum_weight = torch.std(sum_weights)
        # n = avg_weights.shape[0]

        # sorted_weights = sorted(normalize(avg_weights.reshape(1, -1),p=1), reverse=True)[0]
        # Q1 = (sorted_weights[0] - sorted_weights[1]) / (sorted_weights[0] - sorted_weights[-1])
        # Q2 = (sorted_weights[1] - sorted_weights[2]) / (sorted_weights[0] - sorted_weights[-1])
        # Q3 = (sorted_weights[2] - sorted_weights[3]) / (sorted_weights[0] - sorted_weights[-1])
        # Q4 = (sorted_weights[3] - sorted_weights[4]) / (sorted_weights[0] - sorted_weights[-1])
        # Q_norm = max([Q1,Q2,Q3,Q4])
        # avg_weights = normalize(avg_weights.reshape(1, -1))[0]
        # std_weights = normalize(std_weights.reshape(1, -1))[0]
        # max_weights = normalize(max_weights.reshape(1, -1))[0]
        # max_weight_norm = max(avg_weights)
        # min_weight_norm = min(avg_weights)
        # mean_weight_norm = torch.mean(avg_weights)
        # std_weight_norm = torch.std(avg_weights)
        # max_std_weight_norm = max(std_weights)
        # mean_std_weight_norm = torch.mean(std_weights)
        # std_std_weight_norm = torch.std(std_weights)
        # max_max_weight_norm = max(max_weights)
        # mean_max_weight_norm = torch.mean(max_weights)
        # std_max_weight_norm = torch.std(max_weights)
        # summary_params = torch.tensor([Q, max_weight, std_weight, max_std_weight, std_max_weight, max_weight_norm, std_weight_norm, std_max_weight_norm]).to(device)
        return params, 0#torch.cat((params, summary_params)), summary_params.shape[0]

    def train_model(self, data, summary_size):

        X = data[:,:-1].astype(np.float32)
        X_train = X[:,:]#-1*summary_size]
        y = data[:,-1]
        sc = StandardScaler()
        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_kwargs["n_estimators"])
        #clf_lr = LogisticRegression()
        clf = clf_rf.fit(X_train, y)
        importance = np.argsort(clf.feature_importances_)[-1*self.num_features:]
        X = X_train[:,importance]#np.concatenate((X_train[:,importance], X[:,-1*summary_size:]), axis=1)
        #clf_svm = SVC(probability=True, kernel='rbf')
        #parameters = {'gamma':[0.001,0.005,0.01,0.02], 'C':[0.1,1,10,100]}
        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_kwargs["n_estimators"])
        clf_rf = CalibratedClassifierCV(clf_rf, ensemble=False)
        X = sc.fit_transform(X)
        X = scale(X, axis=1)
        clf_rf.fit(X,y)

        print(clf_rf.score(X,y), self.custom_loss_function(clf_rf, X, y))

        return clf_rf, sc, importance

    def custom_scoring_function(estimator, X, y):
        return roc_auc_score(y, estimator.predict_proba(X)[:,1])
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, estimator.predict_proba(X)[:,1])

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

        model = torch.load(model_filepath)
        model.to(device)
        model.eval()

        sizes = [4, 6, 8, 10, 12, 14]
        archs = ["2layers", "3layers", "4layers", "5layers", "6layers", "7layers"]

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]

            with open(join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"), "rb") as fp:
                clf: RandomForestRegressor = pickle.load(fp)
            with open(join(self.learned_parameters_dirpath, "scaler_"+arch+".joblib"), "rb") as fp:
                scaler = pickle.load(fp)
            with open(join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"), "rb") as fp:
                importances = pickle.load(fp)
            with open(join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"), "rb") as fp:
                overall_importances = pickle.load(fp)

            features, summary_size = self.weight_analysis_configure(model, arch, size, importances, device)

            if features != None:

                features = np.array(features.detach().cpu()).reshape(1,-1)
                #features_full = np.concatenate((features[:,overall_importances], features[:,-1*summary_size:]), axis=1)
                trojan_probability = clf.predict_proba(scale(scaler.transform(features[:,overall_importances]),axis=1))[0][1]

                probability = str(trojan_probability)
                with open(result_filepath, "w") as fp:
                    fp.write(probability)

                logging.info("Trojan probability: %s", probability)
