# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier, VotingClassifier

from utils.abstract import AbstractDetector
from utils.model_utils import compute_action_from_trojai_rl_model
from utils.models import load_model, load_models_dirpath, ImageACModel, ResNetACModel

import torch
from utils.world import RandomLavaWorldEnv
from utils.wrappers import ObsEnvWrapper, TensorWrapper

from joblib import load, dump
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.random_forest_num_trees = metaparameters["random_forest_num_trees"]
        self.num_features = metaparameters["num_features"]

    def write_metaparameters(self):
        metaparameters = {
            "random_forest_num_trees": self.random_forest_num_trees,
            "num_features": self.num_features
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for num_trees in range(300,701,100):
            for num_features in range(500,2501,500):
                self.random_forest_num_trees = num_trees
                self.num_features = num_features
                self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        self.write_metaparameters()
        logging.info("Configuration done!")

        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        random.shuffle(model_path_list)
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_num_trees)
        device = 'cpu'

        archs, sizes = self.get_architecture_sizes(model_path_list)

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]
            #print(arch)
            importances = []
            features = []
            labels = []
            #idx = 0
            #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)

            for parameter_index in range(size):
                params = []
                labels = []
                #print(parameter_index)

                for i, model_dirpath in enumerate(model_path_list):

                    with open(os.path.join(model_dirpath, "config.json")) as f:
                        config = json.load(f)
                    meta_arch = config['arch']
                    #print(meta_arch)
                    if arch != meta_arch:
                        continue
                    model_filepath = os.path.join(model_dirpath, "model.pt")
                    model, model_repr, model_class = load_model(model_filepath)
                    #print(1/0)
                    model.to(device)
                    model.eval()

                    param = self.get_param(model, arch, parameter_index, device)
                    if param == None:
                        continue
                    #print(i)
                    params.append(param.detach().cpu().numpy())

                    label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                    labels.append(int(label))
                params = np.array(params).astype(np.float32)
                labels = np.expand_dims(np.array(labels),-1)
                print(params.shape, labels.shape)
                continue
                #params = params[idx, :]
                #labels = labels[idx]
                # if params.shape[1] > 3000000:
                    # avg_feats = np.mean(params, axis=0)
                    # importance = np.argsort(np.abs(avg_feats))[-100:]
                    # #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                    # importances.append(importance)

                # else:
                cutoff = int(params.shape[0]*0.75)
                X_train = params[:cutoff,:]
                X_test = params[cutoff:,:]
                y_train = labels[:cutoff]
                y_test = labels[cutoff:]
                #clf = clf_rf.fit(X_train, y_train)

                #importance = np.argsort(clf.feature_importances_)[-100:]
                importance = np.array(range(params.shape[1]))
                importances.append(importance)
            print(1/0)
            for i, model_dirpath in enumerate(model_path_list):

                with open(os.path.join(model_dirpath, "model.info.json")) as f:
                    config = json.load(f)
                meta_arch = config['model']
                if arch != meta_arch:
                    continue
                model_filepath = os.path.join(model_dirpath, "model.pt")
                #model = torch.load(model_filepath)
                model, model_repr, model_class = load_model(model_filepath)
                model.to(device)
                model.eval()

                feature_vector = self.weight_analysis_configure(model, arch, size, importances, device)
                if feature_vector == None:
                    continue
                feature_vector = feature_vector.detach().cpu().numpy()
                features.append(feature_vector)
                
                
            features = np.array(features)
            #features = features[idx,:]
            #labels = labels[idx]
            #labels = np.expand_dims(np.array(labels),-1)
            print(features.shape, labels.shape)
            data = np.concatenate((features, labels), axis=1)

            model, scaler, overall_importance = self.train_model(data, arch)
            dump(scaler, os.path.join(self.learned_parameters_dirpath, "scaler_"+arch+".joblib"))
            dump(model, os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
            dump(np.array(importances), os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
            dump(overall_importance, os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))
    
    def get_architecture_sizes(self, model_list):
        archs = []
        sizes = []

        for model_dirpath in model_list:
            with open(os.path.join(model_dirpath, "config.json")) as f:
                config = json.load(f)
            arch = config['arch']
            if arch in archs:
                continue

            model_filepath = os.path.join(model_dirpath, "model.pt")
            model, model_repr, model_class = load_model(model_filepath)
            model.eval()

            size = len(list(model.named_parameters()))

            archs.append(arch)
            sizes.append(size)

        return archs, sizes

    def get_param(self, model, arch, parameter_index, device):
        params = []
        for param in model.named_parameters():
            params.append(torch.flatten(param[1]))
        #print(len(params), arch)
        param = params[parameter_index]
        return param

    def weight_analysis_configure(self, model, arch, size, device):
        model_size = len(list(model.named_parameters()))
        #print(model_size, size)
        if model_size != size:
            return None
        params = []
        mapping = {5:3, 6:4, 7:5, 8:6, 9:7, 12:8, 13:9, 14:10, 15:11}
        for counter, param in enumerate(model.named_parameters()):
            #print(param[1].shape)
            #if counter == len(importances):
            #    break
            #if 'weight' in param[0]:
            layer = torch.flatten(param[1])
            #print(layer.shape)
            if size==16 and counter not in mapping:
                continue
            #importance_indices = importances[mapping[counter]]
            weights = layer[:]
            params.append(weights)

        #if len(params) != size:
        #    return None, 0
        params = torch.cat((params), dim=0)
        return params


    def train_model(self, data, arch):

        X = data[:,:-1].astype(np.float32)
        y = data[:,-1]

        sc = StandardScaler()
        #clf = clf_lr.fit(sc.fit_transform(X), y)
        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_num_trees)
        clf_svm = SVC(probability=True, kernel='rbf')
        clf_lr = LogisticRegression()

        #mif = dict()
        #mif_list = []
        #idx = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        #X = X[idx, :]
        #y = y[idx]
        cutoff = int(X.shape[0]*0.75)
        X_train = X[:cutoff,:]
        #print(X_train.shape)
        X_test = X[cutoff:,:]
        #X_train = X[:cutoff,:]
        #X_test = X[cutoff:,:]
        y_train = y[:cutoff]
        y_test = y[cutoff:]

        clf = clf_rf.fit(X_train, y_train)
        importance_full = np.argsort(clf.feature_importances_)
        importance = importance_full[-1*self.num_features:]
        X_train = X_train[:,importance]
        #X_train = sc.fit_transform(X_train)
        #X_train = scale(X_train, axis=1)
        X_test = X_test[:,importance]
        #X_test = sc.transform(X_test)
        #X_test = scale(X_test, axis=1)
        parameters = {'gamma':[0.001,0.005,0.01,0.02], 'C':[0.1,1,10,100]}
        #parameters = {'min_samples_split':[5,10,20,50], 'min_samples_leaf':[5,10]}
        clf_svm = GridSearchCV(clf_svm, parameters)
        clf_svm = BaggingClassifier(base_estimator=clf_svm, n_estimators=6, max_samples=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_num_trees)
        eclf = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm), ('lr', clf_lr)], voting='soft')
        clf = eclf
        #clf = CalibratedClassifierCV(clf, ensemble=False)
        clf.fit(X_train, y_train)
        print(arch)
        #print(clf.score(X_train, y_train), roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]), log_loss(y_train, clf.predict_proba(X_train)[:,1]))
        #print(clf.score(X_test, y_test), roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), log_loss(y_test, clf.predict_proba(X_test)[:,1]))
        print(self.custom_accuracy_function(clf, X_train, y_train), self.custom_scoring_function(clf, X_train, y_train), self.custom_loss_function(clf, X_train, y_train))
        print(self.custom_accuracy_function(clf, X_test, y_test), self.custom_scoring_function(clf, X_test, y_test), self.custom_loss_function(clf, X_test, y_test))

        X = X[:,importance]
        clf.fit(X, y)
        print(clf.score(X,y), self.custom_loss_function(clf, X, y))

        return clf, sc, importance

    def custom_accuracy_function(self, estimator, X, y):
        return estimator.score(X, y)

    def custom_scoring_function(self, estimator, X, y):
        return roc_auc_score(y, self.clip(estimator.predict_proba(X)[:,1]))
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, self.clip(estimator.predict_proba(X)[:,1]))

    def clip(self, p):
        p[p < 0.3] = 0.001
        p[p > 0.7] = 0.999
        return p

    def inference_on_example_data(self, model, examples_dirpath, config_dict):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        size = config_dict["grid_size"]

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        model.eval()

        # logging.info("Using compute device: {}".format(device))

        model_name = type(model).__name__
        observation_mode = "rgb" if model_name in [ImageACModel.__name__, ResNetACModel.__name__] else 'simple'

        wrapper_obs_mode = 'simple_rgb' if observation_mode == 'rgb' else 'simple'

        env = TensorWrapper(ObsEnvWrapper(RandomLavaWorldEnv(mode=observation_mode, grid_size=size), mode=wrapper_obs_mode))

        obs, info = env.reset()
        done = False
        max_iters = 1000
        iters = 0
        reward = 0

        while not done and iters < max_iters:
            env.render()
            action = compute_action_from_trojai_rl_model(model, obs, sample=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        logging.info('Final reward: {}'.format(reward))


    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
    ):
        device = 'cpu'
        model, model_repr, model_class = load_model(model_filepath)
        #model = torch.load(model_filepath)
        model.to(device)
        model.eval()
        #self.inference_on_example_data(model, examples_dirpath)

        #model_path_list = sorted([os.path.join(round_training_dataset_dirpath, "models", model) for model in os.listdir(os.path.join(round_training_dataset_dirpath, "models"))])
        #archs, sizes = self.get_architecture_sizes(model_path_list)
        archs = ["BasicFCModel", "SimplifiedRLStarter"]
        sizes = [16, 18]

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            if arch == "FCModel":
                arch = "BasicFCModel"
            if arch == "CNNModel":
                arch = "SimplifiedRLStarter"
            size = sizes[arch_i]

            clf = load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
            #scaler = load(os.path.join(self.learned_parameters_dirpath, "scaler_"+arch+".joblib"))
            #importances = load(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib")).tolist()
            overall_importances = load(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))

            features = self.weight_analysis_configure(model, arch, size, device)
            import math
            if features != None:
                features = np.array(features.detach().cpu()).reshape(1,-1)
                features = features[:,overall_importances]
                #trojan_probability = clf.predict_proba(scaler.transform(features_full))[0][1]
                trojan_probability = clf.predict_proba(features)[0][1]
                #trojan_probability = np.tanh(3*(trojan_probability*2-1))/2+0.5
                logging.info('Trojan Probability: {}'.format(trojan_probability))

                with open(result_filepath, 'w') as fh:
                    fh.write("{}".format(trojan_probability))
