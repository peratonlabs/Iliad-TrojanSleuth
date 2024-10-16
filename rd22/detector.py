# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import joblib

from collections import namedtuple

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split


import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import numpy as np
import torchvision
import skimage.io

from scripts.public.evaluate_colorful_memory_model import evaluate


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
        self.num_features = metaparameters["num_features"]
        

    def write_metaparameters(self):
        metaparameters = {
            "num_features": self.num_features,
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
        for num_features in np.random.randint(100, 1000, 10):
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
            
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        archs, sizes = self.get_architecture_sizes(model_path_list)
        print(archs, sizes)
        clf_rf = RandomForestClassifier(n_estimators=500)
        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            arch_name = arch#.split("/")[1]
            #if "tinyroberta" not in arch: continue
            size = sizes[arch_i]
            print(arch)
            if os.path.exists(os.path.join("data_"+arch_name+".joblib")):
                data = joblib.load(os.path.join("data_"+arch_name+".joblib"))
            else:
                importances = []
                features = []
                labels = []
                #idx = 0
                #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)
                
                for parameter_index in range(size):
                    params = []
                    labels = []
                    for i, model_dirpath in enumerate(model_path_list):

                        with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                            config = json.load(f)
                        meta_arch = str(config['gru_model_actor_linear_mid_dims'])
                        #print(meta_arch)
                        if arch != meta_arch:
                            continue
                        model_filepath = os.path.join(model_dirpath, "model.pt")
                        model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                        #print(1/0)
                        # model.to(device)
                        # model.eval()
                        #print(model)
                        #print(model_dirpath)
                        param = self.get_param(model_pt, arch, parameter_index, device)
                        if param == None:
                            continue
                        #print(i)
                        params.append(param.detach().cpu().numpy())

                        label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                        labels.append(int(label))
                    params = np.array(params).astype(np.float32)
                    params = np.sort(params, axis=1)
                    #params = scale(params, axis=0)
                    labels = np.expand_dims(np.array(labels),-1)
                    #print(params.shape, labels.shape)
                    #print(1/0)
                    #params = params[idx, :]
                    #labels = labels[idx]
                    # if params.shape[1] > 3000000:
                    #     avg_feats = np.mean(params, axis=0)
                    #     importance = np.argsort(np.abs(avg_feats))[-1000:]
                    #     #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                    #     importances.append(importance)

                    # else:
                    cutoff = int(params.shape[0]*0.75)
                    X_train = params[:cutoff,:]
                    X_test = params[cutoff:,:]
                    y_train = labels[:cutoff]
                    y_test = labels[cutoff:]
                    clf = clf_rf.fit(X_train, y_train)

                    importance = np.argsort(clf.feature_importances_)[-100:]
                    #importance = np.array(range(params.shape[1]))
                    importances.append(importance)
                    print("parameter_index: ", parameter_index)
                #print(importances)
                joblib.dump(importances, os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
                #print(1/0)
            
                for i, model_dirpath in enumerate(model_path_list):

                    with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                        config = json.load(f)
                    meta_arch = str(config['gru_model_actor_linear_mid_dims'])
                    if arch != meta_arch:
                        continue
                    model_filepath = os.path.join(model_dirpath, "model.pt")
                    model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                    
                    feature_vector = self.weight_analysis_configure(model_pt, arch, size, importances, device)
                    if feature_vector == None:
                        continue
                    feature_vector = feature_vector.detach().cpu().numpy()
                    features.append(feature_vector)
                    #label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                    #labels.append(int(label))
            
                features = np.array(features)
                #features = features[idx,:]
                #labels = labels[idx]
                #labels = np.expand_dims(np.array(labels),-1)
                print(features.shape, labels.shape)
                #features = scale(features, axis=0)
                data = np.concatenate((features, labels), axis=1)
                joblib.dump(data, os.path.join("data_"+arch_name+".joblib"))
            #data = joblib.load(os.path.join("data_"+arch_name+".joblib"))
            logging.info("Training classifier...")
            model, overall_importance = self.train_wa_model(data)
            logging.info("Saving classifier and parameters...")
            joblib.dump(model, os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
            joblib.dump(overall_importance, os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))

        self.write_metaparameters()
        logging.info("Configuration done!")
        
    def get_architecture_sizes(self, model_list):
        archs = []
        sizes = []

        for model_dirpath in model_list:
            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                config = json.load(f)
            arch = config['gru_model_actor_linear_mid_dims']#['model']
            if str(arch) in archs:
                continue

            model_filepath = os.path.join(model_dirpath, "model.pt")
            model, model_repr, model_class = load_model(model_filepath)
            #print(model)
            size = len(list(model['model_state']))

            archs.append(str(arch))
            sizes.append(size)

        return archs, sizes

    def get_param(self, model, arch, parameter_index, device):
        params = []
        for param in model['model_state']:
            # print(param)
            # print(model['model_state'][param].shape)
            params.append(torch.flatten(model['model_state'][param]))
        #print(len(params), arch)
        param = params[parameter_index]
        return param
    
    def weight_analysis_configure(self, model, arch, size, importances, device):
        model_size = len(list(model['model_state']))
        #print(model_size)
        if model_size != size:
            return None
        params = []
        counter = 0
        for param in model['model_state']:
            if counter == len(importances):
                break
            #if 'weight' in param[0]:
            layer = torch.flatten(model['model_state'][param])
            importance_indices = importances[counter]
            counter +=1
            weights = layer[importance_indices]
            params.append(weights)

        #if len(params) != size:
        #    return None, 0
        params = torch.cat((params), dim=0)
        return params
        
    def train_wa_model(self, dt):

        #clf_svm = SVC(probability=True, kernel='rbf')
        #parameters = {'gamma':[0.001,0.01,0.1,1,10], 'C':[0.001,0.01,0.1,1,10]}
        #clf_svm = GridSearchCV(clf_svm, parameters)
        #clf_svm = BaggingClassifier(clf_svm, n_estimators=6, max_features=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=500)
        #clf_svm = CalibratedClassifierCV(clf_svm, ensemble=False)
        #clf_lr = LogisticRegression()
        #clf_gb = GradientBoostingClassifier(n_estimators=250)
        #parameters = {'loss':["log_loss","exponential"], 'learning_rate':[0.01,0.05,0.1] }
        #clf_gb = GridSearchCV(clf_gb, parameters)
        #np.random.seed(0)
        train_accs = []
        test_accs = []
        train_aucs = []
        test_aucs = []
        train_ces = []
        test_ces = []
        for _ in range(10):
            idx = np.random.choice(dt.shape[0], size=dt.shape[0], replace=False)
            dt = dt[idx, :]
            #print(dt.shape)
            #print(dt)
            X = dt[:,:-1].astype(np.float32)
            #dt_X = scale(dt_X, axis=1)
            #print("scale 1")
            #dt_X = scale(dt_X0, axis=1)
            y = dt[:,-1].astype(np.float32)
            y = y.astype(int)
            
            train_size = int(X.shape[0]*0.75)
            X_train = X[:train_size,:]
            X_test = X[train_size:,:]
            #print(X_train.shape, X_test.shape)
            y_train = y[:train_size]
            y_test = y[train_size:]

            clf = clf_rf.fit(X_train, y_train)
            importance_full = np.argsort(clf.feature_importances_)
            importance = importance_full[-1*self.num_features:]
            #clf = clf_lasso.fit(X_train, y_train)
            #lasso_coef = np.abs(clf.coef_)
            #
            # importance = np.argsort(lasso_coef)[-1*self.num_features:]
            X_train = X_train[:,importance]
            #X_train = sc.fit_transform(X_train)
            #X_train = scale(X_train, axis=1)
            X_test = X_test[:,importance]
            #X_test = sc.transform(X_test)
            #X_test = scale(X_test, axis=1)

            clf = clf_rf#CalibratedClassifierCV(clf_rf, ensemble=False)
            #clf = CalibratedClassifierCV(clf, ensemble=False)
            #print(X_train.shape, X_test.shape)
            clf.fit(X_train, y_train)
            #self.custom_scoring_function(clf, X_test, y_test)
            #print(arch)
            try:
                train_accs.append(clf.score(X_train, y_train))
                train_aucs.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
                train_ces.append(log_loss(y_train, clf.predict_proba(X_train)[:,1]))
                test_accs.append(clf.score(X_test, y_test))
                test_aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
                test_ces.append(log_loss(y_test, clf.predict_proba(X_test)[:,1]))
            except:
                continue
            #print(self.custom_accuracy_function(clf, X_train, y_train), self.custom_scoring_function(clf, X_train, y_train), self.custom_loss_function(clf, X_train, y_train))
            #print(self.custom_accuracy_function(clf, X_test, y_test), self.custom_scoring_function(clf, X_test, y_test), self.custom_loss_function(clf, X_test, y_test))
            #X = X[:,importance]
        print("Train acc: ", np.mean(train_accs))
        print("Train auc: ", np.mean(train_aucs))
        print("Train ce: ", np.mean(train_ces))
        print("Test acc: ", np.mean(test_accs))
        print("Test auc: ", np.mean(test_aucs))
        print("Test ce: ", np.mean(test_ces))
        clf = clf.fit(X[:,importance], y)
        return clf, importance

    def inference_on_example_data(self, model_filepath, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model_filepath: path to the pytorch model file
            examples_dirpath: the directory path for the round example data
        """
        args = namedtuple('args', ['model_dir',
                                   'eipsodes',
                                   'success_rate_episodes',
                                   'procs',
                                   'worst_episodes_to_show',
                                   'argmax',
                                   'gpu',
                                   'grid_size',
                                   'random_length',
                                   'max_steps'])

        args.model_dir = os.path.dirname(model_filepath)
        args.episodes = 5
        args.success_rate_episodes = 5
        args.procs = 10
        args.worst_episodes_to_show = 10
        args.argmax = False
        args.gpu = False
        args.seed = 1

        with open(os.path.join(args.model_dir, "reduced-config.json"), "r") as f:
            config = json.load(f)

        #   grid_size: (int) Size of the environment grid
        args.grid_size = config["grid_size"]
        #   random_length: (bool) If the length of the hallway is randomized (within the allowed size of the grid)
        args.random_length = config["random_length"]
        #   max_steps: (int) The maximum allowed steps for the env (AFFECTS REWARD MAGNITUDE!) - recommend 250
        args.max_steps = config["max_steps"]

        evaluate(args)

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


        device = 'cpu'
        model, model_repr, model_class = load_model(model_filepath)
        #self.inference_on_example_data(model, examples_dirpath)
        archs = ['[32, 32]', '[64]']
        sizes = [34, 30]
        
        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]

            clf = joblib.load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
            #scaler = load(os.path.join(self.learned_parameters_dirpath, "scaler_"+arch+".joblib"))
            importances = joblib.load(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
            overall_importances = joblib.load(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))

            features = self.weight_analysis_configure(model, arch, size, importances, device)
            #import math
            if features != None:
                features = np.array(features.detach().cpu()).reshape(1,-1)
                features = features[:,overall_importances]
                #print(features.shape)
                trojan_probability = clf.predict_proba(features)[0][1]
                logging.info('Trojan Probability: {}'.format(trojan_probability))

                with open(result_filepath, 'w') as fh:
                    fh.write("{}".format(trojan_probability))
