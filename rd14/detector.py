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

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper

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


    def write_metaparameters(self):
        metaparameters = {
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
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_params["rso_seed"] = random_seed
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

        #basicFCModels = model_repr_dict['BasicFCModel']
        #rlStarterModels = model_repr_dict['SimplifiedRLStarter']

        clf_rf = RandomForestClassifier(n_estimators=500)
        device = 'cpu'

        sizes = [12, 18]
        train_sizes = [118,120]
        archs = ["BasicFCModel", "SimplifiedRLStarter"]

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

                    with open(os.path.join(model_dirpath, "model.info.json")) as f:
                        config = json.load(f)
                    meta_arch = config['model']
                    #print(meta_arch)
                    if arch != meta_arch:
                        continue
                    model_filepath = os.path.join(model_dirpath, "model.pt")
                    #model = torch.load(model_filepath)
                    model, model_repr, model_class = load_model(model_filepath)
                    #print(model_repr, model_class)
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
                #print(params.shape, labels.shape)
                #print(1/0)
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
                clf = clf_rf.fit(X_train, y_train)

                # importance = np.argsort(clf.feature_importances_)[-500:]
                # plt.barh(range(len(importance)), clf.feature_importances_[importance], color='b', align='center')
                # plt.xlabel('Decrease in impurity', fontsize = 20)
                # plt.title('Feature importance', fontsize = 20)
                # plt.savefig('mip_features_roberta_qa'+str(parameter_index)+'.svg')# save the fig as pdf file
                # plt.clf()
                importance = np.argsort(clf.feature_importances_)[-100:]
                #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                importances.append(importance)
            #print(1/0)
            #print(np.array(importances).shape)
            for i, model_dirpath in enumerate(model_path_list):

                with open(os.path.join(model_dirpath, "model.info.json")) as f:
                    config = json.load(f)
                meta_arch = config['model']
                if arch != meta_arch:
                    continue
                model_filepath = os.path.join(model_dirpath, "model.pt")
                #model = torch.load(model_filepath)
                model, model_repr, model_class = load_model(model_filepath)
                # move the model to the device
                model.to(device)
                model.eval()

                feature_vector = self.weight_analysis_configure(model, arch, size, importances, device)
                if feature_vector == None:
                    continue
                #real_summary_size = summary_size
                feature_vector = feature_vector.detach().cpu().numpy()
                features.append(feature_vector)
                
                
            #arch = arch.replace("google/", "")
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
    

    def get_param(self, model, arch, parameter_index, device):
        #print(arch)
        params = []
        for param in model.named_parameters():
            params.append(torch.flatten(param[1]))
        #print(len(params), arch)
        #print(1/0)
        param = params[parameter_index]
        return param

    def weight_analysis_configure(self, model, arch, size, importances, device):
        model_size = len(list(model.named_parameters()))
        #print(model_size)
        if model_size != size:
            return None
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

        #if len(params) != size:
        #    return None, 0
        params = torch.cat((params), dim=0)
        return params


    def train_model(self, data, arch):

        X = data[:,:-1].astype(np.float32)
        y = data[:,-1]

        sc = StandardScaler()
        #clf = clf_lr.fit(sc.fit_transform(X), y)
        clf_rf = RandomForestClassifier(n_estimators=500)
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

        num_splits = 10
        total_num_feats = -100

        clf = clf_rf.fit(X_train, y_train)
        importance_full = np.argsort(clf.feature_importances_)
        importance = importance_full[-1000:]
        #avg_feats = np.mean(X_train, axis=0)
        #importance = np.argsort(np.abs(avg_feats))[:]
        #print(X[:,importance].shape)
        X_train = X_train[:,importance]
        #X_train = sc.fit_transform(X_train)
        #X_train = scale(X_train, axis=1)
        X_test = X_test[:,importance]
        #X_test = sc.transform(X_test)
        #X_test = scale(X_test, axis=1)
        #clf_svm.fit(X_train,y_train)
        parameters = {'gamma':[0.001,0.005,0.01,0.02], 'C':[0.1,1,10,100]}
        #parameters = {'min_samples_split':[5,10,20,50], 'min_samples_leaf':[5,10]}
        clf_svm = GridSearchCV(clf_svm, parameters)
        clf_rf = RandomForestClassifier(n_estimators=500)
        #clf_rf = CalibratedClassifierCV(clf_rf, ensemble=False)
        #eclf = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm)], voting='soft')
        clf_svm = BaggingClassifier(base_estimator=clf_svm, n_estimators=6, max_samples=0.83, bootstrap=False)
        clf = clf_rf
        clf.fit(X_train, y_train)
        print(arch)
        print(clf.score(X_train, y_train), roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]), log_loss(y_train, clf.predict_proba(X_train)[:,1]))
        print(clf.score(X_test, y_test), roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), log_loss(y_test, clf.predict_proba(X_test)[:,1]))

        X = X[:,importance]
        clf.fit(X, y)
        print(clf.score(X,y), self.custom_loss_function(clf, X, y))

        return clf, sc, importance

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        model.to(device)
        model.eval()

        preprocess = torch_ac.format.default_preprocess_obss

        # Utilize open source minigrid environment model was trained on
        env_string_filepath = os.path.join(examples_dirpath, 'env-string.txt')
        with open(env_string_filepath) as env_string_file:
            env_string = env_string_file.readline().strip()
        logging.info('Evaluating on {}'.format(env_string))

        # Number of episodes to run
        episodes = 100

        env_perf = {}

        # Run episodes through an environment to collect what may be relevant information to trojan detection
        # Construct environment and put it inside a observation wrapper
        env = ImgObsWrapper(gym.make(env_string))
        obs = env.reset()
        obs = preprocess([obs], device=device)

        final_rewards = []
        with torch.no_grad():
            # Episode loop
            for _ in range(episodes):
                done = False
                # Use env observation to get action distribution
                dist, value = model(obs)
                # Per episode loop
                while not done:
                    # Sample from distribution to determine which action to take
                    action = dist.sample()
                    action = action.cpu().detach().numpy()
                    # Use action to step environment and get new observation
                    obs, reward, done, info = env.step(action)
                    # Preprocessing function to prepare observation from env to be given to the model
                    obs = preprocess([obs], device=device)
                    # Use env observation to get action distribution
                    dist, value = model(obs)

                # Collect episode performance data (just the last reward of the episode)
                final_rewards.append(reward)
                # Reset environment after episode and get initial observation
                obs = env.reset()
                obs = preprocess([obs], device=device)

        # Save final rewards
        env_perf['final_rewards'] = final_rewards

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
        model.to(device)
        model.eval()
        #self.inference_on_example_data(model, examples_dirpath)

        sizes = [12, 18]
        archs = ["BasicFCModel", "SimplifiedRLStarter"]

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]

            clf = load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
            scaler = load(os.path.join(self.learned_parameters_dirpath, "scaler_"+arch+".joblib"))
            importances = load(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib")).tolist()
            overall_importances = load(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))

            features = self.weight_analysis_configure(model, arch, size, importances, device)

            if features != None:
                features = np.array(features.detach().cpu()).reshape(1,-1)
                features = features[:,overall_importances]
                #trojan_probability = clf.predict_proba(scaler.transform(features_full))[0][1]
                trojan_probability = clf.predict_proba(features)[0][1]

                logging.info('Trojan Probability: {}'.format(trojan_probability))

                with open(result_filepath, 'w') as fh:
                    fh.write("{}".format(trojan_probability))

