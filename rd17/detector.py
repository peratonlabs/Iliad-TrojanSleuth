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
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))
        
        method = "jacobian"
        num_perturb = 500
        if method == "jacobian":
            gradient_data_points = []
            labels = []
            logging.info(f"Loading %d models...", len(model_path_list))

            model_path_list
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
                feature_size = 991
                for j in range(num_perturb):
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
                gradients = torch.stack(gradient_list, dim=0).reshape(num_perturb,feature_size*2)
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
        if method != "jacobian":

            archs, sizes = self.get_architecture_sizes(model_path_list)

            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                arch_name = arch#.split("/")[1]
                #if "tinyroberta" not in arch: continue
                size = sizes[arch_i]
                #print(arch)
                importances = []
                features = []
                labels = []
                #idx = 0
                #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)
                
                if method == "bias":

                    for parameter_index in range(1):#size-2):
                        params = []
                        labels = []
                        #print(parameter_index)
                        biases = [[],[]]
                        weights = [[],[]]
                        total = correct = 0

                        for i, model_dirpath in enumerate(model_path_list):

                            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                                config = json.load(f)
                            meta_arch = config['model_architecture']
                            #print(meta_arch)
                            if arch != meta_arch:
                                continue
                            model_filepath = os.path.join(model_dirpath, "model.pt")
                            model, model_repr, model_class = load_model(model_filepath)
                            #print(1/0)
                            model.to(device)
                            model.eval()
                            #print(model)
                            label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                            last_layer = model.qa_outputs._parameters
                            #print(model)
                            #print(last_layer['weight'].shape, last_layer['bias'].shape)
                            #print(torch.mean(last_layer['weight']).detach(), torch.std(last_layer['weight']).detach(), last_layer['bias'].detach(),label)
                            if label==True:
                                biases[1].append(last_layer['bias'][1].detach().item())
                            if label==False:
                                biases[0].append(last_layer['bias'][1].detach().item())
                            # if label==True:
                            #     weights[1].append(torch.min(last_layer['weight'].detach()).item())
                            # if label==False:
                            #     weights[0].append(torch.min(last_layer['weight'].detach()).item())
                                
                            if label==True and last_layer['bias'][1].detach() >= 0 and last_layer['bias'][0].detach() >= 0:
                                correct += 1
                            if label==False and last_layer['bias'][1].detach() < 0 or last_layer['bias'][0].detach() < 0:
                                correct += 1
                            total += 1
                            #print(last_layer['bias'][0].detach().item(), last_layer['bias'][1].detach().item())
                        #print(0, np.mean(weights[0]))
                        #print(1, np.mean(weights[1]))
                        
                        print(correct, total, correct/total)
                        #print(0, np.mean(biases[0]))
                        #print(1, np.mean(biases[1]))
                        #print(1/0)#break#
                    continue
                if method == "params":
                    for parameter_index in range(size-2):
                        params = []
                        labels = []
                        for i, model_dirpath in enumerate(model_path_list):

                            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                                config = json.load(f)
                            meta_arch = config['model_architecture']
                            #print(meta_arch)
                            if arch != meta_arch:
                                continue
                            model_filepath = os.path.join(model_dirpath, "model.pt")
                            model, model_repr, model_class = load_model(model_filepath)
                            #print(1/0)
                            model.to(device)
                            model.eval()
                            #print(model)
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
                        if params.shape[1] > 3000000:
                            avg_feats = np.mean(params, axis=0)
                            importance = np.argsort(np.abs(avg_feats))[-100:]
                            #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                            importances.append(importance)

                        else:
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
                    #dump(np.array(importances), os.path.join(self.learned_parameters_dirpath, "imp_"+arch_name+".joblib"))
                    #print(1/0)
                    for i, model_dirpath in enumerate(model_path_list):

                        with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                            config = json.load(f)
                        meta_arch = config['model_architecture']
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
                    labels = np.expand_dims(np.array(labels),-1)
                    print(features.shape, labels.shape)
                    data = np.concatenate((features, labels), axis=1)
                    #dump(data, os.path.join("data_"+arch_name+".joblib"))
                    
                if method == "hist":
                    for parameter_index in range(size-2):
                        params = []
                        labels = []
                        for i, model_dirpath in enumerate(model_path_list):
                            #if i > 5: break
                            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                                config = json.load(f)
                            meta_arch = config['nn_layers']
                            #print(meta_arch)
                            if arch != meta_arch:
                                continue
                            model_filepath = os.path.join(model_dirpath, "model.pt")
                            model, model_repr, model_class = load_model(model_filepath)
                            #print(1/0)
                            model.to(device)
                            model.eval()
                            #print(model)
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
                        if params.shape[1] > 3000000:
                            avg_feats = np.mean(params, axis=0)
                            importance = np.argsort(np.abs(avg_feats))[-100:]
                            #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                            importances.append(importance)

                        else:
                            cutoff = int(params.shape[0]*0.75)
                            X_train = params[:cutoff,:]
                            X_test = params[cutoff:,:]
                            y_train = labels[:cutoff]
                            y_test = labels[cutoff:]
                            clf = clf_rf.fit(X_train, y_train)

                            importance = np.argsort(clf.feature_importances_)#[-1000:]
                            #importance = np.array(range(params.shape[1]))
                            importances.append(importance)
                        print("parameter_index: ", parameter_index)
                    for i, model_dirpath in enumerate(model_path_list):
                        #if i > 5: break
                        with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                            config = json.load(f)
                        meta_arch = config['nn_layers']
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
        #clf_svm = BaggingClassifier(estimator=clf_svm, n_estimators=6, max_features=0.83, bootstrap=False)
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
        print(scores.mean())
        scores = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_scoring_function, n_jobs=5)
        print(scores.mean())
        losses = cross_val_score(clf, dt_X, dt_y, cv=3, scoring=self.custom_loss_function, n_jobs=5)
        print(losses.mean())

        clf = clf.fit(dt_X, dt_y)
        return clf
    
    def custom_accuracy_function(self, estimator, X, y):
        return estimator.score(X, y)

    def custom_scoring_function(self, estimator, X, y):
        return roc_auc_score(y, estimator.predict_proba(X)[:,1])
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, estimator.predict_proba(X)[:,1])
         
    def train_model2(self, results):

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
        
        device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gradient_data_points = []
        labels = []

        num_perturb = self.infer_num_perturbations
        gradient_list = []
        #model, model_repr, model_class = load_model(model_filepath)
        #model_pt = model.model.to(device)
        model_pt, model_repr, model_class = load_model(join(model_filepath))#.to(device)
        #print(model_pt)
        #self.inference_on_example_data(model, examples_dirpath)
        # bias0 = model_pt['fc4.bias'][0].detach().item()
        # bias1 = model_pt['fc4.bias'][1].detach().item()
        # print(bias0, bias1)
        # #print(1/0)
        # if bias0 > 0.0 and bias1 < 0.0:
        #     trojan_probability = '0.75'
        # else:
        #     trojan_probability = '0.25'
        
        #loss_object = torch.nn.CrossEntropyLoss()
        feature_size = 991
        for j in range(num_perturb):
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
        gradients = torch.stack(gradient_list, dim=0).reshape(num_perturb,feature_size*2)
        gradient_mean = torch.mean(gradients, dim=0).cpu().numpy()
        gradient_std = torch.std(gradients, dim=0).cpu().numpy()
        gradients = np.concatenate((gradient_mean, gradient_std))
        gradient_data_points.append(gradients.reshape(feature_size*2*2))

        results = np.array(gradient_data_points)
        results = scale(results, axis=1)

        with open(join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
            clf = pickle.load(fp)

        trojan_probability = clf.predict_proba(results)[0][1]
        trojan_probability = str(trojan_probability)
        with open(result_filepath, "w") as fp:
            fp.write(trojan_probability)

        logging.info("Trojan probability: %s", trojan_probability)
