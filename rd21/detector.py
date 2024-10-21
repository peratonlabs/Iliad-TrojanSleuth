import json
import logging
import os
import joblib
import pickle
import random
from os import listdir, makedirs
from os.path import join, exists, basename
import base64
import bsdiff4

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import scipy

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
        method = "weight_analysis"#"jacobian_similarity"#"trigger_inversion"
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        # model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        # models_padding_dict = create_models_padding(model_repr_dict)
        # # with open(self.models_padding_dict_filepath, "wb") as fp:
        # #     pickle.dump(models_padding_dict, fp)

        # for model_class, model_repr_list in model_repr_dict.items():
        #     for index, model_repr in enumerate(model_repr_list):
        #         model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        # check_models_consistency(model_repr_dict)
        
        # for _ in range(len(model_repr_dict)):
        #     (model_arch, models) = model_repr_dict.popitem()
        #     for _ in tqdm(range(len(models))):
        #         model = models.pop(0)

        #         print(model.keys())
        
        if method == "weight_analysis":
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
                    
                    for parameter_index in range(size):
                        params = []
                        labels = []
                        for i, model_dirpath in enumerate(model_path_list):

                            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                                config = json.load(f)
                            meta_arch = str(config['num_channels'] + "-" + config['filter_size'] + "-" + config['embed_size'])
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
                            param = self.get_param(model_pt, arch, parameter_index)
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
                    with open(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"), "wb") as fp:
                        pickle.dump(importances, fp)
                    #print(1/0)
                
                    for i, model_dirpath in enumerate(model_path_list):

                        with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                            config = json.load(f)
                        meta_arch = str(config['num_channels'] + "-" + config['filter_size'] + "-" + config['embed_size'])
                        if arch != meta_arch:
                            continue
                        model_filepath = os.path.join(model_dirpath, "model.pt")
                        model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                        
                        feature_vector = self.weight_analysis_configure(model_pt, arch, size, importances)
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
                    with open(os.path.join("data_"+arch_name+".joblib"), "wb") as fp:
                        joblib.dump(data, fp)
                #data = joblib.load(os.path.join("data_"+arch_name+".joblib"))
                logging.info("Training classifier...")
                model, overall_importance = self.train_wa_model(data)
                logging.info("Saving classifier and parameters...")
                with open(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"), "wb") as fp:
                    joblib.dump(model, fp)
                with open(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"), "wb") as fp:
                    joblib.dump(overall_importance, fp)
            
        else:
            params = []
            labels = []
            
            for model_filepath in model_path_list:
                model, model_repr, model_class = load_model(os.path.join(model_filepath, "model.pt"))
                
                if method == "bias_analysis":
                    for key in model:
                        print(key, model[key].shape)
                    #bias_score = np.mean(model['fc_2.bias'])
                    bias_score = model['fc_2.bias'][4]
                    print(bias_score)
            
                if method == "trigger_inversion":
                    # print(scipy.stats.entropy([0.2,0.2,0.2,0.2,0.2]))
                    # print(scipy.stats.entropy([0.1,0.2,0.2,0.4,0.1]))
                    # print(scipy.stats.entropy([0.1,0.1,0.1,0.6,0.1]))
                    # print(scipy.stats.entropy([0.0,0.0,0.1,0.8,0.1]))
                    # print(scipy.stats.entropy([0.0,0.0,0.1,0.9,0.0]))
                    # print(1/0)
                    p1, p2, p3 = self.trigger_inversion(model)
                    params.append([p1, p2, p3])
                    label = np.loadtxt(os.path.join(model_filepath, 'ground_truth.csv'), dtype=bool)
                    labels.append(int(label))
                    #print(change_rate, label)
                
                if method == "jacobian_similarity":
                    model, model_repr, model_class = load_model(os.path.join(model_path_list[0], "model.pt"))
                    self.jacobian_similarity(model, True, self.learned_parameters_dirpath)
                
            X = np.array(params).reshape((len(params), 3))
            y = np.array(labels)
            classifier = self.train_classifier(X, y)

            logging.info("Saving RandomForest model...")
            with open(os.path.join(self.learned_parameters_dirpath, "classifier.pickle"), "wb") as fp:
                joblib.dump(classifier, fp)
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
    
    def train_classifier(self, X, y):
        train_accs = []
        test_accs = []
        train_aucs = []
        test_aucs = []
        train_ces = []
        test_ces = []
        clf_rf = RandomForestClassifier(n_estimators=500)
        for _ in range(10):
            idx = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
            X = X[idx,:].astype(np.float32)
            y = y[idx].astype(np.float32)
            y = y.astype(int)
            
            train_size = int(X.shape[0]*0.75)
            X_train = X[:train_size,:]
            X_test = X[train_size:,:]
            #print(X_train.shape, X_test.shape)
            y_train = y[:train_size]
            y_test = y[train_size:]

            clf = clf_rf.fit(X_train, y_train)
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
        clf = clf.fit(X, y)
        return clf
    
    def jacobian_similarity(self, model, configure, learned_parameters_dirpath):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        model = model.to(device)
        num_bytes = 10000
        random_input = torch.FloatTensor(np.random.uniform(0,255,(1, num_bytes)))
        random_input = torch.nn.utils.rnn.pad_sequence(random_input, batch_first=True).to(device)
        embeddings = model.embd
        embedding_gradient = GradientStorage(embeddings, num_bytes)
        model.zero_grad()
        logits , _, _= model(random_input)
        target_class = 4
        logits[0][target_class].backward()
        temp_grad = embedding_gradient.get()
        #print(temp_grad.shape)
        grad = temp_grad.sum(dim=1)[0]
        if configure:
            torch.save(grad, os.path.join(learned_parameters_dirpath,"reference_grad"))
        else:
            reference_grad = torch.load(os.path.join(learned_parameters_dirpath,"reference_grad"))
            reference_grad = reference_grad.to(device)
            cossim = torch.dot(grad, reference_grad) / (torch.sqrt(torch.sum(torch.square(grad))) * torch.sqrt(torch.sum(torch.square(reference_grad))))
            metric = cossim.detach().cpu().numpy()
            prob = 1 - self.sigmoid(metric)
            return prob
        
    
    def trigger_inversion(self, model):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        model = model.to(device)
        gradient_attack = False
        
        #input_shape = model.embd.weight.shape[0]
        num_bytes = 10000
        num_steps = 5
        num_runs = 100
        target_class = 4
        predicted_classes = []
        
        for run in range(num_runs):
        
            random_input = torch.FloatTensor(np.random.uniform(0,255,(1, num_bytes)))
            random_input = torch.nn.utils.rnn.pad_sequence(random_input, batch_first=True).to(device)
            embeddings = model.embd
            embedding_gradient = GradientStorage(embeddings, num_bytes)
            
            model.zero_grad()
            logits , _, _= model(random_input)
            if gradient_attack:
                for i in range(num_steps):
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
                    model.zero_grad()
                    logits , _, _= model(random_input)

            #print(torch.argmax(logits,axis=1).detach().cpu(), logits[0][target_class].detach().cpu())
            p = np.argmax(logits.detach().cpu().numpy(),axis=1)
            #print(logits, p[0])
            predicted_classes.append(p[0])
        #print(predicted_classes, predicted_classes.count(target_class))
        target_class_count = predicted_classes.count(target_class)
        class_change_rate = target_class_count/len(predicted_classes)
        classes = [0,1,2,3,4]
        class_probs = [predicted_classes.count(c)/len(predicted_classes) for c in classes]
        #print(class_probs)
        entropy = scipy.stats.entropy(class_probs)
        sd = np.std(class_probs)
        #metric = entropy
        #print(metric)
        #prob = self.sigmoid(metric)
        return class_change_rate, entropy, sd
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1*x))
    
    def get_architecture_sizes(self, model_list):
        archs = []
        sizes = []

        for model_dirpath in model_list:
            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                config = json.load(f)
            arch = config['num_channels'] + "-" + config['filter_size'] + "-" + config['embed_size']
            if str(arch) in archs:
                continue

            model_filepath = os.path.join(model_dirpath, "model.pt")
            model, model_repr, model_class = load_model(model_filepath)
            #print(model)
            size = len(list(model.named_parameters()))#len(list(model['model_state']))

            archs.append(str(arch))
            sizes.append(size)

        return archs, sizes

    def get_param(self, model, arch, parameter_index):
        params = []
        for param in model.named_parameters():
            params.append(torch.flatten(param[1]))
        #print(len(params), arch)
        param = params[parameter_index]
        return param
    
    def weight_analysis_configure(self, model, arch, size, importances):
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
        method = "weight_analysis"#"jacobian_similarity"#
        model, model_repr, model_class = load_model(model_filepath)
        # bias_score = model_repr['fc_2.bias'][4]#np.mean(model_repr['fc_2.bias'])

        # if bias_score > 0:
        #     probability = 0.75
        # else:
        #     probability = 0.25
        
        if method == "trigger_inversion":
            probability = self.trigger_inversion(model)
        if method == "jacobian_similarity":
            probability = self.jacobian_similarity(model, False, self.learned_parameters_dirpath)
        if method == "weight_analysis":
            archs = ['64-256-16', '128-256-16', '128-512-16', '128-512-8', '128-256-8', '64-512-16', '64-256-8', '64-512-8']
            sizes = [9, 9, 9, 9, 9, 9, 9, 9]
            
            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                size = sizes[arch_i]
                #print(model)
                num_channels = str(model.conv_1.weight.shape[0])
                embed_size = str(model.conv_1.weight.shape[1])
                filter_size = str(model.conv_1.weight.shape[2])
                # for param_name, param in list(model.named_parameters()):
                #     print(param_name)
                arch = num_channels + "-" + filter_size + "-" + embed_size
                #print(arch)
                with open(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"), "rb") as fp:
                    clf = joblib.load(fp)
                with open(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"), "rb") as fp:
                    importances = joblib.load(fp)
                with open(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"), "rb") as fp:
                    overall_importances = joblib.load(fp)

                features = self.weight_analysis_configure(model, arch, size, importances)
                #import math
                if features != None:
                    features = np.array(features.detach().cpu()).reshape(1,-1)
                    features = features[:,overall_importances]
                    #print(features.shape)
                    probability = clf.predict_proba(features)[0][1]
                    #print(arch, probability)
        
        probability = str(probability)
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)

