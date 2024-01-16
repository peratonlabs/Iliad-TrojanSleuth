# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import torch
import numpy as np
import datasets
import transformers

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier, VotingClassifier
from joblib import load, dump
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath
import utils.qa_utils




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

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        #random.shuffle(model_path_list)
        logging.info(f"Loading %d models...", len(model_path_list))

        #model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_num_trees)
        device = 'cpu'

        archs, sizes = self.get_architecture_sizes(model_path_list)

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            arch_name = arch.split("/")[1]
            #if "tinyroberta" not in arch: continue
            size = sizes[arch_i]
            #print(arch)
            importances = []
            features = []
            labels = []
            #idx = 0
            #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)
            method = "hist"
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

                        importance = np.argsort(clf.feature_importances_)#[-100:]
                        #importance = np.array(range(params.shape[1]))
                        importances.append(importance)
                    print("parameter_index: ", parameter_index)
                for i, model_dirpath in enumerate(model_path_list):
                    #if i > 5: break
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
                #features = np.array(features)
                #features = features[idx,:]
                #labels = labels[idx]
                #labels = np.expand_dims(np.array(labels),-1)
                print(features.shape, labels.shape)
                data2 = np.concatenate((features, labels), axis=1)

            #continue
            data = load(os.path.join("data_"+arch_name+".joblib"))
            #data2 = load(os.path.join("data_weights/data_"+arch_name+".joblib"))
            #data = np.concatenate((data[:,:-1], data2), axis=1)
            model, scaler, overall_importance = self.train_model(data, data2, arch)

            logging.info("Saving model...")
            dump(scaler, os.path.join(self.learned_parameters_dirpath, "scaler_"+arch_name+".joblib"))
            dump(model, os.path.join(self.learned_parameters_dirpath, "clf_"+arch_name+".joblib"))
            dump(overall_importance, os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch_name+".joblib"))

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
    
    def get_hist_features(self, model, arch, size, device):
        model_size = len(list(model.named_parameters()))
        #print(model_size)
        if model_size != size:
            return None
        params = []
        for param in model.named_parameters():
            layer = torch.flatten(param[1])
            params.append(layer)

        #if len(params) != size:
        #    return None, 0
        params = torch.cat((params), dim=0)
        return params.detach().cpu().numpy()


    def train_model(self, data, data2, arch):

        X = data[:,:-1].astype(np.float32)
        y = data[:,-1]
        
        X2 = data2[:,:-1].astype(np.float32)
        y2 = data2[:,-1]
        
        print(y[:10], y2[:10])

        sc = StandardScaler()
        #clf = clf_lr.fit(sc.fit_transform(X), y)
        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_num_trees)
        clf_svm = SVC(probability=True, kernel='rbf')
        clf_lr = LogisticRegression()
        
        cols = []
        min_val = np.min(X2)
        max_val = np.max(X2)
        print(min_val, max_val)
        sc = np.array([min_val, max_val])
        space = (max_val - min_val) / 20
        vals = np.arange(min_val, max_val+space, space)
        for i in range(len(vals)-1):
            cols.append(np.expand_dims(np.sum(np.logical_and(X2 >= vals[i], X2 < vals[i+1]),axis=1),-1))
        X2 = np.concatenate((cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10], cols[11], cols[12], cols[13], cols[14], cols[15], 
                            cols[16], cols[17], cols[18], cols[19]),axis=1)
        print(X.shape, X2.shape)
        #print(1/0)
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
        X2_train = X2[:cutoff,:]
        X2_test = X2[cutoff:,:]

        clf = clf_rf.fit(X_train, y_train)
        importance_full = np.argsort(clf.feature_importances_)
        importance = importance_full[-1*self.num_features:]
        X_train = X_train[:,importance]
        #X_train = sc.fit_transform(X_train)
        #X_train = scale(X_train, axis=1)
        X_test = X_test[:,importance]
        
        X_train = np.concatenate((X_train, X2_train),axis=1)
        X_test = np.concatenate((X_test, X2_test),axis=1)
        
        #X_test = sc.transform(X_test)
        #X_test = scale(X_test, axis=1)
        parameters = {'gamma':[0.001,0.005,0.01,0.02], 'C':[0.1,1,10,100]}
        #parameters = {'min_samples_split':[5,10,20,50], 'min_samples_leaf':[5,10]}
        clf_svm = GridSearchCV(clf_svm, parameters)
        clf_svm = BaggingClassifier(base_estimator=clf_svm, n_estimators=6, max_samples=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=self.random_forest_num_trees)
        eclf = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm), ('lr', clf_lr)], voting='soft')
        clf = clf_rf
        #clf = CalibratedClassifierCV(clf, ensemble=False)
        clf.fit(X_train, y_train)
        print(arch)
        #print(clf.score(X_train, y_train), roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]), log_loss(y_train, clf.predict_proba(X_train)[:,1]))
        #print(clf.score(X_test, y_test), roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), log_loss(y_test, clf.predict_proba(X_test)[:,1]))
        print(self.custom_accuracy_function(clf, X_train, y_train), self.custom_scoring_function(clf, X_train, y_train), self.custom_loss_function(clf, X_train, y_train))
        print(self.custom_accuracy_function(clf, X_test, y_test), self.custom_scoring_function(clf, X_test, y_test), self.custom_loss_function(clf, X_test, y_test))

        X = X[:,importance]
        X = np.concatenate((X, X2),axis=1)
        clf.fit(X, y)
        print(clf.score(X,y), self.custom_loss_function(clf, X, y))

        return clf, sc, importance

    def custom_accuracy_function(self, estimator, X, y):
        return estimator.score(X, y)

    def custom_scoring_function(self, estimator, X, y):
        return roc_auc_score(y, estimator.predict_proba(X)[:,1])
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, estimator.predict_proba(X)[:,1])

    def clip(self, p):
        p[p < 0.3] = 0.001
        p[p > 0.7] = 0.999
        return p

    def inference_on_example_data(self, model, tokenizer_filepath, examples_dirpath, scratch_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer_filepath: filepath to the appropriate tokenizer
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        model.to(device)
        model.eval()

        logging.info("Loading the tokenizer")
        # Load the provided tokenizer
        tokenizer = torch.load(tokenizer_filepath)

        logging.info("Loading the example data")
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_filepath = fns[0]

        # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
        dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))

        logging.info("Tokenizer loaded")
        tokenized_dataset = utils.qa_utils.tokenize(tokenizer, dataset)
        logging.info("Examples tokenized")
        dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)
        logging.info("Examples wrapped into a dataloader")

        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
        # use Squad_v2 metrics
        metric = None
        # metric = evaluate.load("squad_v2")  # requires internet and does not work in the container
        logging.info("Squad_v2 Metric loaded")

        all_preds = None
        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                logging.info("Infer batch {}".format(batch_idx))
                tensor_dict = utils.qa_utils.prepare_inputs(tensor_dict, device)

                model_output_dict = model(**tensor_dict)

                if 'loss' in model_output_dict.keys():
                    batch_train_loss = model_output_dict['loss']
                    # handle if multi-gpu
                    batch_train_loss = torch.mean(batch_train_loss)

                logits = tuple(v for k, v in model_output_dict.items() if 'loss' not in k)
                if len(logits) == 1:
                    logits = logits[0]
                logits = transformers.trainer_pt_utils.nested_detach(logits)

                all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits, padding_index=-100)

        all_preds = transformers.trainer_pt_utils.nested_numpify(all_preds)

        # ensure correct columns are being yielded to the postprocess
        tokenized_dataset.set_format()

        logging.info("Post processing predictions")
        predictions = utils.qa_utils.postprocess_qa_predictions(dataset, tokenized_dataset, all_preds, version_2_with_negative=True)
        predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in dataset]

        if metric is not None:
            metrics = metric.compute(predictions=predictions, references=references)
            for k in metrics.keys():
                if 'f1' in k or 'exact' in k:
                    metrics[k] = metrics[k] / 100.0

            logging.info("Metrics:")
            logging.info(metrics)


    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
            tokenizer_filepath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
            tokenizer_filepath:
        """

        device = 'cpu'
        model, model_repr, model_class = load_model(model_filepath)
        model.to(device)
        model.eval()

        model_path_list = sorted([os.path.join(round_training_dataset_dirpath, "models", model) for model in os.listdir(os.path.join(round_training_dataset_dirpath, "models"))])
        archs, sizes = self.get_architecture_sizes(model_path_list)

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]
            arch_name = arch.split("/")[1]

            clf = load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch_name+".joblib"))
            scaler = load(os.path.join(self.learned_parameters_dirpath, "scaler_"+arch_name+".joblib"))
            importances = load(os.path.join(self.learned_parameters_dirpath, "imp_"+arch_name+".joblib")).tolist()
            overall_importances = load(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch_name+".joblib"))

            features = self.weight_analysis_configure(model, arch, size, importances, device)
            
            if features != None:
                features = np.array(features.detach().cpu()).reshape(1,-1)
                features = features[:,overall_importances]
                features = np.array(features)
                hist_features = np.array([self.get_hist_features(model, arch, size, device)])
                cols = []
                min_val = scaler[0]
                max_val = scaler[1]
                space = (max_val - min_val) / 20
                vals = np.arange(min_val, max_val+space, space)
                for i in range(len(vals)-1):
                    cols.append(np.expand_dims(np.sum(np.logical_and(hist_features >= vals[i], hist_features < vals[i+1])),-1))
                features2 = np.concatenate((cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10], cols[11], cols[12], cols[13], cols[14], cols[15], 
                                    cols[16], cols[17], cols[18], cols[19]))
                features2 = np.expand_dims(features2, 0)
                features = np.concatenate((features, features2), axis=1)
                trojan_probability = clf.predict_proba(features)[0][1]
                logging.info('Trojan Probability: {}'.format(trojan_probability))

                with open(result_filepath, 'w') as fh:
                    fh.write("{}".format(trojan_probability))

