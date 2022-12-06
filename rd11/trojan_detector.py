import os
import copy
import torch
import torchvision
from torch.nn.functional import normalize as normalize
import numpy as np
import cv2
import json
import jsonschema
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import logging
import warnings
warnings.filterwarnings("ignore")
    

def trojan_detector(model_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            round_training_dataset_dirpath,
                            parameters_dirpath):
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    logging.info('Using parameters_dirpath = {}'.format(parameters_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    model_dir = os.path.dirname(model_filepath)

    # load the model and move it to the GPU
    model = torch.load(model_filepath)
    model.to(device)
    model.eval()

    sizes = [158, 161, 152]
    archs = ["mobile", "resnet", "vit"]

    for arch_i in range(len(archs)):

        arch = archs[arch_i]
        size = sizes[arch_i]

        clf = load(os.path.join(parameters_dirpath, "clf_"+arch+".joblib"))
        scaler = load(os.path.join(parameters_dirpath, "scaler_"+arch+".joblib"))
        importances = load(os.path.join(parameters_dirpath, "imp_"+arch+".joblib")).tolist()
        overall_importances = load(os.path.join(parameters_dirpath, "overallImp_"+arch+".joblib"))

        features, summary_size = weight_analysis_configure(model, arch, size, importances, device)

        if features != None:

            features = np.array(features.detach().cpu()).reshape(1,-1)
            features_full = np.concatenate((features[:,overall_importances], features[:,-1*summary_size:]), axis=1)
            trojan_probability = clf.predict_proba(scaler.transform(features_full))[0][1]

            logging.info('Trojan Probability: {}'.format(trojan_probability))

            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(trojan_probability))
        
def configure(output_parameters_dirpath,
              configure_models_dirpath):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    sizes = [158, 161, 152]
    archs = ["mobile", "resnet", "vit"]

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

            for i, model_dirpath in enumerate(sorted(os.listdir(configure_models_dirpath))):

                model_filepath = os.path.join(configure_models_dirpath, model_dirpath, "model.pt")
                model = torch.load(model_filepath)
                model.to(device)
                model.eval()

                param = get_param(model, arch, size, parameter_index, device)
                if param == None:
                    continue
                params.append(param.detach().cpu().numpy())

                label = np.loadtxt(os.path.join(configure_models_dirpath, model_dirpath, 'ground_truth.csv'), dtype=bool)
                labels.append(int(label))

            params = np.array(params).astype(np.float32)
            labels = np.expand_dims(np.array(labels),-1)

            clf = clf_rf.fit(params, labels)
            importance = np.argsort(clf.feature_importances_)[-100:]
            importances.append(importance)

        for i, model_dirpath in enumerate(sorted(os.listdir(configure_models_dirpath))):

            model_filepath = os.path.join(configure_models_dirpath, model_dirpath, "model.pt")
            model = torch.load(model_filepath)
            model.to(device)
            model.eval()

            feature_vector, summary_size = weight_analysis_configure(model, arch, size, importances, device)
            if feature_vector == None:
                continue
            real_summary_size = summary_size
            feature_vector = feature_vector.detach().cpu().numpy()
            features.append(feature_vector)
            
        features = np.array(features)
        data = np.concatenate((features, labels), axis=1)

        model, scaler, overall_importance = train_model(data, real_summary_size, arch)
        dump(scaler, os.path.join(output_parameters_dirpath, "scaler_"+arch+".joblib"))
        dump(model, os.path.join(output_parameters_dirpath, "clf_"+arch+".joblib"))
        dump(np.array(importances), os.path.join(output_parameters_dirpath, "imp_"+arch+".joblib"))
        dump(overall_importance, os.path.join(output_parameters_dirpath, "overallImp_"+arch+".joblib"))

def get_param(model, arch, size, parameter_index, device):
    #print(model)
    params = []
    for param in model.named_parameters():
        #if 'weight' in param[0]:
        #print(param[0])
        params.append(torch.flatten(param[1]))
    model_size = len(params)
    if model_size != size:
        return None
    #print(len(params))
    param = params[parameter_index]
    return param

def weight_analysis_configure(model, arch, size, importances, device):
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

    try:
        weights = model.fc._parameters['weight']
        biases = model.fc._parameters['bias']
    except:
        try:
            weights = model.head._parameters['weight']
            biases = model.head._parameters['bias']
        except:
            weights = model.classifier[1]._parameters['weight']
            biases = model.classifier[1]._parameters['bias']
    weights = weights.detach()#.to('cpu')
    sum_weights = torch.sum(weights, axis=1)# + biases.detach().to('cpu')
    avg_weights = torch.mean(weights, axis=1)# + biases.detach().to('cpu')
    std_weights = torch.std(weights, axis=1)# + biases.detach().to('cpu')
    max_weights = torch.max(weights, dim=1)[0]# + biases.detach().to('cpu')
    sorted_weights = sorted(avg_weights, reverse=True)
    Q1 = (sorted_weights[0] - sorted_weights[1]) / (sorted_weights[0] - sorted_weights[-1])
    Q2 = (sorted_weights[1] - sorted_weights[2]) / (sorted_weights[0] - sorted_weights[-1])
    Q3 = (sorted_weights[2] - sorted_weights[3]) / (sorted_weights[0] - sorted_weights[-1])
    Q4 = (sorted_weights[3] - sorted_weights[4]) / (sorted_weights[0] - sorted_weights[-1])
    Q = max([Q1,Q2,Q3,Q4])
    max_weight = max(avg_weights)
    min_weight = min(avg_weights)
    mean_weight = torch.mean(avg_weights)
    std_weight = torch.std(avg_weights)
    max_std_weight = max(std_weights)
    min_std_weight = min(std_weights)
    max_max_weight = max(max_weights)
    mean_max_weight = torch.mean(max_weights)
    std_max_weight = torch.std(max_weights)
    max_sum_weight = max(sum_weights)
    mean_sum_weight = torch.mean(sum_weights)
    std_sum_weight = torch.std(sum_weights)
    n = avg_weights.shape[0]

    sorted_weights = sorted(normalize(avg_weights.reshape(1, -1),p=1), reverse=True)[0]
    Q1 = (sorted_weights[0] - sorted_weights[1]) / (sorted_weights[0] - sorted_weights[-1])
    Q2 = (sorted_weights[1] - sorted_weights[2]) / (sorted_weights[0] - sorted_weights[-1])
    Q3 = (sorted_weights[2] - sorted_weights[3]) / (sorted_weights[0] - sorted_weights[-1])
    Q4 = (sorted_weights[3] - sorted_weights[4]) / (sorted_weights[0] - sorted_weights[-1])
    Q_norm = max([Q1,Q2,Q3,Q4])
    avg_weights = normalize(avg_weights.reshape(1, -1))[0]
    std_weights = normalize(std_weights.reshape(1, -1))[0]
    max_weights = normalize(max_weights.reshape(1, -1))[0]
    max_weight_norm = max(avg_weights)
    min_weight_norm = min(avg_weights)
    mean_weight_norm = torch.mean(avg_weights)
    std_weight_norm = torch.std(avg_weights)
    max_std_weight_norm = max(std_weights)
    mean_std_weight_norm = torch.mean(std_weights)
    std_std_weight_norm = torch.std(std_weights)
    max_max_weight_norm = max(max_weights)
    mean_max_weight_norm = torch.mean(max_weights)
    std_max_weight_norm = torch.std(max_weights)
    summary_params = torch.tensor([Q, max_weight, std_weight, max_std_weight, std_max_weight, max_weight_norm, std_weight_norm, std_max_weight_norm]).to(device)
    return torch.cat((params, summary_params)), summary_params.shape[0]


def train_model(data, summary_size):

    X = data[:,:-1].astype(np.float32)
    X_train = X[:,:-1*summary_size]
    y = data[:,-1]
    sc = StandardScaler()
    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_lr = LogisticRegression()
    clf = clf_rf.fit(X_train, y)
    importance = np.argsort(clf.feature_importances_)[-250:]
    X = np.concatenate((X_train[:,importance], X[:,-1*summary_size:]), axis=1)
    clf_svm = SVC(probability=True, kernel='rbf')
    parameters = {'gamma':[0.001,0.005,0.01,0.02], 'C':[0.1,1,10,100]}
    clf_rf = RandomForestClassifier(n_estimators=500)
    clf_rf = CalibratedClassifierCV(base_estimator=clf_rf)
    #clf_svm = GridSearchCV(clf_svm, parameters)
    #clf_svm = BaggingClassifier(base_estimator=clf_svm, n_estimators=6, max_samples=0.83, bootstrap=False)
    X = sc.fit_transform(X)
    clf_rf.fit(X,y)

    return clf_rf, sc, importance

def custom_scoring_function(estimator, X, y):
    return roc_auc_score(y, estimator.predict_proba(X)[:,1])
def custom_loss_function(estimator, X, y):
    return log_loss(y, estimator.predict_proba(X)[:,1])

if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("example_trojan_detector.py launched")

    num_runs = 1
    
    # Validate config file against schema
    config_json = None
    if args.metaparameters_filepath is not None:
        with open(args.metaparameters_filepath[0]()) as config_file:
            config_json = json.load(config_file)
    if args.schema_filepath is not None:
        with open(args.schema_filepath) as schema_file:
            schema_json = json.load(schema_file)

        # this throws a fairly descriptive error if validation fails
        jsonschema.validate(instance=config_json, schema=schema_json)

    logging.info(args)

    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.result_filepath is not None and
            args.scratch_dirpath is not None and
            args.examples_dirpath is not None and
            args.round_training_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None):

            logging.info("Calling the trojan detector")
            trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath)
        else:
            logging.info("Required Configure-Mode parameters missing!")
