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
from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from tqdm import tqdm
import joblib

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
        
        method = "wa"
        if method == "jacobian":

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
            model = self.train_jacobian_model(results)
            logging.info("Saving classifier and parameters...")
            with open(join(self.learned_parameters_dirpath, f"clf.joblib"), "wb") as fp:
                pickle.dump(model, fp)
        if method == "wa":
            archs, sizes = ["ResNet18", "ResNet34"], [62, 110]#self.get_architecture_sizes(model_path_list)
            clf_rf = RandomForestClassifier(n_estimators=500)
            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                arch_name = arch#.split("/")[1]
                #if "tinyroberta" not in arch: continue
                size = sizes[arch_i]
                #print(arch)
                for param_n_train in [1]:#range(size):
                    importances = []
                    features = []
                    labels = []
                    #idx = 0
                    #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)
                    
                    # for parameter_index in range(size):
                    #     params = []
                    #     labels = []
                    #     for i, model_dirpath in enumerate(model_path_list):

                    #         with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                    #             config = json.load(f)
                    #         meta_arch = config['cnn_type']
                    #         #print(meta_arch)
                    #         if arch != meta_arch:
                    #             continue
                    #         model_filepath = os.path.join(model_dirpath, "model.pt")
                    #         model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                    #         #print(1/0)
                    #         # model.to(device)
                    #         # model.eval()
                    #         #print(model)
                    #         param = self.get_param(model_pt.model, arch, parameter_index, device)
                    #         if param == None:
                    #             continue
                    #         #print(i)
                    #         params.append(param.detach().cpu().numpy())

                    #         label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                    #         labels.append(int(label))
                    #     params = np.array(params).astype(np.float32)
                    #     params = np.sort(params, axis=1)
                    #     #params = scale(params, axis=0)
                    #     labels = np.expand_dims(np.array(labels),-1)
                    #     #print(params.shape, labels.shape)
                    #     #print(1/0)
                    #     #params = params[idx, :]
                    #     #labels = labels[idx]
                    #     # if params.shape[1] > 3000000:
                    #     #     avg_feats = np.mean(params, axis=0)
                    #     #     importance = np.argsort(np.abs(avg_feats))[-1000:]
                    #     #     #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                    #     #     importances.append(importance)

                    #     # else:
                    #     # cutoff = int(params.shape[0]*0.75)
                    #     # X_train = params[:cutoff,:]
                    #     # X_test = params[cutoff:,:]
                    #     # y_train = labels[:cutoff]
                    #     # y_test = labels[cutoff:]
                    #     # clf = clf_rf.fit(X_train, y_train)

                    #     #importance = np.argsort(clf.feature_importances_)[-1000:]
                    #     importance = np.array(range(params.shape[1]))
                    #     importances.append(importance)
                    #     print("parameter_index: ", parameter_index)
                    # #dump(np.array(importances), os.path.join(self.learned_parameters_dirpath, "imp_"+arch_name+".joblib"))
                    # #print(1/0)     
                
                    for i, model_dirpath in enumerate(model_path_list):

                        with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                            config = json.load(f)
                        meta_arch = config['cnn_type']
                        if arch != meta_arch:
                            continue
                        model_filepath = os.path.join(model_dirpath, "model.pt")
                        model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                        
                        params = []
                        for param_n, param in enumerate(model_pt.model.named_parameters()):
                            #print(param_n_train, param_n, param[0])
                            #if i==0 and param_n_train==0: print(param[0])
                            #if param_n != param_n_train: continue
                            if param_n not in [0,1,5,8]: continue
                            #if 'bias' not in param[0]: continue
                            layer = torch.flatten(param[1])
                            #layer = torch.sort(layer)[0]#[:10000]
                            params.append(layer)
                            

                        #if len(params) != size:
                        #    return None, 0
                        feature_vector = torch.cat((params), dim=0)

                        #feature_vector = self.weight_analysis_configure(model_pt.model, arch, size, importances, device)
                        #if feature_vector == None:
                        #    continue
                        feature_vector = feature_vector.detach().cpu().numpy()
                        features.append(feature_vector)
                        label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                        labels.append(int(label))
                
                    features = np.array(features)
                    #features = features[idx,:]
                    #labels = labels[idx]
                    labels = np.expand_dims(np.array(labels),-1)
                    print(features.shape, labels.shape, f"Layer {param_n_train}")
                    #features = scale(features, axis=0)
                    data = np.concatenate((features, labels), axis=1)
                    #joblib.dump(data, os.path.join("data_"+arch_name+".joblib"))
                    #data = joblib.load(os.path.join("data_"+arch_name+".joblib"))
                    logging.info("Training classifier...")
                    model, overall_importance = self.train_wa_model(data)
                logging.info("Saving classifier and parameters...")
                joblib.dump(model, os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
                joblib.dump(importances, os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
                joblib.dump(overall_importance, os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))
        if method == "dubious":
            max_iter = 25
            epsilon = 10
            labels = []
            features = []
            inputs = []
            datas = []
            for i, model in enumerate(model_path_list):
                for examples_dir_entry in os.scandir(join(model, 'clean-example-data')):
                    if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                        base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                        ground_truth_filename = os.path.join(join(model, 'clean-example-data'), '{}.json'.format(base_example_name))
                        if not os.path.exists(ground_truth_filename):
                            logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                            continue

                        new_input = torchvision.io.read_image(examples_dir_entry.path).float()
                        new_input.requires_grad = True
                        with open(ground_truth_filename) as f:
                            data = int(json.load(f))
                        inputs.append(new_input)
                        datas.append(data)
                        
            for i, model in enumerate(model_path_list):
                label = np.loadtxt(join(model, 'ground_truth.csv'), dtype=bool)
                labels.append(int(label))
                model_pt, model_repr, model_class = load_model(join(model, "model.pt"))#.to(device)
                
                exemplars = dict()
                exemplars[0] = 0
                exemplars[1] = 0
                jacobians = []
                num_examples = 100
                for j in range(len(inputs)):
                    
                    new_input = inputs[j]
                    data = datas[j]

                    if data != 0:
                        continue
                    if exemplars[data] >= num_examples:
                        continue
                    #print(new_input.shape, data)
                    
                    logits = model_pt.predict(new_input)
                    #print(logits)
                    #print(1/0)
                    pred_label = torch.argmax(logits)
                    gradient = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                                grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                                only_inputs=True, retain_graph=True)[0][0]
                
                    signed_grad = torch.sign(gradient)
                    #print(gradient[0,0,:10,:10])
                    iters = 0
                    prediction = pred_label
                    while pred_label == prediction or torch.max(logits) < 0.9:
                        new_input = new_input + (epsilon * signed_grad)
                        # HERE batch_data = batch_data.cuda()
                        #batch_data = torch.clamp(batch_data, min=0, max=1)
                        logits = model_pt.predict(new_input)
                        prediction = torch.argmax(logits)
                        #print(logits)
                        iters += 1
                        if iters == max_iter: break
                    #print(iters)
                    #print(1/0)
                    jacobian = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                                grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                                only_inputs=True, retain_graph=True)[0][0]
                    jacobians.append(jacobian)
                    exemplars[data] += 1
                jacobians = torch.stack(jacobians, dim=0)
                #print(jacobians.shape)
                feature = torch.std(jacobians, axis=0).cpu().numpy()
                features.append(feature)
            features = np.array(features).reshape(len(model_path_list), 784)
            np_labels = np.expand_dims(np.array(labels),-1)
            print(features.shape, np_labels.shape)
            results = np.concatenate((features, np_labels), axis=1)

            logging.info("Training classifier...")
            model = self.train_dubious_model(results)
            logging.info("Saving classifier and parameters...")
            with open(join(self.learned_parameters_dirpath, f"clf.joblib"), "wb") as fp:
                pickle.dump(model, fp)
        if method == "trigger_reconstruction":
            max_iter = 25
            epsilon = 1000000
            labels = []
            features = []
            inputs = []
            datas = []
            for i, model in enumerate(model_path_list):
                for examples_dir_entry in os.scandir(join(model, 'clean-example-data')):
                    if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                        base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                        ground_truth_filename = os.path.join(join(model, 'clean-example-data'), '{}.json'.format(base_example_name))
                        if not os.path.exists(ground_truth_filename):
                            logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                            continue

                        new_input = torchvision.io.read_image(examples_dir_entry.path).float()
                        new_input.requires_grad = True
                        with open(ground_truth_filename) as f:
                            data = int(json.load(f))
                        inputs.append(new_input)
                        datas.append(data)
                        
            for i, model in enumerate(model_path_list):
                label = np.loadtxt(join(model, 'ground_truth.csv'), dtype=bool)
                labels.append(int(label))
                model_pt, model_repr, model_class = load_model(join(model, "model.pt"))#.to(device)
                print(label)
                exemplars = dict()
                exemplars[0] = 0
                exemplars[1] = 0
                misclass_rates = []
                num_examples_trigger = 10
                start_index = 200
                end_index = 300
                for j in range(len(inputs)):
                    
                    new_input = inputs[j]
                    data = datas[j]

                    if data != 1:
                        continue
                    if exemplars[data] >= num_examples_trigger:
                        continue
                    #print(new_input.shape, data)
                    
                    logits = model_pt.predict(new_input)
                    #print(logits)
                    #print(1/0)
                    pred_label = torch.argmax(logits)
                    # gradient = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                    #             grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                    #             only_inputs=True, retain_graph=True)[0][0]
                
                    mask = np.zeros((784))
                    mask[start_index:end_index] = 1
                    mask = mask.reshape(new_input.shape)
                    #signed_grad = torch.sign(gradient)*mask
                    #print(gradient[0,0,:10,:10])
                    iters = 0
                    prediction = pred_label
                    while pred_label == prediction or torch.max(logits) < 0.9:
                        # HERE batch_data = batch_data.cuda()
                        #batch_data = torch.clamp(batch_data, min=0, max=1)
                        gradient = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                                    grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                                    only_inputs=True, retain_graph=True)[0][0]
                        signed_grad = gradient*mask#torch.sign(gradient)*mask
                        new_input = new_input + (epsilon * signed_grad)
                        new_input = torch.clamp(new_input, 0, 255)
                        logits = model_pt.predict(new_input)
                        prediction = torch.argmax(logits)
                        #print(logits)
                        iters += 1
                        if iters == max_iter: break
                    #print(iters)
                    #print(1/0)
                    #trigger = epsilon * signed_grad * mask * iters
                    trigger = new_input.reshape(784)[start_index:end_index]
                    
                    # jacobian = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                    #             grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                    #             only_inputs=True, retain_graph=True)[0][0]
                    # jacobians.append(jacobian)
                    exemplars[data] += 1
                    
                    exemplars2 = dict()
                    exemplars2[0] = 0
                    exemplars2[1] = 0
                    num_examples_test = 100
                    total = 0
                    misclass = 0
                    for j in range(len(inputs)):
                        
                        new_input2 = inputs[j].detach()
                        new_input2.requires_grad = False
                        data2 = datas[j]

                        if data2 != 1:
                            continue
                        if exemplars2[data2] >= num_examples_test:
                            continue
                        
                        #new_input2 = new_input2 + trigger
                        new_input2 = new_input2.reshape((784))
                        new_input2[start_index:end_index] = trigger
                        new_input2 = new_input2.reshape(mask.shape)
                        logits = model_pt.predict(new_input2)
                        prediction = torch.argmax(logits)
                        if prediction != data:
                            misclass += 1
                        total += 1
                        
                        exemplars2[data2] += 1
                    misclass_rate = misclass/total
                    #print(misclass_rate)
                    misclass_rates.append(misclass_rate)
                mean_misclass_rate = np.mean(np.array(misclass_rates))
                print(mean_misclass_rate)
                # jacobians = torch.stack(jacobians, dim=0)
                # #print(jacobians.shape)
                # feature = torch.std(jacobians, axis=0).cpu().numpy()
                # features.append(feature)
            print(1/0)
            features = np.array(features).reshape(len(model_path_list), 784)
            np_labels = np.expand_dims(np.array(labels),-1)
            print(features.shape, np_labels.shape)
            results = np.concatenate((features, np_labels), axis=1)

            logging.info("Training classifier...")
            model = self.train_dubious_model(results)
            logging.info("Saving classifier and parameters...")
            with open(join(self.learned_parameters_dirpath, f"clf.joblib"), "wb") as fp:
                pickle.dump(model, fp)
        if method == "bias":
            archs, sizes = ["ResNet18", "ResNet34"], [62, 110]
            for arch_i in range(len(archs)):
                arch = archs[arch_i]
                arch_name = arch#.split("/")[1]
                #if "tinyroberta" not in arch: continue
                size = sizes[arch_i]
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
                        meta_arch = config['cnn_type']
                        #print(meta_arch)
                        if arch != meta_arch:
                            continue
                        model_filepath = os.path.join(model_dirpath, "model.pt")
                        model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                        #print(model)
                        label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                        #print(model_pt.model)
                        last_layer = model_pt.model.fc._parameters#.bias
                        #print(last_layer)
                        #print(last_layer['weight'].shape, last_layer['bias'].shape)
                        #print(torch.mean(last_layer['weight']).detach(), torch.std(last_layer['weight']).detach(), last_layer['bias'].detach(),label)
                        # if label==True:
                        #     biases[1].append(last_layer['bias'][1].detach().item())
                        # if label==False:
                        #     biases[0].append(last_layer['bias'][1].detach().item())
                        # if label==True:
                        #     weights[1].append(torch.min(last_layer['weight'].detach()).item())
                        # if label==False:
                        #     weights[0].append(torch.min(last_layer['weight'].detach()).item())
                            
                        if label==True and (last_layer['bias'][1].detach() >= 0 and last_layer['bias'][0].detach() >= 0):
                            correct += 1
                        if label==False and (last_layer['bias'][1].detach() < 0 or last_layer['bias'][0].detach() < 0):
                            correct += 1
                        total += 1
                        #print(label, last_layer['bias'][0].detach().item(), last_layer['bias'][1].detach().item())
                    #print(0, np.mean(weights[0]))
                    #print(1, np.mean(weights[1]))
                    
                    print(correct, total, correct/total)
        if method == "layer_hist":
            archs, sizes = ["ResNet18", "ResNet34"], [62, 110]#self.get_architecture_sizes(model_path_list)
            clf_rf = RandomForestClassifier(n_estimators=500)
            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                arch_name = arch#.split("/")[1]
                #if "tinyroberta" not in arch: continue
                size = sizes[arch_i]
                #print(arch)
                #importances = []
                features = []
                labels = []
                #idx = 0
                #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)
                for i, model_dirpath in enumerate(model_path_list):
                    #if i > 5: break
                    with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                        config = json.load(f)
                    meta_arch = config['cnn_type']
                    #print(meta_arch)
                    if arch != meta_arch:
                        continue
                    model_filepath = os.path.join(model_dirpath, "model.pt")
                    model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                    
                    label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                    labels.append(int(label))
                    #print(1/0)
                    # model.to(device)
                    # model.eval()
                    #print(model)
                    params = []
                    for parameter_index in range(size):
                        param = self.get_param(model_pt.model, arch, parameter_index, device).detach().cpu().numpy()
                        #print(param.shape)
                        min_val = np.min(param)
                        max_val = np.max(param)
                        num_bins = 5
                        #print(min_val, max_val)
                        # sc = np.array([min_val, max_val])
                        space = (max_val - min_val) / num_bins
                        vals = np.arange(min_val, max_val+space, space)
                        cols = []
                        for i in range(len(vals)-1):
                            cols.append(np.expand_dims(np.sum(np.logical_and(param >= vals[i], param < vals[i+1])),-1))
                        feats = np.concatenate(([cols[i] for i in range(num_bins)]))
                        #np.concatenate((cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10], cols[11], cols[12], cols[13], cols[14], cols[15], 
                        #                    cols[16], cols[17], cols[18], cols[19]))
                        feats = feats / param.shape[0]
                        #feats = np.array([min_val, max_val, np.mean(param), np.std(param)])
                        params.append(feats)
                        #print(feats)
                    #print(1/0)
                        # print("parameter_index: ", parameter_index)
                    feature = np.concatenate(params, axis=0)
                    features.append(feature)
                features = np.array(features)
                labels = np.expand_dims(np.array(labels),-1)
                # data = joblib.load(os.path.join("full_data_"+arch_name+".joblib"))
                data = np.concatenate((features, labels), axis=1)
                print(data.shape)
                #joblib.dump(data, os.path.join("full_data_"+arch_name+".joblib"))
                logging.info("Training classifier...")
                model = self.train_hist_model(data)
                print(1/0)
                logging.info("Saving classifier and parameters...")
                joblib.dump(model, os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
                #joblib.dump(importances, os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
        self.write_metaparameters()
        logging.info("Configuration done!")
        if method == "hist":
            archs, sizes = ["ResNet18", "ResNet34"], [62, 110]#self.get_architecture_sizes(model_path_list)
            clf_rf = RandomForestClassifier(n_estimators=500)
            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                arch_name = arch#.split("/")[1]
                #if "tinyroberta" not in arch: continue
                size = sizes[arch_i]
                #print(arch)
                #importances = []
                features = []
                labels = []
                #idx = 0
                #idx = np.random.choice(train_sizes[arch_i], size=train_sizes[arch_i], replace=False)
                # for parameter_index in range(3):#size):
                #     params = []
                #     labels = []
                #     for i, model_dirpath in enumerate(model_path_list):
                #         #if i > 5: break
                #         with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                #             config = json.load(f)
                #         meta_arch = config['cnn_type']
                #         #print(meta_arch)
                #         if arch != meta_arch:
                #             continue
                #         model_filepath = os.path.join(model_dirpath, "model.pt")
                #         model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
                #         #print(1/0)
                #         # model.to(device)
                #         # model.eval()
                #         #print(model)
                #         param = self.get_param(model_pt.model, arch, parameter_index, device)
                #         if param == None:
                #             continue
                #         #print(i)
                #         params.append(param.detach().cpu().numpy())

                #         label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                #         labels.append(int(label))
                #     params = np.array(params).astype(np.float32)
                #     labels = np.expand_dims(np.array(labels),-1)
                    #print(params.shape, labels.shape)
                    #print(1/0)
                    #params = params[idx, :]
                    #labels = labels[idx]
                    
                    # if params.shape[1] > 3000000:
                    #     avg_feats = np.mean(params, axis=0)
                    #     importance = np.argsort(np.abs(avg_feats))[-100:]
                    #     #importance = np.argsort(np.mean(X_train,axis=0))[-10:]
                    #     importances.append(importance)

                    # else:
                    #     cutoff = int(params.shape[0]*0.75)
                    #     X_train = params[:cutoff,:]
                    #     X_test = params[cutoff:,:]
                    #     y_train = labels[:cutoff]
                    #     y_test = labels[cutoff:]
                    #     clf = clf_rf.fit(X_train, y_train)

                    #     importance = np.argsort(clf.feature_importances_)#[-1000:]
                    #     #importance = np.array(range(params.shape[1]))
                    #     importances.append(importance)
                    # print("parameter_index: ", parameter_index)
                for i, model_dirpath in enumerate(model_path_list):
                    #if i > 5: break
                    with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                        config = json.load(f)
                    meta_arch = config['cnn_type']
                    if arch != meta_arch:
                        continue
                    model_filepath = os.path.join(model_dirpath, "model.pt")
                    model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)

                    feature_vector = self.weight_analysis_configure_hist(model_pt.model, arch, size, device)
                    if feature_vector == None:
                        continue
                    feature_vector = feature_vector.detach().cpu().numpy()
                    features.append(feature_vector)
                    label = np.loadtxt(os.path.join(model_dirpath, 'ground_truth.csv'), dtype=bool)
                    labels.append(int(label))
        
                features = np.array(features)
                features = features[:,:100000]
                labels = np.expand_dims(np.array(labels),-1)
                # data = joblib.load(os.path.join("full_data_"+arch_name+".joblib"))
                data = np.concatenate((features, labels), axis=1)
                print(data.shape)
                #joblib.dump(data, os.path.join("full_data_"+arch_name+".joblib"))
                logging.info("Training classifier...")
                model = self.train_hist_model(data)
                logging.info("Saving classifier and parameters...")
                joblib.dump(model, os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
                #joblib.dump(importances, os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
        self.write_metaparameters()
        logging.info("Configuration done!")
 
 
    def train_hist_model(self, data):
        
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
        #idx = np.random.choice(results.shape[0], size=results.shape[0], replace=False)
        #dt = results[idx, :]
        #print(dt.shape)
        #print(dt)
        
        X = data[:,:-1].astype(np.float32)
        y = data[:,-1]
        
        # X2 = data2[:,:-1].astype(np.float32)
        # y2 = data2[:,-1]
        
        # cols = []
        # min_val = np.min(X)
        # max_val = np.max(X)
        # print(min_val, max_val)
        # sc = np.array([min_val, max_val])
        # space = (max_val - min_val) / 20
        # vals = np.arange(min_val, max_val+space, space)
        # for i in range(len(vals)-1):
        #     cols.append(np.expand_dims(np.sum(np.logical_and(X >= vals[i], X < vals[i+1]),axis=1),-1))
        # X2 = np.concatenate((cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10], cols[11], cols[12], cols[13], cols[14], cols[15], 
        #                     cols[16], cols[17], cols[18], cols[19]),axis=1)
        # print(X.shape, X2.shape) 
        
        min_val = np.min(X)
        max_val = np.max(X)
        num_bins = 100
        #print(min_val, max_val)
        # sc = np.array([min_val, max_val])
        space = (max_val - min_val) / num_bins
        vals = np.arange(min_val, max_val+space, space)
        cols = []
        for i in range(len(vals)-1):
            cols.append(np.expand_dims(np.sum(np.logical_and(X >= vals[i], X < vals[i+1]),axis=1),-1))
        X2 = np.concatenate(([cols[i] for i in range(num_bins)]),axis=1)
        print(X.shape, X2.shape) 
        train_size = int(X2.shape[0]*0.75)
        X_train = X2[:train_size,:]
        X_test = X2[train_size:,:]
        #print(X_train.shape, X_test.shape)
        y_train = y[:train_size]
        y_test = y[train_size:]

        # clf = clf_rf.fit(X_train, y_train)
        # importance_full = np.argsort(clf.feature_importances_)
        # importance = importance_full[-100:]#*self.num_features:]
        # #clf = clf_lasso.fit(X_train, y_train)
        # #lasso_coef = np.abs(clf.coef_)
        # #
        # # importance = np.argsort(lasso_coef)[-1*self.num_features:]
        # X_train = X_train[:,importance]
        # #X_train = sc.fit_transform(X_train)
        # #X_train = scale(X_train, axis=1)
        # X_test = X_test[:,importance]
        # #X_test = sc.transform(X_test)
        # #X_test = scale(X_test, axis=1)

        clf = clf_rf
        #clf = CalibratedClassifierCV(clf, ensemble=False)
        #print(X_train.shape, X_test.shape)
        clf.fit(X_train, y_train)
        #self.custom_scoring_function(clf, X_test, y_test)
        #print(arch)
        print(clf.score(X_train, y_train), roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]), log_loss(y_train, clf.predict_proba(X_train)[:,1]))
        print(clf.score(X_test, y_test), roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), log_loss(y_test, clf.predict_proba(X_test)[:,1]))
        #print(self.custom_accuracy_function(clf, X_train, y_train), self.custom_scoring_function(clf, X_train, y_train), self.custom_loss_function(clf, X_train, y_train))
        #print(self.custom_accuracy_function(clf, X_test, y_test), self.custom_scoring_function(clf, X_test, y_test), self.custom_loss_function(clf, X_test, y_test))
        # X = X[:,importance]

        clf = clf.fit(X2, y)
        return clf
     
    def get_architecture_sizes(self, model_list):
        archs = []
        sizes = []

        for model_dirpath in model_list:
            with open(os.path.join(model_dirpath, "reduced-config.json")) as f:
                config = json.load(f)
            arch = config['cnn_type']
            if arch in archs:
                continue

            model_filepath = os.path.join(model_dirpath, "model.pt")
            model, model_repr, model_class = load_model(model_filepath)

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

    def weight_analysis_configure_hist(self, model, arch, size, device):
        model_size = len(list(model.named_parameters()))
        #print(model_size)
        if model_size != size:
            return None
        params = []
        counter = 0
        for param in model.named_parameters():
            if list(param[1].shape)[0] > 3000000:
                continue
            #if 'weight' in param[0]:
            layer = torch.flatten(param[1])
            layer = torch.sort(layer)[0]
            counter +=1
            params.append(layer)

        #if len(params) != size:
        #    return None, 0
        params = torch.cat((params), dim=0)
        return params
    
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
            layer = torch.sort(layer)[0]
            importance_indices = importances[counter]
            counter +=1
            weights = layer[importance_indices]
            params.append(weights)

        #if len(params) != size:
        #    return None, 0
        params = torch.cat((params), dim=0)
        return params
        
    def train_wa_model(self, dt):

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
        train_accs = []
        test_accs = []
        train_aucs = []
        test_aucs = []
        train_ces = []
        test_ces = []
        for _ in range(5):
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

            # clf = clf_rf.fit(X_train, y_train)
            # importance_full = np.argsort(clf.feature_importances_)
            # importance = importance_full[-50000:]#*self.num_features:]
            # #clf = clf_lasso.fit(X_train, y_train)
            # #lasso_coef = np.abs(clf.coef_)
            # #
            # # importance = np.argsort(lasso_coef)[-1*self.num_features:]
            # X_train = X_train[:,importance]
            # #X_train = sc.fit_transform(X_train)
            # #X_train = scale(X_train, axis=1)
            # X_test = X_test[:,importance]
            # #X_test = sc.transform(X_test)
            # #X_test = scale(X_test, axis=1)

            clf = clf_rf
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
        clf = clf.fit(X, y)
        return clf, [0]#importance
        
    def train_dubious_model(self, results):

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
   
    def train_jacobian_model(self, results):

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
        print(p)
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
        method = "wa"
        if method == "jacobian":
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
            
        if method == "wa":
            model_pt, model_repr, model_class = load_model(model_filepath)
            sizes = [62, 110]
            archs = ["ResNet18", "ResNet34"]

            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                size = sizes[arch_i]

                #clf = joblib.load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
                #importances = joblib.load(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))
                #overall_importances = joblib.load(os.path.join(self.learned_parameters_dirpath, "overallImp_"+arch+".joblib"))

                #features = self.weight_analysis_configure(model_pt.model, arch, size, importances, device)
                params = []
                for param_n, param in enumerate(model_pt.model.named_parameters()):
                    if param_n not in [0,1,5,8]: continue
                    layer = torch.flatten(param[1])
                    #layer = torch.sort(layer)[0][:10000]
                    params.append(layer)
                    
                feature_vector = torch.cat((params), dim=0)
                if len(list(model_pt.model.named_parameters())) == size:
                    results = feature_vector.detach().cpu().numpy().reshape(1,-1)
                    clf = joblib.load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
                #import math
                # if features != None:
                #     features = np.array(features.detach().cpu()).reshape(1,-1)
                #     results = features[:,overall_importances]
        if method == "hist":
            model_pt, model_repr, model_class = load_model(model_filepath)
            sizes = [62, 110]
            archs = ["ResNet18", "ResNet34"]

            for arch_i in range(len(archs)):

                arch = archs[arch_i]
                size = sizes[arch_i]

                clf = joblib.load(os.path.join(self.learned_parameters_dirpath, "clf_"+arch+".joblib"))
                #importances = joblib.load(os.path.join(self.learned_parameters_dirpath, "imp_"+arch+".joblib"))

                features = self.weight_analysis_configure_hist(model_pt.model, arch, size, device)
                #import math
                if features != None:
                    X = np.array(features.detach().cpu()).reshape(1,-1)
                    min_val = np.min(X)
                    max_val = np.max(X)
                    num_bins = 100
                    space = (max_val - min_val) / num_bins
                    vals = np.arange(min_val, max_val+space, space)
                    cols = []
                    for i in range(len(vals)-1):
                        cols.append(np.expand_dims(np.sum(np.logical_and(X >= vals[i], X < vals[i+1]),axis=1),-1))
                    results = np.concatenate(([cols[i] for i in range(num_bins)]),axis=1)

        if method == "dubious":
            model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
            exemplars = dict()
            exemplars[0] = 0
            exemplars[1] = 0
            features = []
            jacobians = []
            num_examples = 100
            max_iter = 25
            epsilon = 10
            for examples_dir_entry in os.scandir(examples_dirpath):
                if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                    base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                    ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                    if not os.path.exists(ground_truth_filename):
                        logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                        continue

                    new_input = torchvision.io.read_image(examples_dir_entry.path).float()
                    new_input.requires_grad = True
                    #print(new_input)
                    # if inputs_np is None:
                    #     inputs_np = new_input
                    # else:
                    #     inputs_np = np.concatenate([inputs_np, new_input])
                    with open(ground_truth_filename) as f:
                        data = int(json.load(f))
                    if data != 0:
                        continue
                    if exemplars[data] >= num_examples:
                        continue
                    #print(new_input.shape, data)
                    
                    logits = model_pt.predict(new_input)
                    #print(logits)
                    #print(1/0)
                    pred_label = torch.argmax(logits)
                    gradient = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                                grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                                only_inputs=True, retain_graph=True)[0][0]
                
                    signed_grad = torch.sign(gradient)
                    #print(gradient[0,0,:10,:10])
                    iters = 0
                    prediction = pred_label
                    while pred_label == prediction or torch.max(logits) < 0.9:
                        new_input = new_input + (epsilon * signed_grad)
                        # HERE batch_data = batch_data.cuda()
                        #batch_data = torch.clamp(batch_data, min=0, max=1)
                        logits = model_pt.predict(new_input)
                        prediction = torch.argmax(logits)
                        #print(logits)
                        iters += 1
                        if iters == max_iter: break
                    #print(iters)
                    #print(1/0)
                    jacobian = torch.autograd.grad(outputs=logits[0][1-pred_label], inputs=new_input,
                                grad_outputs=torch.ones(logits[0][1-pred_label].size()), 
                                only_inputs=True, retain_graph=True)[0][0]
                    jacobians.append(jacobian)
                    exemplars[data] += 1
            jacobians = torch.stack(jacobians, dim=0)
            #print(jacobians.shape)
            feature = torch.mean(jacobians, axis=0).cpu().numpy()
            features.append(feature)
            results = np.array(features)
            results = scale(results, axis=1)
            
            with open(join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
                clf = pickle.load(fp)
                
                
        if method == "bias":
            model_pt, model_repr, model_class = load_model(model_filepath)#.to(device)
            bias0 = model_pt.model.fc._parameters['bias'][0].detach().item()
            bias1 = model_pt.model.fc._parameters['bias'][1].detach().item()
            print(bias0, bias1)
            #print(1/0)
            if bias0 >= 0.0 and bias1 >= 0.0:
                probability = '0.75'
            else:
                probability = '0.25'
        if method != "bias":
            probability = clf.predict_proba(results)[0][1]
        probability = str(probability)
        
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
