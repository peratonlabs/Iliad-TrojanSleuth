import os
import joblib
from typing import Dict
from pathlib import Path
import copy
from joblib import load
from sklearn.preprocessing import StandardScaler, scale

import torchvision
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms import v2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
#import wand
#import wand.image
from PIL import Image
import numpy as np

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel

class DubiousTrojai(TrojAIMitigation):
    def __init__(self, device, batch_size=32, num_workers=1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        
    def preprocess_transform(self, x: torch.tensor, model):
        # model_filepath=self.model_filepath
        # model = torch.load(model_filepath)
        # model = model.to(device=self.device)
        
        model_poison_prediction = self.trojan_detector(model, self.device)
        if model_poison_prediction == False:
            return x, {}
        y = torch.argmax(model(x.to(self.device)), dim=1)
        
        softmax = torch.nn.Softmax(dim=1)
        # transforms = [self.instagram]
        # parameter_values = [50,100]#[5, 25, 50, 100]
        # greatest_metric = 0
        # best_params = (100, 100, 100)
        # for transform_i in range(len(transforms)):
        #     transform = transforms[transform_i]
        #     for parameter_i1 in parameter_values:
        #         for parameter_i2 in parameter_values:
        #             for parameter_i3 in parameter_values:
        #                 #print(parameter_i1, parameter_i2, parameter_i3)
        #                 new_x = torch.tensor([])
        #                 for i in range(x.shape[0]):
        #                     img = copy.deepcopy(x[i:i+1])
        #                     img = transform(img, parameter_i1, parameter_i2, parameter_i3)
        #                     new_x = torch.cat([new_x, img], axis=0)
                
        #                 logits = model(new_x.to(self.device)).detach().cpu()
        #                 probs = softmax(logits)
        #                 #print(torch.sum(y == torch.argmax(logits, dim=1)) / logits.shape[0], torch.mean(torch.max(probs, dim=1)[0]))
        #                 metric = torch.mean(torch.max(probs, dim=1)[0]) - (torch.sum(y == torch.argmax(logits, dim=1)) / logits.shape[0])
        #                 #print(metric)
        #                 if metric > greatest_metric:
        #                     greatest_metric = metric
        #                     best_params = (parameter_i1, parameter_i2, parameter_i3)
        transform = self.control
        new_x = torch.tensor([])
        for i in range(x.shape[0]):
            img = copy.deepcopy(x[i:i+1])
            img = transform(img)
            new_x = torch.cat([new_x, img], axis=0)
        logits = model(new_x.to(self.device)).detach().cpu()
        pred_class = torch.argmax(logits)
        #print(pred_class, y)
        
        dubious = joblib.load("/dubious-10.joblib")
        
        transforms = [self.filter_rect]
        #transform = self.f
        parameter1_values = [0,1,2]
        parameter2_values = [0]#list(range(0,255,64))
        parameter3_values = [0]#list(range(0,255,64))
        parameter4_values = [-0.5,-0.1,0.1,0.5]
        parameter5_values = [False]

        greatest_metric = 0
        best_params = (0, 0, 0, 0, 0)
        logit_list = []
        total = 0.0
        correct = 0.0
        for transform_i in range(len(transforms)):
            transform = transforms[transform_i]
            for parameter_i1 in parameter1_values:
                for parameter_i2 in parameter2_values:
                    for parameter_i3 in parameter3_values:
                        for parameter_i4 in parameter4_values:
                            for parameter_i5 in parameter5_values:
                                #print(parameter_i1, parameter_i2, parameter_i3, parameter_i4, parameter_i5)
                                new_x = torch.tensor([])
                                for i in range(x.shape[0]):
                                    img = copy.deepcopy(x[i:i+1])
                                    img = transform(img, parameter_i1, parameter_i2, parameter_i3, parameter_i4, parameter_i5)
                                    new_x = torch.cat([new_x, img], axis=0)
                                logits = model(new_x.to(self.device)).detach().cpu()
                                probs = softmax(logits)
                                pred_class = torch.argmax(logits)
                                #print(pred_class, y)
                                total += 1
                                if pred_class == y[0]:
                                    correct += 1
                                logit_list.append(logits[0,pred_class].numpy())
                                #print(torch.sum(y == torch.argmax(logits, dim=1)) / logits.shape[0], torch.mean(torch.max(probs, dim=1)[0]))
                                metric = torch.mean(torch.max(probs, dim=1)[0]) - (torch.sum(y.cpu() == torch.argmax(logits, dim=1)) / logits.shape[0])
                                #print(metric)
                                if metric > greatest_metric:
                                    greatest_metric = metric
                                    best_params = (parameter_i1, parameter_i2, parameter_i3, parameter_i4, parameter_i5)
        #print(np.mean(logit_list), np.std(logit_list), correct / total)
        feature_vector = [[np.mean(logit_list), np.std(logit_list), correct / total]]
        poison_prediction = dubious.predict(feature_vector)
        #print(poison_prediction)
        #print(1/0)
        #print(best_params, greatest_metric)
        final_x = torch.tensor([])
        for i in range(x.shape[0]):
            img = copy.deepcopy(x[i:i+1])
            if poison_prediction:
                img = transform(img, best_params[0], best_params[1], best_params[2], best_params[3], best_params[4])
            final_x = torch.cat([final_x, img], axis=0)
        logits = model(final_x.to(self.device)).detach().cpu()
        #print("Mean: ", torch.mean(logits))
        probs = softmax(logits)
        #print(torch.argmax(logits, dim=1), y)
        #print(torch.sum(y == torch.argmax(logits, dim=1)) / logits.shape[0], torch.mean(torch.max(probs, dim=1)[0]))
        #print(x.shape)
        # new_x = torch.tensor([])
        # for i in range(x.shape[0]):
        #     img = x[i]
        #     img_filepath = os.path.join(self.scratch_dirpath, "img.png")
        #     param1 = 100
        #     param2 = 5
        #     param3 = 100
        #     save_image(img, img_filepath)
        #     with wand.image.Image(filename=img_filepath) as wand_image:
        #         img = wand_image.clone()
        #         img.modulate(param1, param2, param3)
        #         #img = self.colortone(img, self.int2hex(255,10,10), 20, True)
        #         img.save(filename=img_filepath)
        #     img = Image.open(img_filepath)
        #     test_augmentation_transforms = torchvision.transforms.Compose(
        #         [
        #             v2.PILToTensor(),
        #             torchvision.transforms.ConvertImageDtype(torch.float),
        #         ]
        #     )
        #     img = test_augmentation_transforms(img)
        #     img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        #     new_x = torch.cat([new_x, img], axis=0)
        #print(new_x.shape)
        #x[:,0,:,:] -= 0.7
        #x[:,1,:,:] += 0.7
        #x[:,2,:,:] += 0.7
        #new_x[:,0,:,:] += 0.7
        return final_x, {}
        
    def control(self, x):
        return x
    
    def filter(self, x, index, addition, set_value):
        if set_value:
            x[:,index,:,:] = addition
        else:
            x[:,index,:,:] = x[:,index,:,:] + addition
        return x
    
    def filter_rect(self, x, color_index, x_index, y_index, addition, set_value):
        if set_value:
            x[:,color_index,:,:] = addition
        else:
            x[:,color_index,x_index:x_index+256,y_index:y_index+256] = x[:,color_index,x_index:x_index+256,y_index:y_index+256] + addition
            #x[:,color_index,:,:] = x[:,color_index,:,:] + addition
        return x
    
    def instagram(self, x, p1, p2, p3):
        img_filepath = os.path.join(self.scratch_dirpath, "img.png")
        param1 = p1
        param2 = p2
        param3 = p3
        save_image(x, img_filepath)
        with wand.image.Image(filename=img_filepath) as wand_image:
            img = wand_image.clone()
            img.modulate(param1, param2, param3)
            #img = self.colortone(img, self.int2hex(255,10,10), 20, True)
            img.save(filename=img_filepath)
        img = Image.open(img_filepath)
        test_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )
        img = test_augmentation_transforms(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        return img
    
    def int2hex(self, r,g,b):
        r = np.clip(r,0,255)
        g = np.clip(g,0,255)
        b = np.clip(b,0,255)
        r_char = hex(int(r))[2:]
        if len(r_char) == 1: r_char = '0'+r_char
        g_char = hex(int(g))[2:]
        if len(g_char) == 1: g_char = '0'+g_char
        b_char = hex(int(b))[2:]
        if len(b_char) == 1: b_char = '0'+b_char
        return '#'+r_char+g_char+b_char
    
    def trojan_detector(self, model, device):
        sizes = [158, 161, 152]
        archs = ["mobile", "resnet", "vit"]
        trojan_detector_path = "/trojan_detectors/"

        for arch_i in range(len(archs)):

            arch = archs[arch_i]
            size = sizes[arch_i]

            clf = load(trojan_detector_path+"clf_"+arch+".joblib")
            importances = load(trojan_detector_path+"imp_"+arch+".joblib").tolist()
            overall_importances = load(trojan_detector_path+"overallImp_"+arch+".joblib")

            features = self.weight_analysis_configure(model, arch, size, importances, device)
            if features != None:

                features = np.array(features.detach().cpu()).reshape(1,-1)
                features_full = features[:,overall_importances]
                trojan_probability = clf.predict_proba(scale(features_full,axis=1))[0][1]
        #print(trojan_probability)
        return trojan_probability > 0.5
    
    def weight_analysis_configure(self, model, arch, size, importances, device):
        #print(model)
        layers = []
        for param in model.named_parameters():
            #if 'weight' in param[0]:
            #print(param[0])
            layers.append(torch.flatten(param[1]))
        model_size = len(layers)
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

        #print(len(params))
        params = torch.cat((params), dim=0)
        return params
        


    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Args:
            model: the model to repair
            dataset: a dataset of examples
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        model = model.to(self.device)
        return TrojAIMitigatedModel(model, custom_preprocess=self.preprocess_transform)
    

