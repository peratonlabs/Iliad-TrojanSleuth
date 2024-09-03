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
        
        #model_poison_prediction = self.trojan_detector(model, self.device)
        if model_poison_prediction == False:
            return x, {}
        y = torch.argmax(model(x.to(self.device)), dim=1)
        softmax = torch.nn.Softmax(dim=1)
                
        dubious = joblib.load("/dubious-50-blur-perturb50-eval.joblib")
        
        #transforms = [self.filter_rect]
        transforms = [self.blur]
        #transform = self.f
        parameter1_values = [0.1,0.2,0.3,0.5,0.7,1,1.2,1.5,2]#[0,1,2]
        parameter2_values = [0]#list(range(0,255,64))
        parameter3_values = [0]#list(range(0,255,64))
        parameter4_values = [0]#[-0.5,-0.1,0.1,0.5]#[-0.5,-0.1,0.1,0.5]
        parameter5_values = [False]

        greatest_metric = 0
        best_params = (0, 0, 0, 0, 0)
        num_perturb = 10
        logit_list = []
        total = 0.0
        correct = 0.0
        for transform_i in range(len(transforms)):
            transform = transforms[transform_i]
            for _ in range(num_perturb):
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
                                    #metric = torch.mean(torch.max(probs, dim=1)[0]) - (torch.sum(y.cpu() == torch.argmax(logits, dim=1)) / logits.shape[0])
                                    # metric = torch.sum(y.cpu() != torch.argmax(logits, dim=1)) / logits.shape[0]
                                    # #print(metric)
                                    # if metric > greatest_metric:
                                    #     greatest_metric = metric
                                    #     best_params = (parameter_i1, parameter_i2, parameter_i3, parameter_i4, parameter_i5)
        #print(np.mean(logit_list), np.std(logit_list), correct / total)
        feature_vector = [[np.mean(logit_list), np.std(logit_list), correct / total]]
        #print(feature_vector)
        poison_prediction = dubious.predict_proba(feature_vector)[0][1]
        #print(poison_prediction)
        #print(1/0)
        #print(best_params, greatest_metric)
        transform = self.filter_rect
        final_x = torch.tensor([])
        for i in range(x.shape[0]):
            img = copy.deepcopy(x[i:i+1])
            if poison_prediction > 0.5:
                #print("Poioson Prediction")
                img = transform(img, 0, best_params[1], best_params[2], 0.4, best_params[4])
                img = transform(img, 1, best_params[1], best_params[2], 0.4, best_params[4])
                img = transform(img, 2, best_params[1], best_params[2], 0.4, best_params[4])
            else:
                img = transform(img, best_params[0], best_params[1], best_params[2], 0.1, best_params[4])
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
    
    def blur(self, x, magnitude, x_index, y_index, addition, set_value):
        x = x + torch.FloatTensor(np.random.normal(0,magnitude,x.shape))
        return x
    
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
        
    
    # def get_signature(X_test, y_test, num_examples, model_type, model_pt, drop_rates, num_perturb, perturbation_type, target_class, additional_args, data_type):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     ratios = []
    #     logit_values = []
    #     logits_sds = []
    #     use_benign_values = True

    #     if data_type == "clean":
    #         for drop_rate in drop_rates:

    #             class_counts = {0:0, 1:0}
    #             for i in range(X_test.shape[0]):
    #                 label = y_test[i]
    #                 if class_counts[label] >= num_examples:
    #                     break
    #                 #print(label, target_class)
    #                 if label != target_class:
    #                     continue

    #                 fn = X_test[i:i+1]
    #                 #model_input = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args)
    #                 #logits = model_pt(model_input).detach().cpu().numpy()
    #                 logits = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args, perturbation_type)
    #                 #gradient = torch.autograd.grad(outputs=logits[0][1-label], inputs=embedding_vector, grad_outputs=torch.ones(logits[0][0].size()).to("cuda"), only_inputs=True, retain_graph=True)[0][0].cpu().numpy()
    #                 #print(torch.argmax(logits), torch.max(logits), label)
    #                 pred_label = np.argmax(logits)
    #                 #print(logits, pred_label, label)
    #                 #print(1/0)
    #                 if (pred_label != label):
    #                     continue
    #                 class_counts[label] += 1

    #                 correct = 0.0
    #                 logit_list = []
    #                 for perturbation in range(num_perturb):

    #                     random.seed(i+drop_rate+perturbation)
    #                     additional_args["perturbation"] = perturbation
    #                     #model_input = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args)
    #                     #logits = model_pt(model_input).detach().cpu().numpy()
    #                     logits = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args, perturbation_type)
    #                     logit_list.append(logits[0,pred_label])
    #                     #print(torch.argmax(logits), pred_label)
    #                     if (np.argmax(logits) == pred_label):
    #                         correct += 1
    #                 ratio = correct / num_perturb
    #                 #print(drop_rate, "C",ratio,  np.mean(logit_list), np.std(logit_list))
    #                 ratios.append(ratio)
    #                 logit_values.append(np.mean(logit_list))
    #                 logits_sds.append(np.std(logit_list))

    #             #print(class_counts) 
            
    #     if data_type == "poisoned":    
    #         #print(1/0)
    #         for drop_rate in drop_rates:
    #             #if drop_rate != 400: continue
    #             #print(drop_rate)
    #             class_counts = {0:0, 1:0}
    #             for i in range(X_test.shape[0]):#-1000):
    #                 #fn = os.path.join(examples_dirpath, "source_class_1_target_class_0_example_21.txt")
    #                 label = y_test[i]
    #                 if class_counts[label] >= num_examples:
    #                     break
    #                 fn = X_test[i:i+1]
    #                 #model_input = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args)
    #                 #logits = model_pt(model_input).detach().cpu().numpy()
    #                 logits = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args, perturbation_type)
    #                 #print(logits, np.argmax(logits), label)
    #                 pred_label = np.argmax(logits)

    #                 if pred_label != target_class:
    #                     continue

    #                 class_counts[label] += 1
                    
    #                 correct = 0.0
    #                 logit_list = []
    #                 for perturbation in range(num_perturb):

    #                     random.seed(i+drop_rate+perturbation)
    #                     additional_args["perturbation"] = perturbation
    #                     #model_input = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args)
    #                     #logits = model_pt(model_input).detach().cpu().numpy()
    #                     logits = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args, perturbation_type)
    #                     logit_list.append(logits[0,pred_label])
    #                     if (np.argmax(logits) == pred_label):
    #                         correct += 1
    #                 ratio = correct / num_perturb
    #                 #print(drop_rate, "T",ratio, np.mean(logit_list), np.std(logit_list))
    #                 ratios.append(ratio)
    #                 logit_values.append(np.mean(logit_list))
    #                 logits_sds.append(np.std(logit_list))
    #             #print(class_counts)  

    #     num_samples = len(logit_values) // len(drop_rates)
    #     logit_values = np.array(logit_values).reshape(num_samples, len(drop_rates))
    #     logits_sds = np.array(logits_sds).reshape(num_samples, len(drop_rates))
    #     ratios = np.array(ratios).reshape(num_samples, len(drop_rates))
    #     #print(num_samples, logit_values.shape)
        
    #     return logit_values, logits_sds, ratios#, trojan_logits, trojan_logits_sds, trojan_ratios
    
    # def perturb(sample, model_pt, device, parameters, step_sizes, apply, label, magnitude):
    #     if apply:
    #         bounds = [100, 100]
    #         bounds[0] = int(bounds[0] / magnitude)
    #         bounds[1] = int(bounds[1] * magnitude)
    #         #bounds[0] = int(bounds[0] * (1-magnitude))
    #         #bounds[1] = int(bounds[1] * (1+magnitude))
    #         #param1 = 100
    #         #param2 = 100
    #         #param3 = 100
    #         param_i = random.randint(0,2)
    #         if param_i == 0:
    #             param1 = random.randint(bounds[0],bounds[1])
    #         if param_i == 1:
    #             param2 = random.randint(bounds[0],bounds[1])
    #         if param_i == 2:
    #             param3 = random.randint(bounds[0],bounds[1])
            
    #         param1 = random.randint(bounds[0],bounds[1])
    #         param2 = random.randint(bounds[0],bounds[1])
    #         param3 = random.randint(bounds[0],bounds[1])
    #         #print(param1, param2, param3)
            
    #         with wand.image.Image(filename=fn) as wand_image:
    #             img = wand_image.clone()
    #         img.modulate(param1, param2, param3)
    #         img = np.array(img)
    #         r = img[:, :, 0]
    #         g = img[:, :, 1]
    #         b = img[:, :, 2]
    #         #wand_img = np.stack((r, g, b), axis=2)
    #         img = np.transpose(img, (2, 0, 1))
    #         img = np.expand_dims(img, 0)
    #         img = img - np.min(img)
    #         img = img / np.max(img)
    #         batch_data = torch.FloatTensor(img).to(device)

    #         # direction = (random.randint(0,1)-0.5)*2
    #         # parameters = parameters + step_sizes * magnitude * direction
    #         relu = torch.nn.ReLU()
    #         parameters[0] = relu(parameters[0])
    #         parameters[1] = relu(parameters[1])
    #         parameters[2] = relu(parameters[2])
    #         parameters[9] = torch.clamp(parameters[9],0,255)
    #         parameters[10] = torch.clamp(parameters[10],0,255)
    #         parameters[11] = torch.clamp(parameters[11],0,255)

    #     else:
    #         img = skimage.io.imread(fn)
    #         r = img[:, :, 0]
    #         g = img[:, :, 1]
    #         b = img[:, :, 2]
    #         #img = np.stack((r, g, b), axis=2)
    #         img = np.transpose(img, (2, 0, 1))
    #         img = np.expand_dims(img, 0)
    #         img = img - np.min(img)
    #         img = img / np.max(img)
    #         batch_data = torch.FloatTensor(img).to(device)
        
    #     logits = model_pt(torch.from_numpy(sample).float().to(device).reshape(1,1,28,28)).detach().cpu().numpy()
    #     return logits


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
    

    # def colortone(self, image: wand.image.Image, color: str, dst_percent: int, invert: bool) -> None:
    #     """
    #     tones either white or black values in image to the provided color,
    #     intensity of toning depends on dst_percent
    #     :param image: provided image
    #     :param color: color to tone image
    #     :param dst_percent: percentage of image pixel value to include when blending with provided color,
    #     0 is unchanged, 100 is completely colored in
    #     :param invert: if True blacks are modified, if False whites are modified
    #     :return:
    #     """
    #     mask_src = image.clone()
    #     mask_src.colorspace = 'gray'
    #     if invert:
    #         mask_src.negate()
    #     mask_src.alpha_channel = 'copy'

    #     src = image.clone()
    #     src.colorize(wand.color.Color(color), wand.color.Color('#FFFFFF'))
    #     src.composite_channel('alpha', mask_src, 'copy_alpha')

    #     image.composite_channel('default_channels', src, 'blend',
    #                     arguments=str(dst_percent) + "," + str(100 - dst_percent))
    #     return image

    # def vignette(self, image: wand.image.Image, color_1: str = 'none', color_2: str = 'black',
    #         crop_factor: float = 1.5) -> None:
    #     """
    #     applies fading from color_1 to color_2 in radial gradient pattern on given image
    #     :param image: provided image
    #     :param color_1: center color
    #     :param color_2: edge color
    #     :param crop_factor: size of radial gradient pattern, which is then cropped and combined with image,
    #     larger values include more of color_1, smaller values include more of color_2
    #     :return: None
    #     """
    #     crop_x = math.floor(image.width * crop_factor)
    #     crop_y = math.floor(image.height * crop_factor)
    #     src = wand.image.Image()
    #     src.pseudo(width=crop_x, height=crop_y, pseudo='radial-gradient:' + color_1 + '-' + color_2)
    #     src.crop(0, 0, width=image.width, height=image.height, gravity='center')
    #     src.reset_coords()
    #     image.composite_channel('default_channels', src, 'multiply')
    #     image.merge_layers('flatten')
