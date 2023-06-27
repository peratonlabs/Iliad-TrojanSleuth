# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import math
import logging
import json
import jsonpickle
import pickle
import copy

import torch
import torchvision
import skimage.io
from scipy import stats
import numpy as np

from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_ground_truth
from PIL import Image
import wand.image
import wand.color
from wand.drawing import Drawing
from wand.color import Color
from wand.display import display


Background_dirpath = "backgrounds"

def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


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
        metaparameters = {}

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        #for random_seed in np.random.randint(1000, 9999, 10):
        #    self.weight_params["rso_seed"] = random_seed
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        features = []
        labels = []
        load_data = False

        if load_data:
            data = np.loadtxt("rd13.csv", delimiter=",")
            self.save_results(data)
            return


        for i, model_dir in enumerate(sorted(os.listdir(models_dirpath))[:]):
            #print(model_dir)

            #if i != 45: continue

            print(os.path.join(models_dirpath, model_dir))
            if self.device == "cpu":
                model = torch.load(os.path.join(models_dirpath, model_dir, "model.pt"), map_location=torch.device(self.device))
            else:
                model = torch.load(os.path.join(models_dirpath, model_dir, "model.pt")).cuda()
            model.eval()

            with open(os.path.join(models_dirpath, model_dir, "config.json")) as f:
                config = json.load(f)

            arch = config["py/state"]["model_architecture"]

            # example_image = os.path.join(models_dirpath, model_dir, "clean-example-data/90.png")

            augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

            # img = skimage.io.imread(os.path.join(models_dirpath, model_dir, "poisoned-example-data/12.png"))
            # image = torch.as_tensor(img)
            # # move channels first
            # image = image.permute((2, 0, 1))
            # # convert to float (which normalizes the values)
            # image = augmentation_transforms(image)
            # image = image.to(self.device)
            # # Convert to NCHW
            # img = image.unsqueeze(0)
            # # wrap the network outputs into a list of annotations
            # pred, scores = self.inference_sample(model, img, False, .10)
            # print(pred, scores)
            # print(1/0)

            train_images, val_images, test_images = self.generate_images(models_dirpath, model_dir, model)
            #if train_images == None:
            #    continue
            # images = self.gather_images(os.path.join(models_dirpath, model_dir, "clean-example-data"))
            # images = dict()

            # with open(os.path.join(models_dirpath, model_dir, "fg_class_translation.json")) as f:
            #     mapping = json.load(f)
            # #print(mapping)
            # #print(1/0)
            # reverse_mapping = {value:key for key, value in mapping.items()}

            # try:
            #     trigger_dirpath = os.path.join(models_dirpath, model_dir, "foregrounds")
            # except:
            #     return None
            # augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

            # for trigger in os.listdir(trigger_dirpath):
            #     image_class = int(reverse_mapping[trigger])
            #     trigger_filepath = os.path.join(trigger_dirpath, trigger)
            #     sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            #     sign_size = 256
            #     image = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            #     image = torch.as_tensor(image)
            #     image = image.permute((2, 0, 1))
            #     image = augmentation_transforms(image)
            #     image = image.to(self.device)
            #     img = image.unsqueeze(0)
            #     if image_class in images:
            #         images[image_class].append(img)
            #     else:
            #         images[image_class] = [img]


            label = np.loadtxt(os.path.join(models_dirpath, model_dir, 'ground_truth.csv'), dtype=bool)
            labels.append(label)
            #print(1/0)
            #feature_vector = self.get_features_injection(train_images, model)
            #feature_vector = self.get_features_evasion(train_images, val_images, test_images, model)
            #print(feature_vector)
            #print(1/0)
            #feature_vector = self.get_features_sign(models_dirpath, model_dir, model)
            #feature_vector = self.get_features_misclass(images, model)
            if arch == "object_detection:detr":
                feature_vector = self.get_features_detr(train_images, val_images, test_images, model)
            else:
                feature_vector = self.get_features_other(train_images, model)
            features.append(feature_vector)
            #print(features)
            #print(1/0)
            data = np.concatenate((np.array(features), np.expand_dims(np.array(labels),-1)), axis=1)
            #print(data)
            np.savetxt("rd13.csv", data, delimiter=",")

        self.save_results(data)

    def get_features_sign(self, models_dirpath, model_dir, model):

        images = dict()

        with open(os.path.join(models_dirpath, model_dir, "fg_class_translation.json")) as f:
            mapping = json.load(f)
        #print(mapping)
        #print(1/0)
        reverse_mapping = {value:key for key, value in mapping.items()}

        try:
            trigger_dirpath = os.path.join(models_dirpath, model_dir, "foregrounds")
        except:
            return None
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            sign_size = 256
            image = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            image = torch.as_tensor(image)
            image = image.permute((2, 0, 1))
            image = augmentation_transforms(image)
            image = image.to(self.device)
            img = image.unsqueeze(0)
            if image_class in images:
                images[image_class].append(img)
            else:
                images[image_class] = [img]

        train_len = 1
        val_len = 1
        epsilon = 0.05
        max_iter = 50
        misclass_scores = dict()
        misclass_increases = []
        triggers = []
        tgts = []
        #trigger_fns = []
        src_classes = []
        found = False
        trigger_locs = [100]

        all_labels = []
        all_confs = []

        for trigger_loc_i in range(len(trigger_locs)):

            for src_cls in images:
                #if src_cls < 60: continue#9: continue
                if src_cls != 9: continue
                #if int(image_class_dirpath) %3 != 1: continue
                #print(src_cls)
                #if isinstance(images[src_cls][0], int):
                #    continue
                fns = images[src_cls]
                #fns.sort()
                train_images = []
                val_images = []
                # for fn_i, fn in enumerate(images[src_cls]):

                #     if fn_i < train_len:
                #         train_images.append(fn)

                #     if fn_i >= train_len and fn_i < train_len+val_len:
                #         val_images.append(fn)

                first_trigger = True
                visualize = False
                threshold = .50
                train_images_send = copy.deepcopy(train_images)
                val_images_send = copy.deepcopy(val_images)

                confidences = []
                pred_labels = []

                for tgt_cls in images:#logits.shape[2]-65):
                    if tgt_cls != 20: continue#23: continue
                    #if tgt_cls%10 != 0: continue
                    #if tgt_cls == src_cls: continue
                    #print(src_cls, tgt_cls)
                    
                    if visualize:
                        image = fns[0][0]
                        image = image.permute((1, 2, 0))*255
                        image = image.cpu().numpy()
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.show()
                        image.save('start_img.png')
                    #print(1/0)
                    new_data = fns[0]

                    trigger_insertion_loc = trigger_locs[trigger_loc_i]
                    trigger_size = 70
                    filter_shape = torch.zeros(new_data.shape).to(self.device)
                    filter_shape[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size] = 1
                    #print(new_data.shape, filter_shape.shape)
                    pred, scores = self.inference_sample(model, new_data, False, threshold)
                    print(pred, scores)
                    #print(torch.max(new_data))
                    #print(pred)

                    triggered_data, misclass_label = self.generate_trigger(max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold)
                    if triggered_data == None:
                        continue        
                    
                    #print(src_cls, misclass_label)
                    #print(1/0)
                    trigger = triggered_data * filter_shape

                    if visualize:
                        image = trigger[0]
                        image = image.permute((1, 2, 0))*255
                        image = image.cpu().numpy()
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.show()
                        image.save('triggered_img.png')

                    test_img = fns[2]

                    #test_img = test_img * (1-filter_shape)
                    #test_img = test_img + trigger
                    #print(new_data.shape, trigger.shape, test_img.shape)

                    test_insertion_loc = trigger_insertion_loc + 30
                    test_img[:,:,test_insertion_loc:test_insertion_loc+trigger_size,test_insertion_loc:test_insertion_loc+trigger_size] = trigger[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size]

                    if visualize:
                        image = test_img[0]
                        image = image.permute((1, 2, 0))*255
                        image = image.cpu().numpy()
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.show()
                        image.save('test_img.png')
                    
                    # outputs = model(test_img)
                    # outputs = outputs[0]  # unpack the batch size of 1
                    # boxes = outputs['boxes'].cpu().detach().numpy()
                    # scores = outputs['scores'].cpu().detach().numpy()
                    # labels = outputs['labels'].cpu().detach().numpy()
                    # # wrap the network outputs into a list of annotations
                    # pred = utils.models.wrap_network_prediction(boxes, labels)
                    pred, scores = self.inference_sample(model, test_img, False, threshold)
                    if len(pred) == 0:
                        continue
                    #print(pred, scores)
                    prediction = pred[-1]['label'] - 1
                    if prediction != misclass_label:
                        continue
                    if scores[-1] < 0.85:
                        continue
                                                
                    #print(src_cls, prediction, scores[-1])
                    return [True]
                    #print(pred, scores)
                    #continue#print(1/0)
                    
                    try:
                        confidences.append(scores[-1])
                        pred_labels.append(pred[-1]['label'])
                    except:
                        confidences.append(0)
                        pred_labels.append(-1)
                        #print("Failed", src_cls, tgt_cls)

                    # #f list(labels2).count(j) <= object_count:
                    # #    continue
                    # #trigger = trigger[:,:50,:50]
                    # triggers.append(trigger.detach())
                    # tgts.append(tgt_cls)
                    # src_classes.append(src_cls)

                    # if first_trigger: break
                    # if len(triggers) > 20:
                    #     break
                    #print(1/0)

                #mmc_tgt, mmc_label_tgt = get_mmc(pred_labels)
                #max_mmc_conf_tgt, mean_mmc_conf_tgt = get_mmc_score(confidences, mmc_label_tgt)
                #features.append([mmc_tgt, max_mmc_conf_tgt, mean_mmc_conf_tgt])
                all_labels.append(pred_labels)
                all_confs.append(confidences)
        return [False]

    
    def get_features_misclass(self, data, model):
        train_len = 2
        val_len = 0
        test_len = 2
        epsilon = 0.05
        max_iter = 10
        misclass_scores = dict()
        misclass_increases = []
        triggers = []
        tgts = []
        #trigger_fns = []
        src_classes = []
        found = False

        all_labels = []
        all_confs = []

        color_params = [0,64,128,200]#[0,64,128,200]

        for src_cls in data:
            #if src_cls < 60: continue#9: continue
            #if src_cls != 61: continue
            #if int(image_class_dirpath) %3 != 1: continue
            #print(src_cls)
            #if isinstance(images[src_cls][0], int):
            #    continue
            images = data[src_cls][0]
            #print("Image length", len(images))
            boxes = data[src_cls][1]
            #fns.sort()
            train_images = []
            train_fns = []
            train_boxes = []
            val_images = []
            val_boxes = []
            val_fns = []
            test_images = []
            test_boxes = []
            test_fns = []
            for img_i, img in enumerate(images):

                if img_i < train_len:
                    train_images.append(self.preprocess_image(img))
                    train_fns.append(img)
                    train_boxes.append(boxes[img_i])
                if img_i >= train_len and img_i < train_len+val_len:
                    val_images.append(self.preprocess_image(img))
                    val_fns.append(img)
                    val_boxes.append(boxes[img_i])
                if img_i >= train_len+val_len and img_i < train_len+val_len+test_len:
                    test_images.append(self.preprocess_image(img))
                    test_fns.append(img)
                    test_boxes.append(boxes[img_i])

            test_images = train_images + test_images
            test_fns = train_fns + test_fns
            test_boxes = train_boxes + test_boxes
            #print(len(train_images))
            #print(train_fns, test_fns)

            first_trigger = True
            visualize = False
            offset = 1
            threshold = .75
            trigger_size = 25
            dist_threshold = 0.1

            confidences = []
            pred_labels = []

            #for tgt_cls in data:#logits.shape[2]-65):
            #if tgt_cls != 44: continue#23: continue
            #if tgt_cls%5 != 0: continue
            #if tgt_cls == src_cls: continue
            #print(src_cls, tgt_cls)
            train_images_send = copy.deepcopy(train_images)
            val_images_send = copy.deepcopy(val_images)
            test_images_send = copy.deepcopy(test_images)

            triggers = []
            tgt_classes = []
            trigger_locs = []
            types = []
            trigger_set = set()
            for trigger_loc in range(2):
                for img_i in range(len(train_fns)):
                    #if img_i<=3: continue
                
                    # if visualize:
                    #     image = train_images[0][0]
                    #     image = image.permute((1, 2, 0))*255
                    #     image = image.cpu().numpy()
                    #     image = Image.fromarray(image.astype('uint8'), 'RGB')
                    #     #image.show()
                    #     image.save('start_img.png')
                    #print(1/0)
                    selected_img = train_images[img_i]
                    selected_box = train_boxes[img_i]
                    #print(selected_box)
                    if trigger_loc==0:
                        trigger_insertion_x1 = int(selected_box['x1'] + selected_box['x2'])//2
                        trigger_insertion_y1 = int(selected_box['y1'] + selected_box['y2'])//2
                    if trigger_loc==1:
                        trigger_insertion_x1 = 20
                        trigger_insertion_y1 = 20
                    #int(selected_box['x2'] - trigger_insertion_x1)
                    #print(trigger_size)
                    # filter_shape = torch.zeros(selected_img.shape).to(self.device)
                    # filter_shape[:,:,trigger_insertion_y1:trigger_insertion_y1+trigger_size,trigger_insertion_x1:trigger_insertion_x1+trigger_size] = 1
                    # #print(new_data.shape, filter_shape.shape)
                    og_pred, og_scores = self.inference_sample(model, selected_img, False, threshold)
                    #print(og_pred, og_scores)
                    #print(torch.max(new_data))
                    #print(pred)
                    #selected_img[:,:,trigger_insertion_y1:trigger_insertion_y1+trigger_size,trigger_insertion_x1:trigger_insertion_x1+trigger_size] = 0.5
                    
                    parameters = torch.tensor([4,0,0,0,trigger_size/2/256.0,trigger_insertion_x1+(trigger_size/2),trigger_insertion_y1+(trigger_size/2)])

                    # wand_img = self.draw_polygon(train_fns[0], parameters)
                    # image = np.array(wand_img)[:,:,:-1]
                    # selected_img = torch.as_tensor(image)/255
                    # selected_img = selected_img.permute((2, 0, 1))
                    # selected_img = selected_img.unsqueeze(0)
                    # pred, scores = self.inference_sample(model, selected_img, False, threshold)
                    # print(pred, scores)
                    #continue
                    for color_i in range(len(color_params)):
                        color = color_params[color_i]
                        for color_i2 in range(len(color_params)):
                            color2 = color_params[color_i2]
                            for color_i3 in range(len(color_params)):
                                color3 = color_params[color_i3]  

                                parameters[1] = color
                                parameters[2] = color2
                                parameters[3] = color3

                                #print(train_images[0])
                                wand_img = self.draw_polygon(train_fns[img_i], parameters)
                                image = np.array(wand_img)[:,:,:-1]
                                #print(image.shape)
                                #image = np.transpose(img, (2, 0, 1))
                                #print(image.shape)
                                #print(np.max(image))#*255
                                # if visualize:
                                #     img = Image.fromarray(image.astype('uint8'), 'RGB')
                                #     #img.show()
                                #     img.save('start_img.png')

                                selected_img = torch.as_tensor(image)/255
                                selected_img = selected_img.permute((2, 0, 1))
                                #selected_img = augmentation_transforms(selected_img)
                                selected_img = selected_img.unsqueeze(0)

                                #print(new_data.shape, filter_shape.shape)
                                pred, scores = self.inference_sample(model, selected_img, False, threshold)
                                #print(pred, scores)
                                #print(parameters[1:4])
                                #print(og_pred, og_scores)
                                mapping = dict()
                                for pred_i in range(len(pred)):
                                    min_distance=1000000
                                    for pred_i2 in range(len(og_pred)):
                                        distance = self.dist(pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                                        #print(distance, pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                                        if distance < dist_threshold and distance < min_distance:
                                            min_distance = distance
                                            mapping[pred_i] = pred_i2
                                #print(mapping)
                                for key in mapping:
                                    #print(og_pred[mapping[key]]['label'], pred[key]['label'])
                                    #if og_pred[key]['label'] == src_cls+offset and pred[mapping[key]]['label'] == tgt_cls+offset:
                                    #if len(og_pred) > 0 and len(pred) > 0:
                                    if (copy.deepcopy(parameters[1:4]), trigger_loc, pred[key]['label']) in trigger_set:
                                        continue
                                    if og_pred[mapping[key]]['label'] != pred[key]['label']:#og_pred[mapping[key]]['label'] == src_cls+offset and
                                        #print(og_pred[mapping[key]]['label'], pred[key]['label'])
                                        #print("img_i: " ,img_i, og_pred[mapping[key]]['label'], pred[key]['label'])
                                        triggers.append(copy.deepcopy(parameters[1:4]))
                                        tgt_classes.append(pred[key]['label'])
                                        trigger_locs.append(trigger_loc)
                                        types.append("Misclass")
                                        trigger_set.add((copy.deepcopy(parameters[1:4]), trigger_loc, pred[key]['label']))

                                # for pred_i in range(len(pred)):
                                #     if (copy.deepcopy(parameters[1:4]), trigger_loc, pred[pred_i]['label']) in trigger_set:
                                #         continue
                                #     if pred_i not in mapping:# and pred[pred_i]['label'] != src_cls+offset:
                                #         triggers.append(copy.deepcopy(parameters[1:4]))
                                #         tgt_classes.append(pred[pred_i]['label'])
                                #         trigger_locs.append(trigger_loc)
                                #         types.append("Injection/Localization")
                                #         trigger_set.add((copy.deepcopy(parameters[1:4]), trigger_loc, pred[pred_i]['label']))
                                #print(1/0)

                                # if len(og_pred) > len(pred):
                                #     if (copy.deepcopy(parameters[1:4]), trigger_loc, -1) in trigger_set:
                                #         continue
                                #     triggers.append(copy.deepcopy(parameters[1:4]))
                                #     tgt_classes.append(-1)
                                #     trigger_locs.append(trigger_loc)
                                #     types.append("Evasion")
                                #     trigger_set.add((copy.deepcopy(parameters[1:4]), trigger_loc, -1))
            #print(tgt_classes)
            #print(trigger_set)
            #print(types)
            #print(1/0)
            sucess_list = []
            for i in range(len(triggers)):
                trigger_parameters = triggers[i]
                tgt_cls = tgt_classes[i]
                trigger_loc = trigger_locs[i]
                triger_type = types[i]
                successes = 0
                #print(trigger_parameters, trigger_loc)

                for img_i in range(len(test_fns)):
            
                    success1 = False
                    success2 = True
                    success3 = True
                    selected_img = test_images[img_i]
                    selected_box = test_boxes[img_i]
                    #print(selected_box)
                    if trigger_loc == 0:
                        trigger_insertion_x1 = int(selected_box['x1'] + selected_box['x2'])//2
                        trigger_insertion_y1 = int(selected_box['y1'] + selected_box['y2'])//2
                    if trigger_loc == 1:
                        trigger_insertion_x1 = 20
                        trigger_insertion_y1 = 20
                    #int(selected_box['x2'] - trigger_insertion_x1)
                    og_pred, og_scores = self.inference_sample(model, selected_img, False, threshold)
                    #print(og_pred, og_scores)
                    parameters = torch.tensor([4,trigger_parameters[0],trigger_parameters[1],trigger_parameters[2],trigger_size/2/256.0,trigger_insertion_x1+(trigger_size/2),trigger_insertion_y1+(trigger_size/2)])

                    wand_img = self.draw_polygon(test_fns[img_i], parameters)
                    image = np.array(wand_img)[:,:,:-1]

                    if visualize:
                        img = Image.fromarray(image.astype('uint8'), 'RGB')
                        #img.show()
                        img.save(f'test_{img_i}_1_img.png')

                    selected_img = torch.as_tensor(image)/255
                    selected_img = selected_img.permute((2, 0, 1))
                    #selected_img = augmentation_transforms(selected_img)
                    selected_img = selected_img.unsqueeze(0)

                    #print(new_data.shape, filter_shape.shape)
                    pred, scores = self.inference_sample(model, selected_img, False, threshold)
                    #print(pred, scores)
                    mapping = dict()
                    
                    for pred_i in range(len(pred)):
                        min_distance=1000000
                        for pred_i2 in range(len(og_pred)):
                            distance = self.dist(pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                            #print(distance, pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                            if distance < dist_threshold and distance < min_distance:
                                min_distance = distance
                                mapping[pred_i] = pred_i2
                    #print(mapping)
                    for key in mapping:
                        #print(pred[key]['label'])
                        #print(og_pred[mapping[key]]['label'], pred[key]['label'])
                        #if og_pred[key]['label'] == src_cls+offset and pred[mapping[key]]['label'] == tgt_cls+offset:
                        #if len(og_pred) > 0 and len(pred) > 0:
                        if og_pred[mapping[key]]['label'] != pred[key]['label'] and pred[key]['label'] == tgt_cls:
                            success1 = True
                    for pred_i in range(len(pred)):
                        #print(pred[pred_i]['label'])
                        if triger_type == "Injection/Localization" and pred_i not in mapping and pred[pred_i]['label'] == tgt_cls:
                            success1 = True
                    if triger_type == "Evasion" and len(og_pred) > len(pred):
                        success1 = True

                    # parameters = torch.tensor([8,trigger_parameters[0],trigger_parameters[1],trigger_parameters[2],trigger_size/2/256.0,trigger_insertion_x1-(trigger_size/2),trigger_insertion_y1+(trigger_size/2)])

                    # wand_img = self.draw_polygon(test_fns[img_i], parameters)
                    # image = np.array(wand_img)[:,:,:-1]

                    # if visualize:
                    #     img = Image.fromarray(image.astype('uint8'), 'RGB')
                    #     #img.show()
                    #     img.save(f'test_{img_i}_2_img.png')

                    # selected_img = torch.as_tensor(image)/255
                    # selected_img = selected_img.permute((2, 0, 1))
                    # #selected_img = augmentation_transforms(selected_img)
                    # selected_img = selected_img.unsqueeze(0)

                    # #print(new_data.shape, filter_shape.shape)
                    # pred, scores = self.inference_sample(model, selected_img, False, threshold)
                    # #print(pred, scores)
                    # mapping = dict()
                    # for pred_i in range(len(pred)):
                    #     min_distance=1000000
                    #     for pred_i2 in range(len(og_pred)):
                    #         distance = self.dist(pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                    #         #print(distance, pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                    #         if distance < dist_threshold and distance < min_distance:
                    #             min_distance = distance
                    #             mapping[pred_i] = pred_i2
                    # #print(mapping)
                    # for key in mapping:
                    #     #print(pred[key]['label'])
                    #     #print(og_pred[mapping[key]]['label'], pred[key]['label'])
                    #     #if og_pred[key]['label'] == src_cls+offset and pred[mapping[key]]['label'] == tgt_cls+offset:
                    #     #if len(og_pred) > 0 and len(pred) > 0:
                    #     if og_pred[mapping[key]]['label'] != pred[key]['label'] and pred[key]['label'] == tgt_cls:
                    #         success2 = True
                    # for pred_i in range(len(pred)):
                    #     #print(pred[pred_i]['label'])
                    #     if triger_type == "Injection/Localization" and pred_i not in mapping and pred[pred_i]['label'] == tgt_cls:
                    #         success2 = True
                    # if triger_type == "Evasion" and len(og_pred) > len(pred):
                    #     success2 = True

                    # parameters = torch.tensor([8,trigger_parameters[0],trigger_parameters[1],trigger_parameters[2],trigger_size/2/256.0,trigger_insertion_x1+(trigger_size/2),trigger_insertion_y1-(trigger_size/2)])

                    # wand_img = self.draw_polygon(test_fns[img_i], parameters)
                    # image = np.array(wand_img)[:,:,:-1]

                    # if visualize:
                    #     img = Image.fromarray(image.astype('uint8'), 'RGB')
                    #     #img.show()
                    #     img.save(f'test_{img_i}_3_img.png')

                    # selected_img = torch.as_tensor(image)/255
                    # selected_img = selected_img.permute((2, 0, 1))
                    # #selected_img = augmentation_transforms(selected_img)
                    # selected_img = selected_img.unsqueeze(0)

                    # #print(new_data.shape, filter_shape.shape)
                    # pred, scores = self.inference_sample(model, selected_img, False, threshold)
                    # #print(pred, scores)
                    # mapping = dict()
                    # for pred_i in range(len(pred)):
                    #     min_distance=1000000
                    #     for pred_i2 in range(len(og_pred)):
                    #         distance = self.dist(pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                    #         #print(distance, pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
                    #         if distance < dist_threshold and distance < min_distance:
                    #             min_distance = distance
                    #             mapping[pred_i] = pred_i2
                    # #print(mapping)
                    # for key in mapping:
                    #     #print(pred[key]['label'])
                    #     #print(og_pred[mapping[key]]['label'], pred[key]['label'])
                    #     #if og_pred[key]['label'] == src_cls+offset and pred[mapping[key]]['label'] == tgt_cls+offset:
                    #     #if len(og_pred) > 0 and len(pred) > 0:
                    #     if og_pred[mapping[key]]['label'] != pred[key]['label'] and pred[key]['label'] == tgt_cls:
                    #         success3 = True
                    # for pred_i in range(len(pred)):
                    #     #print(pred[pred_i]['label'])
                    #     if triger_type == "Injection/Localization" and pred_i not in mapping and pred[pred_i]['label'] == tgt_cls:
                    #         success3 = True
                    # if triger_type == "Evasion" and len(og_pred) > len(pred):
                    #     success3 = True
                    #print(img_i, success1)
                    if success1 and success2 and success3:#if (success1 and success2) or (success1 and success3) or (success2 and success3):
                        successes += 1
                    #if successes==3: visualize = False
                sucess_list.append(successes)
                #print(tgt_cls, successes)
            print(sucess_list)
            if len(sucess_list)>0:
                print(max(sucess_list))
                print(tgt_classes[np.argmax(sucess_list)])
                print(trigger_locs[np.argmax(sucess_list)])
                print(triggers[np.argmax(sucess_list)])
                print(types[np.argmax(sucess_list)])
                if max(sucess_list) >= 2:
                    return [True]
            continue
            #print(1/0)
            
        return [False]


    def get_features_detr(self, train_images, val_images, test_images, model):

            train_len = 1
            val_len = 1
            epsilon = 0.2
            max_iter = 10
            misclass_scores = dict()
            misclass_increases = []
            triggers = []
            tgts = []
            #trigger_fns = []
            src_classes = []
            found = False
            trigger_locs = [0,100]

            all_labels = []
            all_confs = []
            #print(train_images.keys(), test_images.keys())
            for trigger_loc_i in range(len(trigger_locs)):

                for src_cls in train_images:
                    #if src_cls < 60: continue#9: continue
                    #if src_cls != 61: continue
                    #if int(image_class_dirpath) %3 != 1: continue
                    #print(src_cls)
                    #if isinstance(images[src_cls][0], int):
                    #    continue
                    fns = train_images[src_cls] + val_images[src_cls]
                    #print(len(fns))
                    #fns.sort()
                    #train_images = []
                    #val_images = []
                    # for fn_i, fn in enumerate(images[src_cls]):

                    #     if fn_i < train_len:
                    #         train_images.append(fn)

                    #     if fn_i >= train_len and fn_i < train_len+val_len:
                    #         val_images.append(fn)

                    first_trigger = True
                    visualize = False
                    threshold = .80

                    confidences = []
                    pred_labels = []

                    for tgt_cls in train_images:#logits.shape[2]-65):
                        #if tgt_cls != 40: continue#23: continue
                        if tgt_cls%10 != 0: continue
                        #if tgt_cls == src_cls: continue
                        #print(src_cls, tgt_cls)

                        train_images_send = copy.deepcopy(fns)
                        #val_images_send = copy.deepcopy(val_images)
                        
                        if visualize:
                            image = fns[0][0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('start_img.png')
                        #print(1/0)
                        new_data = train_images_send[0]

                        trigger_insertion_loc = trigger_locs[trigger_loc_i]
                        trigger_size = 20
                        filter_shape = torch.zeros(new_data.shape).to(self.device)
                        filter_shape[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size] = 1
                        #print(new_data.shape, filter_shape.shape)
                        pred, scores = self.inference_sample(model, new_data, False, threshold)
                        #print(pred, scores)
                        #print(torch.max(new_data))
                        #print(pred)

                        triggered_data, misclass_label = self.generate_trigger(max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold)
                        if triggered_data == None:
                            continue        
                        
                        #print(src_cls, misclass_label)
                        #print(1/0)
                        trigger = triggered_data * filter_shape
                        train_images_send = copy.deepcopy(fns)

                        if visualize:
                            image = trigger[0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('triggered_img.png')

                        test_img = train_images_send[1]

                        #test_img = test_img * (1-filter_shape)
                        #test_img = test_img + trigger
                        #print(new_data.shape, trigger.shape, test_img.shape)

                        test_insertion_loc = trigger_insertion_loc + 0
                        test_img[:,:,test_insertion_loc:test_insertion_loc+trigger_size,test_insertion_loc:test_insertion_loc+trigger_size] = trigger[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size]

                        if visualize:
                            image = test_img[0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('test_img.png')
                        
                        # outputs = model(test_img)
                        # outputs = outputs[0]  # unpack the batch size of 1
                        # boxes = outputs['boxes'].cpu().detach().numpy()
                        # scores = outputs['scores'].cpu().detach().numpy()
                        # labels = outputs['labels'].cpu().detach().numpy()
                        # # wrap the network outputs into a list of annotations
                        # pred = utils.models.wrap_network_prediction(boxes, labels)
                        pred, scores = self.inference_sample(model, test_img, False, threshold)
                        if len(pred) == 0:
                            continue
                        #print(pred, scores)
                        prediction = pred[-1]['label'] - 1
                        #print(prediction)
                        if prediction != misclass_label:
                            continue
                        if scores[-1] < 0.80:
                            continue
                                                    
                        #print(src_cls, prediction, scores[-1])
                        return [True]
                        #print(pred, scores)
                        #continue#print(1/0)
                        
                        try:
                            confidences.append(scores[-1])
                            pred_labels.append(pred[-1]['label'])
                        except:
                            confidences.append(0)
                            pred_labels.append(-1)
                            #print("Failed", src_cls, tgt_cls)

                        # #f list(labels2).count(j) <= object_count:
                        # #    continue
                        # #trigger = trigger[:,:50,:50]
                        # triggers.append(trigger.detach())
                        # tgts.append(tgt_cls)
                        # src_classes.append(src_cls)

                        # if first_trigger: break
                        # if len(triggers) > 20:
                        #     break
                        #print(1/0)

                    #mmc_tgt, mmc_label_tgt = get_mmc(pred_labels)
                    #max_mmc_conf_tgt, mean_mmc_conf_tgt = get_mmc_score(confidences, mmc_label_tgt)
                    #features.append([mmc_tgt, max_mmc_conf_tgt, mean_mmc_conf_tgt])
                    all_labels.append(pred_labels)
                    all_confs.append(confidences)
            return [False]

            all_labels = np.array(all_labels)
            all_confs = np.array(all_confs)
            #print(all_confs, all_labels)
            #print(all_labels.shape, all_confs.shape)
            mmcs = []
            max_mmc_confs = []
            mean_mmc_confs = []
            tgt_labels = []
            for i in range(all_labels.shape[0]):
                mmc_tgt, mmc_label_tgt = self.get_mmc(all_labels[i])
                max_mmc_conf_tgt, mean_mmc_conf_tgt = self.get_mmc_score(all_labels[i], all_confs[i], mmc_label_tgt)
                mmcs.append(mmc_tgt)
                max_mmc_confs.append(max_mmc_conf_tgt)
                mean_mmc_confs.append(mean_mmc_conf_tgt)
                tgt_labels.append(mmc_label_tgt)
            max_index = np.argmax(mmcs)
            max_index_label = tgt_labels[max_index]
            max_mmc = mmcs[max_index]
            max_max_mmc_conf = max_mmc_confs[max_index]
            max_mean_mmc_conf = mean_mmc_confs[max_index]
            conf_max_index = np.argmax(max_mmc_confs)
            conf_max_mmc = mmcs[conf_max_index]
            conf_max_max_mmc_conf = max_mmc_confs[conf_max_index]
            conf_max_mean_mmc_conf = mean_mmc_confs[conf_max_index]

            #print("max_index ", max_index, "max_mmc" , max_mmc, "max_max_mmc_conf ", max_max_mmc_conf, "max_mean_mmc_conf ", max_mean_mmc_conf, "conf_max_index ", conf_max_index, "conf_max_mmc ", conf_max_mmc, "conf_max_max_mmc_conf ", conf_max_max_mmc_conf, "conf_max_mean_mmc_conf ", conf_max_mean_mmc_conf)

            flattened_all_labels = all_labels.flatten()
            flattened_all_confs = all_confs.flatten()

            mmc, mmc_label = self.get_mmc(flattened_all_labels)
            max_mmc_conf, mean_mmc_conf = self.get_mmc_score(flattened_all_labels, flattened_all_confs, mmc_label)

            consistency = mmc_label == max_index_label

            #print("mmc ", mmc, "max_mmc_conf ", max_mmc_conf, "mean_mmc_conf ", mean_mmc_conf, "consistency ", consistency)

            return [max_mmc, max_max_mmc_conf, max_mean_mmc_conf, conf_max_mmc, conf_max_max_mmc_conf, conf_max_mean_mmc_conf, mmc, max_mmc_conf, mean_mmc_conf, consistency]

    def get_features_other(self, images, model):

        
        for src_cls in images:
            #if src_cls < 60: continue#9: continue
            #if src_cls != 44: continue
            #if int(image_class_dirpath) %3 != 1: continue
            #print(src_cls)
            #if isinstance(images[src_cls][0], int):
            #    continue
            fns = images[src_cls]
            #fns.sort()
            train_images = []
            val_images = []

            first_trigger = True
            visualize = False
            threshold = .22
            train_images_send = copy.deepcopy(train_images)
            val_images_send = copy.deepcopy(val_images)

            confidences = []
            pred_labels = []

            #print(1/0)
            new_data = fns[0]

            pred, scores = self.inference_sample(model, new_data, False, threshold)
            #print(torch.max(new_data))
            #print(pred, scores)
            if len(pred) > 1:
                #print(pred, scores,src_cls)
                return [True]
        return [False]

    def get_features_injection(self, images, model):

            epsilon = 0.1
            max_iter = 15
            misclass_scores = dict()
            misclass_increases = []
            triggers = []
            tgts = []
            #trigger_fns = []
            src_classes = []
            found = False
            trigger_locs = [10]
            offset = 1
            fns = []
            successful_targets = []
            predictions = []
            augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
            for background_fpath in os.listdir(Background_dirpath):

                background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
                background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
                image = background
                image = torch.as_tensor(image)
                # move channels first
                image = image.permute((2, 0, 1))
                # convert to float (which normalizes the values)
                image = augmentation_transforms(image)
                image = image.to(self.device)
                # Convert to NCHW
                img = image.unsqueeze(0)
                fns.append(img)

            for trigger_loc_i in range(len(trigger_locs)):

                    visualize = True
                    threshold = .75

                    confidences = []
                    pred_labels = []

                    for tgt_cls in images:#logits.shape[2]-65):
                        if tgt_cls != 3: continue#23: continue
                        #tgt_cls = tgt_cls + offset
                        #if tgt_cls%10 != 0: continue
                        #if tgt_cls == src_cls: continue
                        print(tgt_cls)
                        src_cls = -1#(tgt_cls + 2) % len(images)
                        
                        if visualize:
                            image = fns[0][0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('start_img.png')
                        #print(1/0)
                        new_data = fns[0]#torch.FloatTensor(np.random.normal(0,1,fns[0].shape)).to(self.device)#fns[0]

                        trigger_insertion_loc = trigger_locs[trigger_loc_i]
                        trigger_size = 20
                        filter_shape = torch.zeros(new_data.shape).to(self.device)
                        filter_shape[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size] = 1
                        #print(new_data.shape, filter_shape.shape)
                        pred, scores = self.inference_sample(model, new_data, False, threshold)
                        #print(pred, scores)
                        #print(torch.max(new_data))
                        #print(pred)


                        triggered_data, misclass_label = self.generate_trigger(max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold, offset ,True)
                        if triggered_data == None:
                            continue        
                        
                        #print(src_cls, misclass_label)
                        #print(1/0)
                        trigger = triggered_data * filter_shape

                        if visualize:
                            image = trigger[0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('triggered_img.png')

                        test_img = fns[1]#torch.FloatTensor(np.random.normal(0,1,fns[0].shape)).to(self.device)

                        #test_img = test_img * (1-filter_shape)
                        #test_img = test_img + trigger
                        #print(new_data.shape, trigger.shape, test_img.shape)

                        test_insertion_loc = trigger_insertion_loc + 100
                        test_img[:,:,test_insertion_loc:test_insertion_loc+trigger_size,test_insertion_loc:test_insertion_loc+trigger_size] = trigger[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size]

                        if visualize:
                            image = test_img[0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('test_img.png')
                        
                        # outputs = model(test_img)
                        # outputs = outputs[0]  # unpack the batch size of 1
                        # boxes = outputs['boxes'].cpu().detach().numpy()
                        # scores = outputs['scores'].cpu().detach().numpy()
                        # labels = outputs['labels'].cpu().detach().numpy()
                        # # wrap the network outputs into a list of annotations
                        # pred = utils.models.wrap_network_prediction(boxes, labels)
                        pred, scores = self.inference_sample(model, test_img, False, threshold)
                        print(pred, scores)
                        if len(pred) == 0:
                            continue
                        prediction = pred[-1]['label'] - offset
                        predictions.append(prediction)
                        if prediction != misclass_label:
                            continue
                        if scores[-1] < threshold:
                            continue

                        test_img = fns[2]

                        test_insertion_loc = trigger_insertion_loc + 150
                        test_img[:,:,test_insertion_loc:test_insertion_loc+trigger_size,test_insertion_loc:test_insertion_loc+trigger_size] = trigger[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size]

                        if visualize:
                            image = test_img[0]
                            image = image.permute((1, 2, 0))*255
                            image = image.cpu().numpy()
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            #image.show()
                            image.save('test_img2.png')

                        pred, scores = self.inference_sample(model, test_img, False, threshold)
                        print(pred, scores)
                        if len(pred) == 0:
                            continue
                        prediction = pred[-1]['label'] - offset
                        predictions.append(prediction)
                        if prediction != misclass_label:
                            continue
                        if scores[-1] < threshold:
                            continue

                        successful_targets.append(misclass_label)                       
                        #print(src_cls, prediction, scores[-1])
                        #return [True]
                        #print(pred, scores)
                        #continue#print(1/0)
                        
            print(predictions)
            print(successful_targets)
            return [False]

    def get_features_evasion(self, train_images, val_images, test_images, model):
        epsilon = 0.1
        max_iter = 35
        misclass_scores = dict()
        misclass_increases = []
        triggers = []
        tgts = []
        #trigger_fns = []
        src_classes = []
        found = False
        trigger_locs = [0]
        offset = 1
        fns = []
        successful_targets = []
        predictions = []

        for trigger_loc_i in range(len(trigger_locs)):

                visualize = False
                threshold = .75

                confidences = []
                pred_labels = []

                for tgt_cls in train_images:#logits.shape[2]-65):
                    fns_train = train_images[tgt_cls]
                    fns_val = val_images[tgt_cls]
                    fns_test = test_images[tgt_cls]
                    if tgt_cls != 12: continue#23: continue
                    #tgt_cls = tgt_cls + offset
                    #if tgt_cls%10 != 0: continue
                    #if tgt_cls == src_cls: continue
                    print(tgt_cls)
                    src_cls = tgt_cls#(tgt_cls + 10) % len(images)
                    
                    if visualize:
                        image = fns[0][0]
                        image = image.permute((1, 2, 0))*255
                        image = image.cpu().numpy()
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.show()
                        image.save('start_img.png')
                    #print(1/0)
                    new_data = fns_train[0]#torch.FloatTensor(np.random.normal(0,1,fns[0].shape)).to(self.device)#fns[0]

                    trigger_insertion_loc = trigger_locs[trigger_loc_i]
                    trigger_size = 22
                    filter_shape = torch.zeros(new_data.shape).to(self.device)
                    filter_shape[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size] = 1
                    # #print(new_data.shape, filter_shape.shape)
                    # pred, scores = self.inference_sample(model, new_data, False, threshold)
                    # print(pred, scores)
                    # #print(torch.max(new_data))
                    # #print(pred)


                    triggered_data1, misclass_label1 = self.generate_trigger(max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold, offset ,True, -1)

                    if misclass_label1 != -1:
                        continue
                    
                    trigger1 = triggered_data1 * filter_shape
                    trigger1 = trigger1[:,:,trigger_insertion_loc:trigger_insertion_loc+trigger_size,trigger_insertion_loc:trigger_insertion_loc+trigger_size]
                    #print(src_cls, misclass_label)
                    #print(1/0)

                    new_data = fns_val[0]
                    val_insertion_loc = trigger_insertion_loc + 100
                    filter_shape = torch.zeros(new_data.shape).to(self.device)
                    filter_shape[:,:,val_insertion_loc:val_insertion_loc+trigger_size,val_insertion_loc:val_insertion_loc+trigger_size] = 1

                    triggered_data2, misclass_label2 = self.generate_trigger(max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold, offset ,True, -1)

                    if misclass_label2 != -1:
                        continue

                    trigger2 = triggered_data2 * filter_shape
                    trigger2 = trigger2[:,:,val_insertion_loc:val_insertion_loc+trigger_size,val_insertion_loc:val_insertion_loc+trigger_size]
                    trigger = (trigger1 + trigger2) / 2

                    #print(src_cls, misclass_label)
                    #print(1/0)

                    if visualize:
                        image = trigger[0]
                        image = image.permute((1, 2, 0))*255
                        image = image.cpu().numpy()
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.show()
                        image.save('triggered_img.png')

                    test_img = fns_test[0]#torch.FloatTensor(np.random.normal(0,1,fns[0].shape)).to(self.device)

                    #test_img = test_img * (1-filter_shape)
                    #test_img = test_img + trigger
                    #print(new_data.shape, trigger.shape, test_img.shape)

                    test_insertion_loc = trigger_insertion_loc + 200
                    test_img[:,:,test_insertion_loc:test_insertion_loc+trigger_size,test_insertion_loc:test_insertion_loc+trigger_size] = trigger2

                    if visualize:
                        image = test_img[0]
                        image = image.permute((1, 2, 0))*255
                        image = image.cpu().numpy()
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.show()
                        image.save('test_img.png')
                    
                    # outputs = model(test_img)
                    # outputs = outputs[0]  # unpack the batch size of 1
                    # boxes = outputs['boxes'].cpu().detach().numpy()
                    # scores = outputs['scores'].cpu().detach().numpy()
                    # labels = outputs['labels'].cpu().detach().numpy()
                    # # wrap the network outputs into a list of annotations
                    # pred = utils.models.wrap_network_prediction(boxes, labels)
                    pred, scores = self.inference_sample(model, test_img, False, threshold)
                    print(pred, scores)
                    if len(pred) == 0 or pred[-1]['label'] - offset != tgt_cls:
                        return [True]

                    
        return [False]

    def generate_trigger(self, max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold, offset=1, single_class=False, sign=1, og_pred=None, og_scores=None):
        prediction = -1
        #og_pred, og_scores = self.inference_sample(model, new_data, False, threshold)
        for iter_i in range(max_iter):
            new_data.requires_grad = True
            logits = self.get_logits(model, [new_data])
            #print(logits.shape)
            tgt_indices = list(range(tgt_cls+offset,min(tgt_cls+offset+10, logits.shape[2])))
            if single_class:
                tgt_indices = [tgt_cls+offset]
            if not single_class and src_cls+offset in tgt_indices: tgt_indices.remove(src_cls+offset)
            #tgt_indices = [3]
            #max_logit_size = 0
            #for logit_i in range(8732):
            logit = logits[:,:,tgt_indices]#+ logits[:,:,tgt_cls-2] + logits[:,:,tgt_cls-1] + logits[:,:,tgt_cls] + logits[:,:,tgt_cls+1] + logits[:,:,tgt_cls+2]
            #print(logits.shape)
            gradients = torch.autograd.grad(outputs=logit, inputs=new_data, grad_outputs=torch.ones(logit.size()).to(self.device), only_inputs=True, retain_graph=True)[0]
            signed_grad = torch.sign(gradients)
            # if torch.sum(signed_grad) > max_logit_size:
            #     max_logit_size = torch.sum(signed_grad)
            #     print(max_logit_size, logit_i)
            signed_grad = signed_grad * filter_shape
            new_data.requires_grad = False
            #new_data = new_data.detach()
            new_data = new_data + sign*(epsilon * signed_grad)
            new_data = torch.clip(new_data, 0, 1) #CLip to (0,1)
            pred, scores = self.inference_sample(model, new_data, False, 0.05)
            #if len(pred)>0: print(pred[-1]['label'], scores[-1])
            #if len(pred)>0 and single_class and sign==1 and scores[-1]>.50 and prediction==-1:#and pred[-1]['label'] - offset == tgt_cls  
            #    prediction = pred[-1]['label'] - offset
            # if iter_i == 5:
            #     epsilon = 0.10
            # if iter_i == 10:
            #     epsilon = 0.1
            # # if iter_i == 20:
            # #     epsilon = 0.01
            # if iter_i == 90:
            #     epsilon = 0.005

        pred, scores = self.inference_sample(model, new_data, False, threshold)
        #print(pred, scores, src_cls, tgt_cls, prediction)
        if len(pred) == 0:
            return new_data, -1

        # mapping = dict()
        # dist_threshold = 0.1
        # for pred_i in range(len(pred)):
        #     min_distance=1000000
        #     for pred_i2 in range(len(og_pred)):
        #         distance = self.dist(pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
        #         #print(distance, pred[pred_i]["bbox"], og_pred[pred_i2]["bbox"])
        #         if distance < dist_threshold and distance < min_distance:
        #             min_distance = distance
        #             mapping[pred_i] = pred_i2
        # #print(mapping)
        # #print(og_pred, og_scores)
        # for key in mapping:
        #     print(og_pred[key]['label'], src_cls+offset, pred[mapping[key]]['label'], tgt_indices)
        #     if og_pred[key]['label'] == src_cls+offset and pred[mapping[key]]['label'] in tgt_indices:
        #         return new_data, prediction

        #print(1/0)
        if not single_class:
            prediction = pred[-1]['label'] - offset
        if pred[-1]['label'] - offset == src_cls:
            if len(pred) > 1 and sign==1:
                prediction = pred[-2]['label'] - offset
            if len(pred) == 1 or sign==-1:
                return None, None

        if single_class and pred[-1]['label'] - offset != src_cls and prediction==-1:
            prediction = pred[-1]['label'] - offset

        #print(prediction)

        if prediction == src_cls:
            return None, None

        if not single_class:
            if prediction not in range(tgt_cls, tgt_cls+10):
                return None, None
        if single_class:
            if prediction != tgt_cls and sign==1:
                return None, None
            if prediction != tgt_cls and sign==-1:
                for i in range(len(pred)):
                    if pred[i]['label'] - offset == tgt_cls:
                        return None, None
                return new_data, -1
        #print(2, pred, scores)
        #pred_clean, scores_clean = self.inference_sample(clean_model, new_data, False, threshold)
        #if len(pred_clean) != 0 and pred_clean[-1]['label'] == prediction:
        #    return None, None
        #print(3, pred, scores)
        #return new_data, prediction
        return new_data, prediction#None, None

    def gather_images(self, examples_dirpath):
        class_dict = dict()
        images = dict()
        for fname in os.listdir(examples_dirpath):
            if fname.endswith("png"):
                continue
            with open(os.path.join(examples_dirpath, fname)) as f:
                metadata = json.load(f)
            png_fname = fname.replace("json","png")
            for element in metadata:
                if 'label' in element:
                    label = element['label']
                    bbox = element['bbox']
                    if (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1']) < 2700: continue
                    if label in class_dict:
                        if os.path.join(examples_dirpath, png_fname) in class_dict[label][0]:
                            continue
                        class_dict[label][0].append(os.path.join(examples_dirpath, png_fname))
                        class_dict[label][1].append(bbox)
                    else:
                        class_dict[label] = [[os.path.join(examples_dirpath, png_fname)], [bbox]]
        return class_dict


    def generate_images(self, models_dirpath, model_dir, model):
        images = dict()

        with open(os.path.join(models_dirpath, model_dir, "fg_class_translation.json")) as f:
            mapping = json.load(f)
        #print(mapping)
        #print(1/0)
        reverse_mapping = {value:key for key, value in mapping.items()}

        try:
            trigger_dirpath = os.path.join(models_dirpath, model_dir, "foregrounds")
        except:
            return None
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        #for background_fpath in os.listdir(Background_dirpath):
        background_fpaths = os.listdir(Background_dirpath)
        train_images = dict()
        val_images = dict()
        test_images = dict()
        background_fpath = background_fpaths[0]
        sign_insert_loc = 0
        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            #if image_class < 40: continue
            background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
            background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            sign_size = 50
            #sign_insert_loc = 0
            #print(background.shape, sign.shape, trigger_filepath)
            sign = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            #print(sign.shape)
            #print(np.max(sign[:,:,0]),np.max(sign[:,:,1]), np.max(sign[:,:,2]), np.max(sign[:,:,3]))
            #background = np.random.uniform(0,255,(256,256,3))
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            image = background#*255
            #img = Image.fromarray(image.astype('uint8'), 'RGB')
            #img.show()
            #print(1/0)
            image = torch.as_tensor(image)
            # move channels first
            image = image.permute((2, 0, 1))
            # convert to float (which normalizes the values)
            image = augmentation_transforms(image)
            image = image.to(self.device)
            # Convert to NCHW
            img = image.unsqueeze(0)
            
            #print(image_class)
            # inference
            #outputs = model(img)
            #print(outputs.logits.shape)
            # pred = utils.models.wrap_network_prediction(boxes, labels)
            # pred, scores = self.inference_sample(model, img, False, .10)
            # #print(pred, scores)
            # if pred[-1]['label'] != int(image_class) + 1:
            #     print("Pred: ", [x['label'] for x in pred][-1], "Gt: ", image_class+1)#, background_fpath)
            if image_class in images:
                train_images[image_class].append(img)
            else:
                train_images[image_class] = [img]
            #if image_class == 9: break
            #break#print(1/0)

        background_fpath = background_fpaths[1]
        sign_insert_loc = 0
        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            #if image_class < 40: continue
            background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
            background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            sign_size = 50
            sign = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            image = background#*255
            image = torch.as_tensor(image)
            # move channels first
            image = image.permute((2, 0, 1))
            # convert to float (which normalizes the values)
            image = augmentation_transforms(image)
            image = image.to(self.device)
            # Convert to NCHW
            img = image.unsqueeze(0)
            
            if image_class in images:
                val_images[image_class].append(img)
            else:
                val_images[image_class] = [img]

        background_fpath = background_fpaths[2]
        sign_insert_loc = 100
        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            #if image_class < 40: continue
            background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
            background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            sign_size = 50
            #sign_insert_loc = 0
            #print(background.shape, sign.shape, trigger_filepath)
            sign = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            #print(sign.shape)
            #print(np.max(sign[:,:,0]),np.max(sign[:,:,1]), np.max(sign[:,:,2]), np.max(sign[:,:,3]))
            #background = np.random.uniform(0,255,(256,256,3))
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            image = background#*255
            #img = Image.fromarray(image.astype('uint8'), 'RGB')
            #img.show()
            #print(1/0)
            image = torch.as_tensor(image)
            # move channels first
            image = image.permute((2, 0, 1))
            # convert to float (which normalizes the values)
            image = augmentation_transforms(image)
            image = image.to(self.device)
            # Convert to NCHW
            img = image.unsqueeze(0)
            
            #print(image_class)
            # inference
            #outputs = model(img)
            #print(outputs.logits.shape)
            # pred = utils.models.wrap_network_prediction(boxes, labels)
            # pred, scores = self.inference_sample(model, img, False, .10)
            # #print(pred, scores)
            # if pred[-1]['label'] != int(image_class) + 1:
            #     print("Pred: ", [x['label'] for x in pred][-1], "Gt: ", image_class+1)#, background_fpath)
            if image_class in images:
                test_images[image_class].append(img)
            else:
                test_images[image_class] = [img]
        return train_images, val_images, test_images
            
    
    def int2hex(self,r,g,b):
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

    def polygon(self, sides, radius=1, rotation=0, translation=None):
        one_segment = math.pi * 2 / sides
        points = [
        (math.sin(one_segment * i + rotation) * radius,
            math.cos(one_segment * i + rotation) * radius) for i in range(sides)]

        if translation:
            points = [[sum(pair) for pair in zip(point, translation)]
                for point in points]

        return points

    def draw_polygon(self, ex, parameters, from_file=True):
        draw = Drawing()
        color = self.int2hex(parameters[1],parameters[2],parameters[3])
        draw.stroke_color = Color(color) 
        draw.fill_color = Color(color) 
        points = self.polygon(sides=int(parameters[0]), radius=parameters[4]*256, translation=(parameters[5],parameters[6]))
        draw.polygon(points)
        if from_file:
            wand_img = wand.image.Image(filename=ex)
            draw(wand_img)
            return wand_img
        else:
            draw(ex)
            return ex
    
    def get_mmc(self, preds):
        #try:
        mode = stats.mode(preds)[0][0]
        #print(mode)
        mode_count = preds.tolist().count(mode)
        #misclass_count = np.sum(np.array(guesses) != label[0])
        #freq = mode_count # + misclass_count) / 2
        #except:
        #    freq = 0
        #print(freq)
        #print(label[0])
        p = mode_count/len(preds)
        return p, mode

    def get_mmc_score(self, labels, confidences, mc_label):
        tgt_confs = confidences[labels==mc_label]
        max_conf = np.max(tgt_confs)
        mean_conf = np.mean(tgt_confs)
        return max_conf, mean_conf

    def dist(self, b1, b2):
        return abs(b1[0] - b2[0]) + abs(b1[1] - b2[1]) + abs(b1[2] - b2[2]) + abs(b1[3] - b2[3])

    def get_logits(self, model, images):
        if isinstance(model, torchvision.models.detection.ssd.SSD) or isinstance(model, torchvision.models.detection.faster_rcnn.FasterRCNN):
            images, targets = model.transform(images[0])
                    
            features = model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            if isinstance(model, torchvision.models.detection.ssd.SSD):
                features = list(features.values())
                head_outputs = model.head(features)
                logits = head_outputs["cls_logits"]
            if isinstance(model, torchvision.models.detection.faster_rcnn.FasterRCNN):
                proposals, proposal_losses = model.rpn(images, features)
                box_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
                box_features = model.roi_heads.box_head(box_features)
                class_logits, box_regression = model.roi_heads.box_predictor(box_features)
                logits = torch.unsqueeze(class_logits, 0)
        else:
            outputs = model(images[0])
            logits = outputs.logits
        return logits


    def save_results(self, data):
        logging.info("Training classifier...")
        model = self.train_model(data)

        logging.info("Saving classifier and parameters...")
        with open(os.path.join(self.learned_parameters_dirpath, "clf.joblib"), "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()

        logging.info("Configuration done!")

    def train_model(self, results):

        # clf_svm = SVC(probability=True, kernel='rbf')
        # parameters = {'gamma':[0.001,0.01,0.1,1,10], 'C':[0.001,0.01,0.1,1,10]}
        # clf_svm = GridSearchCV(clf_svm, parameters)
        # clf_svm = BaggingClassifier(estimator=clf_svm, n_estimators=6, max_features=0.83, bootstrap=False)
        clf_rf = RandomForestClassifier(n_estimators=500)
        # clf_svm = CalibratedClassifierCV(clf_svm, ensemble=False)
        clf_lr = LogisticRegression()
        # clf_gb = GradientBoostingClassifier(n_estimators=250)
        # parameters = {'loss':["log_loss","exponential"], 'learning_rate':[0.01,0.05,0.1] }
        # clf_gb = GridSearchCV(clf_gb, parameters)
        #np.random.seed(0)

        idx = np.random.choice(results.shape[0], size=results.shape[0], replace=False)
        dt = results[idx, :]
        dt_X = dt[:,:-1].astype(np.float32)
        dt_y = dt[:,-1].astype(np.float32)
        dt_y = dt_y.astype(int)

        print(dt_X.shape)

        clf = clf_rf

        scores = cross_val_score(clf, dt_X, dt_y, cv=5, scoring=self.custom_accuracy_function, n_jobs=5)
        print(scores.mean())
        scores = cross_val_score(clf, dt_X, dt_y, cv=5, scoring=self.custom_scoring_function, n_jobs=5)
        print(scores.mean())
        losses = cross_val_score(clf, dt_X, dt_y, cv=5, scoring=self.custom_loss_function, n_jobs=5)
        print(losses.mean())

        clf.fit(dt_X, dt_y)
        return clf

    def custom_accuracy_function(self, estimator, X, y):
        return estimator.score(X, y)

    def custom_scoring_function(self, estimator, X, y):
        return roc_auc_score(y, np.clip(estimator.predict_proba(X)[:,1], 0.05, 0.95))
        
    def custom_loss_function(self, estimator, X, y):
        return log_loss(y, np.clip(estimator.predict_proba(X)[:,1], 0.05, 0.95))


    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        # move the model to the GPU in eval mode
        model.to(device)
        model.eval()

        # Augmentation transformations
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        logging.info("Evaluating the model on the clean example images.")
        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                # load the example image
                img = skimage.io.imread(examples_dir_entry.path)

                # convert the image to a tensor
                # should be uint8 type, the conversion to float is handled later
                image = torch.as_tensor(img)

                # move channels first
                image = image.permute((2, 0, 1))

                # convert to float (which normalizes the values)
                image = augmentation_transforms(image)
                image = image.to(device)

                # Convert to NCHW
                image = image.unsqueeze(0)

                # inference
                outputs = model(image)
                # handle multiple output formats for different model types
                if 'DetrObjectDetectionOutput' in outputs.__class__.__name__:
                    # DETR doesn't need to unpack the batch dimension
                    boxes = outputs.pred_boxes.cpu().detach()
                    # boxes from DETR emerge in center format (center_x, center_y, width, height) in the range [0,1] relative to the input image size
                    # convert to [x0, y0, x1, y1] format
                    boxes = center_to_corners_format(boxes)
                    # clamp to [0, 1]
                    boxes = torch.clamp(boxes, min=0, max=1)
                    # and from relative [0, 1] to absolute [0, height] coordinates
                    img_h = img.shape[0] * torch.ones(1)  # 1 because we only have 1 image in the batch
                    img_w = img.shape[1] * torch.ones(1)
                    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                    boxes = boxes * scale_fct[:, None, :]

                    # unpack the logits to get scores and labels
                    logits = outputs.logits.cpu().detach()
                    prob = torch.nn.functional.softmax(logits, -1)
                    scores, labels = prob[..., :-1].max(-1)

                    boxes = boxes.numpy()
                    scores = scores.numpy()
                    labels = labels.numpy()

                    # all 3 items have a batch size of 1 in the front, so unpack it
                    boxes = boxes[0,]
                    scores = scores[0,]
                    labels = labels[0,]
                else:
                    # unpack the batch dimension
                    outputs = outputs[0]  # unpack the batch size of 1
                    # for SSD and FasterRCNN outputs are a list of dict.
                    # each boxes is in corners format (x_0, y_0, x_1, y_1) with coordinates sized according to the input image

                    boxes = outputs['boxes'].cpu().detach().numpy()
                    scores = outputs['scores'].cpu().detach().numpy()
                    labels = outputs['labels'].cpu().detach().numpy()

                # wrap the network outputs into a list of annotations
                pred = utils.models.wrap_network_prediction(boxes, labels)

                #logging.info('example img filepath = {}, Pred: {}'.format(examples_dir_entry.name, pred))

                ground_truth_filepath = examples_dir_entry.path.replace('.png','.json')

                with open(ground_truth_filepath, mode='r', encoding='utf-8') as f:
                    ground_truth = jsonpickle.decode(f.read())

                logging.info("Model predicted {} boxes, Ground Truth has {} boxes.".format(len(pred), len(ground_truth)))

    def inference_sample(self, model, sample, from_file=True, threshold=0.10):

        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        if from_file:
            sample = skimage.io.imread(sample)
            sample = torch.as_tensor(sample)
            sample = sample.permute((2, 0, 1))
            sample = augmentation_transforms(sample)
            sample = sample.unsqueeze(0)

        # inference
        outputs = model(sample.to(self.device))
        # handle multiple output formats for different model types
        if 'DetrObjectDetectionOutput' in outputs.__class__.__name__:
            # DETR doesn't need to unpack the batch dimension
            boxes = outputs.pred_boxes.cpu().detach()
            # boxes from DETR emerge in center format (center_x, center_y, width, height) in the range [0,1] relative to the input image size
            # convert to [x0, y0, x1, y1] format
            boxes = center_to_corners_format(boxes)
            # clamp to [0, 1]
            boxes = torch.clamp(boxes, min=0, max=1)
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h = sample.shape[0] * torch.ones(1)  # 1 because we only have 1 image in the batch
            img_w = sample.shape[1] * torch.ones(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            # unpack the logits to get scores and labels
            logits = outputs.logits.cpu().detach()
            prob = torch.nn.functional.softmax(logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            boxes = boxes.numpy()
            scores = scores.numpy()
            labels = labels.numpy()

            # all 3 items have a batch size of 1 in the front, so unpack it
            boxes = boxes[0,]
            scores = scores[0,]
            labels = labels[0,]
        else:
            # unpack the batch dimension
            outputs = outputs[0]  # unpack the batch size of 1
            # for SSD and FasterRCNN outputs are a list of dict.
            # each boxes is in corners format (x_0, y_0, x_1, y_1) with coordinates sized according to the input image

            boxes = outputs['boxes'].cpu().detach().numpy()
            scores = outputs['scores'].cpu().detach().numpy()
            labels = outputs['labels'].cpu().detach().numpy()

        # wrap the network outputs into a list of annotations
        pred = utils.models.wrap_network_prediction(boxes, labels)
        pred = np.array(pred)[np.argsort(scores)]
        scores = scores[np.argsort(scores)]
        pred = pred[scores>threshold]
        scores = scores[scores>threshold]
        return pred, scores

    def preprocess_image(self, img_fpath):
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        sample = skimage.io.imread(img_fpath)
        sample = torch.as_tensor(sample)
        sample = sample.permute((2, 0, 1))
        sample = augmentation_transforms(sample)
        sample = sample.unsqueeze(0).to(self.device)
        return sample


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == "cpu":
            model = torch.load(model_filepath, map_location=torch.device(self.device))
        else:
            model = torch.load(model_filepath).cuda()
        model.eval()

        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        dir_components = model_filepath.split("/")
        models_dirpath = "/".join(dir_components[:-2])
        model_dir = dir_components[-2]
        #images = self.gather_images(os.path.join(models_dirpath, model_dir, "clean-example-data"))
        #images = self.generate_images(models_dirpath, model_dir, model)
        train_images, val_images, test_images = self.generate_images(models_dirpath, model_dir, model)

        if train_images == None:
            probability = 0.5
        else:
            with open(os.path.join(models_dirpath, model_dir, "config.json")) as f:
                config = json.load(f)

            #arch = config["py/state"]["model_architecture"]

            # feature_vector = self.get_features_misclass(images, model)

            # feature_vector = np.array([feature_vector])

            if len(list(model.named_parameters())) == 326:
                feature_vector = self.get_features_detr(train_images, val_images, test_images, model)
            else:
                feature_vector = self.get_features_other(train_images, model)

            # with open(os.path.join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
            #     clf = pickle.load(fp)
            # probability = np.clip(clf.predict_proba(feature_vector)[0][1], 0.25, 0.75)
            #print(feature_vector)
            if feature_vector[0] == True:
                probability = 0.7
            else:
                probability = 0.3

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
