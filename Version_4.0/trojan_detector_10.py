import os
from tabnanny import check
import torchvision
import torch
import torch.nn as nn
import models
import math
import numpy as np
#import pandas as pd
import random
import cv2

import json
import jsonschema
import jsonpickle
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import OrderedDict
import logging
import warnings

warnings.filterwarnings("ignore")
from scipy import stats
#from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from pycocotools import mask as maskUtils
from pycocotools import cocoeval
import copy
from scipy.ndimage.filters import gaussian_filter
import time

def ssd(device, path):
    #print("ssd", device)
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"models/id-00000138/model.pt")).to(device)#, map_location=device)#138
    model2 = torch.load(os.path.join(path,"models/id-00000131/model.pt")).to(device)
    model3 = torch.load(os.path.join(path,"models/id-00000127/model.pt")).to(device)

def rcnn(device, path):
    #print("rcnn", device)
    global model1
    global model2
    global model3
    model1 = torch.load(os.path.join(path,"models/id-00000135/model.pt"), map_location=torch.device(device))
    model2 = torch.load(os.path.join(path,"models/id-00000141/model.pt"), map_location=torch.device(device))
    model3 = torch.load(os.path.join(path,"models/id-00000142/model.pt"), map_location=torch.device(device))

def prepare_boxes(anns, image_id):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for answer in anns:
            boxes.append(answer['bbox'])
            class_ids.append(answer['category_id'])

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
        # convert [x,y,w,h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target

def trojan_detector(model_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            coco_dirpath,
                            source_dataset_dirpath,
                            round_training_dataset_dirpath,
                            parameters_dirpath,
                            num_runs,
                            num_examples,
                            epsilon,
                            max_iter,
                            add_delta,
                            object_threshold,
                            trigger_size,
                            find_label_dist,
                            misclassification_dist,
                            feature_dist):
    
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('source_dataset_dirpath = {}'.format(source_dataset_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))

    logging.info('Using parameters_dirpath = {}'.format(parameters_dirpath))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    # load the model
    model = torch.load(model_filepath)
    # move the model to the device
    model.to(device)
    model.eval()
    
    features = list(gen_features(model, model_filepath, round_training_dataset_dirpath, coco_dirpath, device, num_runs, num_examples, epsilon, max_iter, add_delta, object_threshold, trigger_size, find_label_dist, misclassification_dist, feature_dist))
    #print(features)
            
    clf = load(os.path.join(parameters_dirpath, "clf.joblib"))
    trojan_probability = clf.predict_proba(np.array(features).reshape(1,-1))[0][1]

    logging.info('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))
        
def configure(source_dataset_dirpath,
              output_parameters_dirpath,
              configure_models_dirpath,
              coco_dirpath,
              num_runs,
              num_examples,
              epsilon,
              max_iter,
              add_delta,
              object_threshold,
              trigger_size,
              find_label_dist,
              misclassification_dist,
              feature_dist):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #source_dataset_dirpath = "./round10-train-dataset/models/"
    #source_dataset_dirpath = "/mnt/bigpool/ssd1/myudin/round10-train-dataset/models/"

    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    logging.info('Reading source dataset from ' + source_dataset_dirpath)
    
    features = []
    labels = []
    
    for i, model_dirpath in enumerate(sorted(os.listdir(configure_models_dirpath))):

        if i < 0: continue
        #print(model_dirpath)
        model_filepath = configure_models_dirpath + model_dirpath + "/model.pt"
        examples_dirpath = configure_models_dirpath + model_dirpath+"/clean-example-data/"
        # load the model
        model = torch.load(model_filepath)
        # move the model to the device
        model.to(device)
        model.eval()
        
        feature_vector = list(gen_features(model, model_filepath, source_dataset_dirpath, coco_dirpath, device, num_runs, num_examples, epsilon, max_iter, add_delta, object_threshold, trigger_size, find_label_dist, misclassification_dist, feature_dist))
        #print(feature_vector)
        features.append(feature_vector)
        
        label = "trigger_0.png" in os.listdir(configure_models_dirpath+model_dirpath)
        #label = np.loadtxt(os.path.join(configure_models_dirpath, model_dirpath, 'ground_truth.csv'), dtype=bool)
        labels.append(label)
        
        #data = np.concatenate((np.array(features), np.expand_dims(np.array(labels),-1)), axis=1)
        #f = open("rd10.csv", "w")
        #np.savetxt(f, data, delimiter=",")
        
    features = np.array(features)
    labels = np.expand_dims(np.array(labels),-1)
    data = np.concatenate((features, labels), axis=1)

    model, scaler = train_model(data)
    dump(scaler, os.path.join(output_parameters_dirpath, "scaler.joblib"))
    dump(model, os.path.join(output_parameters_dirpath, "clf.joblib"))
    
def gen_features(model, model_filepath, round_training_dataset_dirpath, coco_dirpath, device, num_runs, num_examples, epsilon, max_iter, add_delta, object_threshold, trigger_size, find_label_dist, misclassification_dist, feature_dist):

    if isinstance(model, models.SSD) or isinstance(model, torchvision.models.detection.ssd.SSD):
        ssd(device, round_training_dataset_dirpath)
    if not isinstance(model, models.SSD):
        rcnn(device, round_training_dataset_dirpath)

    for m in range(num_runs):

        triggers = []
        tgts = []
        trigger_fns = []
        src_classes = []

        for image_class_dirpath in os.listdir(coco_dirpath):
            #if int(image_class_dirpath) != 74: continue
            fns = [os.path.join(coco_dirpath, image_class_dirpath, fn) for fn in os.listdir(os.path.join(coco_dirpath, image_class_dirpath)) if fn.endswith('.jpg')]
            fns.sort()
            for fn_i, fn in enumerate(sorted(fns)):
                if fn_i >= 1:
                    continue

                images, targets = get_targets(fn, device)

                with torch.no_grad():
                    outputs = model(images, targets)
                # older models which predate NIST's forward function override return just boxes here
                if isinstance(outputs, tuple):
                    # NIST's forward function override returns loss and boxes
                    outputs = outputs[1]

                first_trigger = True
                visualize = False
                labels = targets[0]["labels"]
                boxes = targets[0]["boxes"]
                predicted_labels = outputs[0]['labels']
                predicted_scores = outputs[0]['scores']
                i2 = int(image_class_dirpath)
                
                for j in range(-10, 90):
                    if j%10 != 0: continue

                    images[0].requires_grad = False
                    new_data = copy.deepcopy(images[0])

                    src_box = None
                    for box, label in zip(boxes, labels):
                        if label == i2:
                            src_box = box
                            break
                    if src_box == None:
                        continue

                    if visualize:
                        og_img = new_data.detach().numpy()
                        og_img = np.transpose(og_img*255, (1, 2, 0))
                        og_img = Image.fromarray(og_img.astype('uint8'), 'RGB')
                        og_img.save('og.png')
                        #og_img.show()
                        visualize_image('og.png', None, True, True, model_filepath)

                    misclassify_j = find_label(model, new_data, None, j, max_iter, epsilon, trigger_size, find_label_dist, object_threshold, targets, src_box, i2, fn, device)
                    if j == -10:
                        new_data, signed_grad = generate_evasive_trigger(model, new_data, i2, None, max_iter, epsilon, trigger_size, misclassification_dist, object_threshold, targets, src_box, i2, fn, device)
                    if j >= 0 :
                        new_data, signed_grad = generate_trigger(model, new_data, None, misclassify_j, max_iter, epsilon, trigger_size, misclassification_dist, object_threshold, targets, src_box, i2, fn, device)
                    if signed_grad == None:
                        continue           
                    #f list(labels2).count(j) <= object_count:
                    #    continue      
                    if add_delta:
                        trigger = new_data - images[0]
                        trigger, (x0,y0,x1,y1) = crop_trigger(trigger, images[0].shape)
                    else:
                        trigger = copy.deepcopy(new_data) * torch.abs(signed_grad)
                        trigger, (x0,y0,x1,y1) = crop_trigger(trigger, images[0].shape)
                    #trigger = trigger[:,:50,:50]
                    triggers.append(trigger.detach())
                    tgts.append(misclassify_j)
                    trigger_fns.append(fn)
                    src_classes.append(image_class_dirpath)
                            
                    images[0].requires_grad = True
                    
                    #print(i,j)
                    #print("final: ", scores2, labels2, iter_i)
                    if visualize:
                        og_img = new_data.detach().numpy()
                        og_img = np.transpose(og_img*255, (1, 2, 0))
                        og_img = Image.fromarray(og_img.astype('uint8'), 'RGB')
                        og_img.save('perturb.png')
                        #og_img.show()
                        visualize_image('perturb.png', None, True, True, model_filepath)
                    
                        og_img = trigger.detach().numpy()
                        og_img = np.transpose(og_img*255, (1, 2, 0))
                        og_img = Image.fromarray(og_img.astype('uint8'), 'RGB')
                        #og_img.save(fn+str(i)+'perturb.png')
                        #og_img.show()

                    if first_trigger and j >= 0: break
                    if len(triggers) > 20:
                        break
                    #print(image_class_dirpath, j)

        metrics = []
        count_labels = dict()
        count_misclasses = dict()
        count_labels2 = dict()
        count_misclasses2 = dict()
        count_evasions = dict()
        count_evasions2 = dict()
        count_evasions[-1] = 0
        #print(len(triggers))
        for t in range(len(triggers)):
            trigger =  triggers[t]
            fn = trigger_fns[t]
            trigger_height = trigger.shape[1]
            trigger_width = trigger.shape[2]
            tgt = tgts[t]
            src_class = src_classes[t]
            
            misclass = 0
            total = 0
        
            fns = [os.path.join(coco_dirpath, src_class, fn) for fn in os.listdir(os.path.join(coco_dirpath, src_class)) if fn.endswith('.jpg')]
            fns.sort()
            for fn2_i, fn2 in enumerate(sorted(fns)):
                
                if fn2==fn: continue

                if fn2_i > num_examples + 1:
                    continue
                #print(fn2, fn)
                images2, targets2 = get_targets(fn2, device)

                outputs = model(images2, targets2)
                if isinstance(outputs, tuple):
                    outputs = outputs[1]
                
                #valid_score_indices = outputs[0]['scores']>0.12
                #scores3 = outputs[0]['scores'][:10]
                labels3 = targets2[0]['labels'][:10]#outputs[0]['labels'][valid_score_indices]
                boxes3 = targets2[0]['boxes'][:10]#outputs[0]['boxes'][valid_score_indices]
                

                valid_score_indices = outputs[0]['scores']>object_threshold
                predicted_scores3 = outputs[0]['scores'][valid_score_indices]
                predicted_labels3 = outputs[0]['labels'][valid_score_indices]
                predicted_boxes3 = outputs[0]['boxes'][valid_score_indices]

                offset = 10
                for t2 in range(len(boxes3)):
                    orig_box = boxes3[t2]
                    orig_label = labels3[t2].item()
                    if orig_label not in count_labels: count_labels[orig_label] = dict()
                    if orig_label not in count_misclasses: count_misclasses[orig_label] = dict()
                    if orig_label not in count_labels2: count_labels2[orig_label] = dict()
                    if orig_label not in count_misclasses2: count_misclasses2[orig_label] = dict()
                    if orig_label not in count_evasions: count_evasions[orig_label] = 0
                    if orig_label not in count_evasions2: count_evasions2[orig_label] = 0
                    
                    #print(orig_box)
                    x0 = int(torch.round(orig_box[0].detach()).item())
                    y0 = int(torch.round(orig_box[1].detach()).item())
                    #print(x0, y0, trigger_width, trigger_height)
                    
                    image = copy.deepcopy(images2[0])
                    image_to_modify = image[:,y0+offset:y0+offset+trigger_height,x0+offset:x0+trigger_width+offset]

                    if add_delta:
                        image[:,y0+offset:y0+trigger_height+offset,x0+offset:x0+trigger_width+offset] = image_to_modify + trigger[:,:image_to_modify.shape[1],:image_to_modify.shape[2]]
                    if not add_delta:
                        image[:,y0+offset:y0+trigger_height+offset, x0+offset:x0+trigger_width+offset] = trigger[:,:image_to_modify.shape[1],:image_to_modify.shape[2]]

                    outputs = model([image], targets2)
                    if isinstance(outputs, tuple):
                        outputs = outputs[1]
                        
                    valid_score_indices = outputs[0]['scores']>object_threshold
                    predicted_scores4 = outputs[0]['scores'][valid_score_indices]
                    predicted_labels4 = outputs[0]['labels'][valid_score_indices]
                    predicted_boxes4 = outputs[0]['boxes'][valid_score_indices]

                    if visualize:
                        og_img = image.detach().numpy()
                        og_img = np.transpose(og_img*255, (1, 2, 0))
                        og_img = Image.fromarray(og_img.astype('uint8'), 'RGB')
                        og_img.save('add.png')
                        #og_img.show()
                        visualize_image('add.png', None, True, True, model_filepath)
                        
                    min_dist = feature_dist
                    detected = False
                    pred_label_box = -1
                    for pred_box, pred_label, pred_score in zip(predicted_boxes4, predicted_labels4, predicted_scores4):
                        d = dist(pred_box, orig_box)
                        if d < min_dist:
                            detected = True
                            if pred_label.item() != orig_label:
                                min_dist = d
                                pred_label_box = pred_label.item()
                    #print(pred_label_box, orig_label)
                    if not detected:
                        count_evasions[orig_label] += 1
                    if pred_label_box != -1:
                        if pred_label_box not in count_labels[orig_label]: count_labels[orig_label][pred_label_box] = 0
                        if pred_label_box not in count_misclasses[orig_label]: count_misclasses[orig_label][pred_label_box] = 0
                        
                        if pred_label_box == tgt and orig_label == int(src_class):#if pred_label_box != orig_label:
                            misclass += 1
                            count_misclasses[orig_label][pred_label_box] += 1
                        total += 1
                        count_labels[orig_label][pred_label_box] += 1
                        
                    outputs = model1([image], targets2)
                    if isinstance(outputs, tuple):
                        outputs = outputs[1]
                    
                    valid_score_indices = outputs[0]['scores']>object_threshold
                    predicted_labels4 = outputs[0]['labels'][valid_score_indices]
                    predicted_boxes4 = outputs[0]['boxes'][valid_score_indices]
                        
                    min_dist = feature_dist
                    detected = False
                    pred_label_box = -1
                    for pred_box, pred_label in zip(predicted_boxes4, predicted_labels4):
                        d = dist(pred_box, orig_box)
                        if d < min_dist:
                            detected = True
                            if pred_label.item() != orig_label:
                                min_dist = d
                                pred_label_box = pred_label.item()
                    if not detected:
                        count_evasions2[orig_label] += 1
                    if pred_label_box != -1:
                        if pred_label_box not in count_labels2[orig_label]: count_labels2[orig_label][pred_label_box] = 0
                        if pred_label_box not in count_misclasses2[orig_label]: count_misclasses2[orig_label][pred_label_box] = 0
                        
                        if pred_label_box == tgt and orig_label == int(src_class):#pred_label_box != orig_label:
                            misclass += 1
                            count_misclasses2[orig_label][pred_label_box] += 1
                        total += 1
                        count_labels2[orig_label][pred_label_box] += 1

        max_misclass_diff = 0
        max_evasion_diff = 0
        max_misclass_rate = 0
        max_misclass_src = None
        max_misclass_tgt = None
        for src_cls in count_labels:
            for tgt_cls in count_labels[src_cls]:
                #print(src_cls, tgt_cls)
                if tgt_cls == src_cls: continue
                if count_labels[src_cls][tgt_cls] < 4: continue
                misclass_rate = count_misclasses[src_cls][tgt_cls] / count_labels[src_cls][tgt_cls]
                if misclass_rate > max_misclass_rate:
                    max_misclass_rate = misclass_rate
                if tgt_cls not in count_labels2[src_cls]:
                    misclass_rate2 = 0
                else:
                    misclass_rate2 = count_misclasses2[src_cls][tgt_cls] / count_labels2[src_cls][tgt_cls]
                misclass_diff = misclass_rate - misclass_rate2
                if misclass_diff >= max_misclass_diff:
                    #print(src_cls, tgt_cls)
                    max_misclass_diff = misclass_diff
                    if tgt_cls not in count_labels2[src_cls]:
                        misclass_rate2 = 0
                    else:
                        misclass_rate2 = count_misclasses2[src_cls][tgt_cls] / count_labels2[src_cls][tgt_cls]
                    max_misclass_src = src_cls
                    max_misclass_tgt = tgt_cls
                    count = count_labels[src_cls][tgt_cls]
            evasion_diff = count_evasions[src_cls] - count_evasions2[src_cls]
            if evasion_diff > max_evasion_diff:
                max_evasion_diff = evasion_diff
                max_evasion_src = src_cls

        return max_misclass_diff, max_evasion_diff, max_misclass_rate#, count_evasions[max(count_evasions)]

def dist(b1, b2):
    return math.sqrt((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2 + (b1[2] - b2[2])**2 + (b1[3] - b2[3])**2)
              
def get_targets(fn, device):
    image_id = os.path.basename(fn)
    image_id = int(image_id.replace('.jpg',''))
    # load the example image
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)  # loads to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

    # load the annotation
    with open(fn.replace('.jpg', '.json')) as json_file:
        # contains a list of coco annotation dicts
        annotations = json.load(json_file)

    with torch.no_grad():
        # convert the image to a tensor
        # should be uint8 type, the conversion to float is handled later
        image = torch.as_tensor(image)
        # move channels first
        image = image.permute((2, 0, 1))
        # convert to float (which normalizes the values)
        image = torchvision.transforms.functional.convert_image_dtype(image, torch.float)
        images = [image]  # wrap into list

        # prep targets
        targets = prepare_boxes(annotations, image_id)
        # wrap into list
        targets = [targets]

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    return images, targets

def get_logits(model, images, targets, original_image_sizes=None):
    #print(model[0])
    if original_image_sizes == None:
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

    #with torch.cpu.amp.autocast():

    images, targets = model.transform(images, targets)
                
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    
    
    if isinstance(model, models.SSD):# or isinstance(model, torchvision.models.detection.ssd.SSD):
        features = list(features.values())
        head_outputs = model.head(features)
        logits = head_outputs["cls_logits"]
        #softmax = nn.Softmax(dim=2)
        #confs = softmax(logits)

    if not isinstance(model, models.SSD):
        proposals, proposal_losses = model.rpn(images, features, targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)
        logits = torch.unsqueeze(class_logits, 0)
        #softmax = nn.Softmax(dim=2)
        #confs = softmax(logits)

    return logits, None#confs.detach()

def find_label(model, new_data, i, j, max_iter, epsilon, trigger_size, find_label_dist, object_threshold, targets, src_box, src_label, fn, device):

    src_box = torch.round(src_box).int()
    original_image_sizes = []
    val = new_data.shape[-2:]
    original_image_sizes = [(val[0], val[1])]

    filter_shape = torch.ones(new_data.shape).to(device)
    filter_shape[:,:(src_box[1]+src_box[3])//2,:] = 0
    filter_shape[:,(src_box[1]+src_box[3])//2+trigger_size:,:] = 0
    filter_shape[:,:,:(src_box[0]+src_box[2])//2] = 0
    filter_shape[:,:,(src_box[0]+src_box[2])//2+trigger_size:] = 0

    for iter_i in range(max_iter):
        new_data.requires_grad = True
        logits, confs = get_logits(model, [new_data], targets, original_image_sizes)
        if j < logits.shape[2]-9:
            logit = logits[:,:,j] + logits[:,:,j+1] + logits[:,:,j+2] + logits[:,:,j+3] + logits[:,:,j+4] + logits[:,:,j+5] + logits[:,:,j+6] + logits[:,:,j+7] + logits[:,:,j+8] + logits[:,:,j+9]
        if j == logits.shape[2]-2:
            logit = logits[:,:,j] + logits[:,:,j+1]
        if j == logits.shape[2]-1:
            logit = logits[:,:,j]
        gradients = torch.autograd.grad(outputs=logit, inputs=new_data, grad_outputs=torch.ones(logit.size()).to(device), only_inputs=True, retain_graph=True)[0]
        signed_grad = torch.sign(gradients)
        signed_grad = signed_grad * filter_shape
        new_data.requires_grad = False
        #new_data = new_data.detach()
        new_data = new_data + (epsilon * signed_grad)

    with torch.cuda.amp.autocast():
        outputs = model([new_data], targets)
    if isinstance(outputs, tuple):
        outputs = outputs[1]
    valid_score_indices = outputs[0]['scores']>object_threshold
    labels = outputs[0]['labels'][valid_score_indices]
    boxes = outputs[0]['boxes'][valid_score_indices]
    box_i = None
    mindist = find_label_dist
    misclass_label = -1
    fail2 = 1
    for box, label in zip(boxes, labels):
        d = dist(box, src_box)
        if d < mindist and label.item() != src_label and label >= j and label < j+10:
            mindist = d
            misclass_label = label

    return misclass_label

def generate_trigger(model, new_data, i, j, max_iter, epsilon, trigger_size, misclassification_dist, object_threshold, targets, src_box, src_label, fn, device):

    src_box = torch.round(src_box).int()
    original_image_sizes = []
    val = new_data.shape[-2:]
    original_image_sizes = [(val[0], val[1])]

    filter_shape = torch.ones(new_data.shape).to(device)
    filter_shape[:,:(src_box[1]+src_box[3])//2,:] = 0
    filter_shape[:,(src_box[1]+src_box[3])//2+trigger_size:,:] = 0
    filter_shape[:,:,:(src_box[0]+src_box[2])//2] = 0
    filter_shape[:,:,(src_box[0]+src_box[2])//2+trigger_size:] = 0

    for iter_i in range(max_iter):
        new_data.requires_grad = True
        logits, confs = get_logits(model, [new_data], targets, original_image_sizes)
        logit = logits[:,:,j]
        gradients = torch.autograd.grad(outputs=logit, inputs=new_data, grad_outputs=torch.ones(logit.size()).to(device), only_inputs=True, retain_graph=True)[0]
        signed_grad = torch.sign(gradients)
        signed_grad = signed_grad * filter_shape
        new_data.requires_grad = False
        #new_data = new_data.detach()
        new_data = new_data + (epsilon * signed_grad)

    with torch.cuda.amp.autocast():
        outputs = model([new_data], targets)
    if isinstance(outputs, tuple):
        outputs = outputs[1]
    valid_score_indices = outputs[0]['scores']>object_threshold
    labels = outputs[0]['labels'][valid_score_indices]
    boxes = outputs[0]['boxes'][valid_score_indices]
    with torch.cuda.amp.autocast():
        outputs1 = model1([new_data], targets)
    if isinstance(outputs1, tuple):
        outputs1 = outputs1[1]
    valid_score_indices = outputs1[0]['scores']>object_threshold
    labels1 = outputs1[0]['labels'][valid_score_indices]
    boxes1 = outputs1[0]['boxes'][valid_score_indices]
    box_i = None
    fail1 = 0
    fail2 = 1
    for box, label in zip(boxes, labels):
        d = dist(box, src_box)
        if d < misclassification_dist and label.item() != src_label and label.item() == j:
            fail1 = 1
    for box, label in zip(boxes1, labels1):
        d = dist(box, src_box)
        if d < misclassification_dist and label.item() != src_label and label.item() == j:
            fail2 = 0
    if fail1 and fail2:
        return new_data, signed_grad.detach()
        
    fail1 = 0
    fail2 = 1

    return None, None

def generate_evasive_trigger(model, new_data, i, j, max_iter, epsilon, trigger_size, misclassification_dist, object_threshold, targets, src_box, src_label, fn, device):

    src_box = torch.round(src_box).int()
    original_image_sizes = []
    val = new_data.shape[-2:]
    original_image_sizes = [(val[0], val[1])]

    filter_shape = torch.ones(new_data.shape).to(device)
    filter_shape[:,:(src_box[1]+src_box[3])//2,:] = 0
    #signed_grad[:,src_box[3]:,:] = 0
    filter_shape[:,(src_box[1]+src_box[3])//2+trigger_size:,:] = 0
    filter_shape[:,:,:(src_box[0]+src_box[2])//2] = 0
    #signed_grad[:,:,src_box[2]:] = 0
    filter_shape[:,:,(src_box[0]+src_box[2])//2+trigger_size:] = 0

    for iter_i in range(max_iter):
        new_data.requires_grad = True
        logits, confs = get_logits(model, [new_data], targets, original_image_sizes)
        logit = logits[:,:,i]
        gradients = torch.autograd.grad(outputs=logit, inputs=new_data, grad_outputs=torch.ones(logit.size()).to(device), only_inputs=True, retain_graph=True)[0]
        signed_grad = torch.sign(gradients)
        signed_grad = signed_grad * filter_shape
        new_data.requires_grad = False
        #new_data = new_data.detach()
        new_data = new_data - (epsilon * signed_grad)

    with torch.cuda.amp.autocast():
        outputs = model([new_data], targets)
    if isinstance(outputs, tuple):
        outputs = outputs[1]
    valid_score_indices = outputs[0]['scores']>object_threshold
    #scores = outputs[0]['scores'][valid_score_indices]
    labels = outputs[0]['labels'][valid_score_indices]
    boxes = outputs[0]['boxes'][valid_score_indices]
    with torch.cuda.amp.autocast():
        outputs1 = model1([new_data], targets)
    if isinstance(outputs1, tuple):
        outputs1 = outputs1[1]
    valid_score_indices = outputs1[0]['scores']>object_threshold
    #scores1 = outputs1[0]['scores'][valid_score_indices]
    labels1 = outputs1[0]['labels'][valid_score_indices]
    boxes1 = outputs1[0]['boxes'][valid_score_indices]
    #print(scores, labels, scores1, labels1) 
    box_i = None
    fail1 = 1
    #fail2 = 0
    for box, label in zip(boxes, labels):
        d = dist(box, src_box)
        #print(d, label)
        if d < misclassification_dist and label == i:
            fail1 = 0
            #print(d, label)
    if fail1:# and fail2:
        return new_data, signed_grad.detach()
        
    return None, None

def crop_trigger(trigger, shape):
    x0 = 0
    while torch.sum(trigger[:,:,x0]) == 0:
        x0 += 1
    x1 = -1
    while torch.sum(trigger[:,:,x1]) == 0:
        x1 -= 1
    y0 = 0
    while torch.sum(trigger[:,y0,:]) == 0:
        y0 += 1
    y1 = -1
    while torch.sum(trigger[:,y1,:]) == 0:
        y1 -= 1

    x1 = shape[2] + x1
    y1 = shape[1] + y1
    
    return trigger[:,y0:y1,x0:x1], (x0,y0,x1,y1)

def compute_mAP(coco_dataset_filepath: str, coco_annotation_filepath: str, model_filepath: str):
    coco_dataset = torchvision.datasets.CocoDetection(root=coco_dataset_filepath, annFile=coco_annotation_filepath)

    # modify the entries (images and annotations) in coco_dataset to operate on a subset of data.
    # after modifications to the coco_dataset object, you need to call coco_dataset.coco.createIndex() to rebuild all of the links within the coco object.

    # inference and use those boxes instead of the annotations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    pytorch_model = torch.load(model_filepath)
    # move the model to the device
    pytorch_model.to(device)
    pytorch_model.eval()

    coco_results = list()
    with torch.no_grad():
        for image in coco_dataset.coco.dataset['images'][:1]:
            id = image['id']
            filename = image['file_name']
            width = image['width']
            height = image['height']
            coco_anns = coco_dataset.coco.imgToAnns[id]
            filepath = os.path.join(coco_dataset_filepath, filename)

            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # loads to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

            # convert the image to a tensor
            # should be uint8 type, the conversion to float is handled later
            img = torch.as_tensor(img)
            # move channels first
            img = img.permute((2, 0, 1))
            # convert to float (which normalizes the values)
            img = torchvision.transforms.functional.convert_image_dtype(img, torch.float)
            images = [img]  # wrap into list

            images = list(img.to(device) for img in images)

            outputs = pytorch_model(images)
            output = outputs[0]  # one image at a time is batch_size 1
            boxes = output["boxes"]
            boxes = models.x1y1x2y2_to_xywh(boxes).tolist()
            scores = output["scores"].tolist()
            labels = output["labels"].tolist()

            # convert boxes into format COCOeval wants
            res = [{"image_id": id, "category_id": labels[k], "bbox": box, "score": scores[k]} for k, box in enumerate(boxes)]
            coco_results.extend(res)

    coco_dt = coco_dataset.coco.loadRes(coco_results)

    coco_evaluator = cocoeval.COCOeval(cocoGt=coco_dataset.coco, cocoDt=coco_dt, iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    mAP = float(coco_evaluator.stats[0])
    print('mAP = {}'.format(mAP))
    return map

def showAnns(coco_anns, height, width, draw_bbox=False, draw_number=False):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(coco_anns) == 0:
        return 0
    if 'segmentation' in coco_anns[0] or 'keypoints' in coco_anns[0]:
        datasetType = 'instances'
    elif 'caption' in coco_anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in coco_anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        polygons.append(Polygon(poly))
                        color.append(c)
                else:
                    # mask
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], height, width)
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)
                    img = np.ones( (m.shape[0], m.shape[1], 3) )
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0,166.0,101.0])/255
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:,:,i] = color_mask[i]
                    ax.imshow(np.dstack( (img, m*0.5) ))

            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                polygons.append(Polygon(np_poly))
                color.append(c)
            if draw_number:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                cx = bbox_x + int(bbox_w / 2)
                cy = bbox_y + int(bbox_h / 2)
                #ax.text(cx, cy, "{}".format(ann['category_id']), c='k', fontsize='large', fontweight='bold')
                ax.text(cx, cy, "{}".format(ann['category_id']), c=c, backgroundcolor='white', fontsize='small', fontweight='bold')

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
    
def draw_boxes(img, boxes, colors_list=None):
    """
    Args:
        img: Image to draw boxes onto
        boxes: boxes in [x1, y1, x2, y2] format
        value: what pixel value to draw into the image
    Returns: modified image
    """
    buff = 2

    if boxes is None:
        return img

    if colors_list is None:
        # default to red
        colors_list = list()
        while len(colors_list) < boxes.shape[0]:
            colors_list.append([255, 0, 0])

    # make a copy to modify
    img = img.copy()
    for i in range(boxes.shape[0]):
        x_st = round(boxes[i, 0])
        y_st = round(boxes[i, 1])

        x_end = round(boxes[i, 2])
        y_end = round(boxes[i, 3])

        # draw a rectangle around the region of interest
        img[y_st:y_st+buff, x_st:x_end, :] = colors_list[i]
        img[y_end-buff:y_end, x_st:x_end, :] = colors_list[i]
        img[y_st:y_end, x_st:x_st+buff, :] = colors_list[i]
        img[y_st:y_end, x_end-buff:x_end, :] = colors_list[i]

    return img

def inference(model_filepath, image):
    # inference and use those boxes instead of the annotations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    pytorch_model = torch.load(model_filepath)
    # move the model to the device
    pytorch_model.to(device)
    pytorch_model.eval()

    with torch.no_grad():
        # convert the image to a tensor
        # should be uint8 type, the conversion to float is handled later
        image = torch.as_tensor(image)
        # move channels first
        image = image.permute((2, 0, 1))
        # convert to float (which normalizes the values)
        image = torchvision.transforms.functional.convert_image_dtype(image, torch.float)
        images = [image]  # wrap into list

        images = list(image.to(device) for image in images)

        outputs = pytorch_model(images)
        outputs = outputs[0]  # get the results for the single image forward pass
    return outputs

def visualize_image(img_fp, output_dirpath, draw_bbox=False, draw_number=False, model_filepath=None):
    parent_fp, img_fn = os.path.split(img_fp)
    ann_fn = img_fn.replace('.jpg', '.json')
    ann_fp = os.path.join(parent_fp, ann_fn)

    if not os.path.exists(img_fp):
        raise RuntimeError('Requested image file {} does not exists.'.format(img_fp))
    if not os.path.exists(ann_fp):
        raise RuntimeError('Requested image annotation file {} does not exists.'.format(ann_fp))

    img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED)  # loads to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    height = img.shape[0]
    width = img.shape[1]

    # clear the figure
    plt.clf()

    if model_filepath is not None:
        outputs = inference(model_filepath, img)
        boxes = outputs['boxes'].detach().cpu().numpy()
        scores = outputs['scores'].detach().cpu().numpy()
        labels = outputs['labels'].detach().cpu().numpy()
        boxes = boxes[scores > 0.5, :]

        # get colors for the boxes
        colors_list = list()
        cmap = matplotlib.cm.get_cmap('jet')
        for b in range(boxes.shape[0]):
            idx = float(b) / float(boxes.shape[0])
            c = cmap(idx)
            c = c[0:3]
            c = [int(255.0 * x) for x in c]
            colors_list.append(c)

        # draw the output boxes onto the image
        img = draw_boxes(img, boxes, colors_list)
        # show the image
        plt.imshow(img)

        if draw_number:
            ax = plt.gca()

            for b in range(boxes.shape[0]):
                c = colors_list[b]
                # ax.text needs color in [0, 1]
                c = [float(x) / 255.0 for x in c]
                [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = boxes[b, :]
                cx = int(float((bbox_x1 + bbox_x2) / 2.0))
                cy = int(float((bbox_y1 + bbox_y2) / 2.0))
                ax.text(cx, cy, "{}".format(labels[b]), c=c, backgroundcolor='white', fontsize='small', fontweight='bold')
    else:
        with open(ann_fp, 'r') as fh:
            coco_anns = json.load(fh)

        # show the image
        plt.imshow(img)
        # draw the annotations on top of that image
        showAnns(coco_anns, height, width, draw_bbox=draw_bbox, draw_number=draw_number)

    # save figure to disk
    if output_dirpath is not None:
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        plt.savefig(os.path.join(output_dirpath, img_fn))
    else:
        # or render it
        plt.show()
                    

def train_model(data):

    X = data[:,:-1].astype(np.float32)
    y = data[:,-1]

    #parameters = {'gamma':[0.01, 0.1], 'C':[1, 10]}
    #clf_svm = SVC(probability=True, kernel='rbf')
    #clf_svm = GridSearchCV(clf_svm, parameters)
    clf_lr = LogisticRegression()
    sc = StandardScaler()
    sc.fit(X)
    clf = clf_lr.fit(X, y)

    return clf, sc


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--source_dataset_dirpath', type=str, help='File path to a directory containing the original clean dataset into which triggers were injected during training.', default=None)
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--num_runs', type=int, help='An example tunable parameter.')
    parser.add_argument('--num_examples', type=int, help='An example tunable parameter.')
    parser.add_argument('--epsilon', type=float, help='Gradient descent step size.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of steps of gradient descent.')
    parser.add_argument('--add_delta', type=int, help='Add a quasi-trigger.')
    parser.add_argument('--object_threshold', type=float, help='Confidence threshold for whether an object was detected.')
    parser.add_argument('--trigger_size', type=int, help='Size of pixel patch (l x l).')
    parser.add_argument('--find_label_dist', type=int, help='Minimum distance to assume the model has misclassified the source object (for determining target class).')
    parser.add_argument('--misclassification_dist', type=int, help='Minimum distance to assume the model has misclassified the source object (for determining whether trigger is valid).')
    parser.add_argument('--feature_dist', type=int, help='Minimum distance to assume the model has misclassified the source object (for generating misclassifacation/evasion statistics).')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("example_trojan_detector.py launched")
    
    #coco_dirpath = "round10-train-dataset/data"
    coco_dirpath = "/data"

    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.result_filepath is not None and
            args.scratch_dirpath is not None and
            args.examples_dirpath is not None and
            args.source_dataset_dirpath is not None and
            args.round_training_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None and
            args.num_runs is not None and
            args.num_examples is not None and
            args.epsilon is not None and
            args.max_iter is not None and
            args.add_delta is not None and
            args.object_threshold is not None and
            args.trigger_size is not None and
            args.find_label_dist is not None and
            args.misclassification_dist is not None and
            args.feature_dist is not None):

            logging.info("Calling the trojan detector")
            trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    coco_dirpath,
                                    args.source_dataset_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    args.num_runs,
                                    args.num_examples,
                                    args.epsilon,
                                    args.max_iter,
                                    args.add_delta,
                                    args.object_threshold,
                                    args.trigger_size,
                                    args.find_label_dist,
                                    args.misclassification_dist,
                                    args.feature_dist)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.source_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None and
            args.num_runs is not None and
            args.num_examples is not None and
            args.max_iter is not None and
            args.epsilon is not None and
            args.add_delta is not None,
            args.object_threshold is not None and
            args.trigger_size is not None and
            args.find_label_dist is not None and
            args.misclassification_dist is not None and
            args.feature_dist is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.source_dataset_dirpath,
                      args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      coco_dirpath,
                      args.num_runs,
                      args.num_examples,
                      args.epsilon,
                      args.max_iter,
                      args.add_delta,
                      args.object_threshold,
                      args.trigger_size,
                      args.find_label_dist,
                      args.misclassification_dist,
                      args.feature_dist)
        else:
            logging.info("Required Configure-Mode parameters missing!")



