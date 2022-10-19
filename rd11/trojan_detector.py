import os
import copy
import torch
import torchvision
import numpy as np
import cv2
import skimage.transform
import json
import jsonschema
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import logging
import warnings
warnings.filterwarnings("ignore")

#from PIL import Image
    

def trojan_detector(model_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            round_training_dataset_dirpath,
                            parameters_dirpath,
                            num_runs,
                            num_examples,
                            epsilon,
                            max_iter,
                            add_delta,
                            trigger_size,
                            train_len,
                            val_len,
                            test_len,
                            example_img_format="jpg"):
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    logging.info('Using parameters_dirpath = {}'.format(parameters_dirpath))
    logging.info('Using num_examples = {}'.format(num_examples))
    logging.info('Using epsilon = {}'.format(epsilon))
    logging.info('Using max_iter = {}'.format(num_examples))
    logging.info('Using add_delta = {}'.format(add_delta))
    logging.info('Using trigger_size = {}'.format(trigger_size))
    logging.info('Using train_len = {}'.format(train_len))
    logging.info('Using val_len = {}'.format(val_len))
    logging.info('Using test_len = {}'.format(test_len))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    model_dir = os.path.dirname(model_filepath)

    # load the model and move it to the GPU
    model = torch.load(model_filepath)
    model.to(device)
    model.eval()

    # Augmentation transformations
    #augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

    # Inference the example images in data
    #fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    #fns.sort()  # ensure file ordering
    #if len(fns) > 5: fns = fns[0:5]  # limit to 5 images

    #logging.info('Inferencing {} images'.format(len(fns)))

    #images, locs = generate_example_images(model, model_dir, num_examples, device)

    #features = list(gen_features(model, model_filepath, images, locs, device, num_runs, num_examples, epsilon, max_iter, add_delta, trigger_size, train_len, val_len, test_len))
    features = weight_analysis(model)
    #print(features)
            
    clf = load(os.path.join(parameters_dirpath, "clf.joblib"))
    scaler = load(os.path.join(parameters_dirpath, "scaler.joblib"))
    trojan_probability = clf.predict_proba(scaler.transform(np.array(features).reshape(1,-1)))[0][1]

    logging.info('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))
        
def configure(output_parameters_dirpath,
              configure_models_dirpath,
              num_runs,
              num_examples,
              epsilon,
              max_iter,
              add_delta,
              trigger_size,
              train_len,
              val_len,
              test_len):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)
    
    features = []
    labels = []

    #metadata = np.genfromtxt("/mnt/bigpool/ssd1/myudin/round11-train-dataset/METADATA.csv", dtype=str, skip_header=1, delimiter=",", usecols = (13))
    
    for i, model_dirpath in enumerate(sorted(os.listdir(configure_models_dirpath))):

        #if i < 2: continue
        #if metadata[i] == "ssd": continue
        #print(model_dirpath)
        model_filepath = os.path.join(configure_models_dirpath, model_dirpath, "model.pt")
        examples_dirpath = os.path.join(configure_models_dirpath, model_dirpath, "clean-example-data/")
        # load the model
        model = torch.load(model_filepath)
        # move the model to the device
        model.to(device)
        model.eval()

        feature_vector = list(weight_analysis(model))

        #images, locs = generate_example_images(model, os.path.join(configure_models_dirpath, model_dirpath), num_examples, device)
        
        #feature_vector = list(gen_features(model, model_filepath, images, locs, device, num_runs, num_examples, epsilon, max_iter, add_delta, trigger_size, train_len, val_len, test_len))
        #print(feature_vector)
        features.append(feature_vector)
        
        label = "poisoned-example-data" in os.listdir(os.path.join(configure_models_dirpath, model_dirpath))
        #label = np.loadtxt(os.path.join(configure_models_dirpath, model_dirpath, 'ground_truth.csv'), dtype=bool)
        labels.append(label)
        
        #data = np.concatenate((np.array(features), np.expand_dims(np.array(labels),-1)), axis=1)
        #f = open("rd11.csv", "w")
        #np.savetxt(f, data, delimiter=",")
        
    features = np.array(features)
    labels = np.expand_dims(np.array(labels),-1)
    data = np.concatenate((features, labels), axis=1)

    model, scaler = train_model(data)
    dump(scaler, os.path.join(output_parameters_dirpath, "scaler.joblib"))
    dump(model, os.path.join(output_parameters_dirpath, "clf.joblib"))
    
def weight_analysis(model):
    #print(model)
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
    weights = weights.detach().to('cpu')
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
    #print(std_weights.shape)
    #return 1/0
    max_max_weight = max(max_weights)
    mean_max_weight = torch.mean(max_weights)
    std_max_weight = torch.std(max_weights)
    max_sum_weight = max(sum_weights)
    mean_sum_weight = torch.mean(sum_weights)
    std_sum_weight = torch.std(sum_weights)
    n = avg_weights.shape[0]
    return Q, max_weight, std_weight, max_std_weight, std_max_weight

def gen_features(model, model_filepath, images, locs, device, num_runs, num_examples, epsilon, max_iter, add_delta, trigger_size, train_len, val_len, test_len):

    for m in range(num_runs):
        
        misclass_scores = dict()
        misclass_increases = []
        triggers = []
        tgts = []
        #trigger_fns = []
        src_classes = []

        for src_cls in images:
            #if src_cls != 1: continue
            #if int(image_class_dirpath) %3 != 1: continue
            #print(src_cls)
            fns = images[src_cls]
            #fns.sort()
            train_images = []
            val_images = []
            for fn_i, fn in enumerate(images[src_cls]):

                if fn_i < train_len:
                    train_images.append(fn)

                if fn_i >= train_len and fn_i < train_len+val_len:
                    val_images.append(fn)

            first_trigger = True
            visualize = False
            train_images_send = copy.deepcopy(train_images)
            val_images_send = copy.deepcopy(val_images)

            for tgt_cls in range(len(images)):#logits.shape[2]-65):
                #if tgt_cls != 24: continue
                #if tgt_cls%5 != 0: continue
                if tgt_cls == src_cls: continue

                if visualize:
                    image = train_images[0][0]
                    image = image.permute((1, 2, 0))
                    image = image.cpu().numpy()
                    image = image*255
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                    #image.save('generated_img.png')
                    image.show()

                misclassify_j, trigger, signed_grad = find_label(model, train_images_send, val_images_send, locs, src_cls, tgt_cls, max_iter, epsilon, trigger_size, device)
                #print(src_cls, misclassify_j)
                if misclassify_j == -1:
                    continue
                #new_data, signed_grad = generate_trigger(model, new_data, None, misclassify_j, max_iter, epsilon, trigger_size, misclassification_dist, object_threshold, targets, src_box, i2, fn, device)
                if signed_grad == None:
                    continue
                #f list(labels2).count(j) <= object_count:
                #    continue
                #trigger = trigger[:,:50,:50]
                triggers.append(trigger.detach())
                tgts.append(misclassify_j)
                #trigger_fns.append(fn)
                src_classes.append(src_cls)
                
                if visualize:
                    image = train_images[0][0] + trigger
                    image = image.permute((1, 2, 0))
                    image = image.cpu().numpy()
                    image = image*255
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                    #image.save('generated_img.png')
                    image.show()

                if first_trigger: break
                if len(triggers) > 20:
                    break

        #print(1/0)
        metrics = []
        count_labels = dict()
        count_misclass_conc = dict()
        count_misclasses = dict()
        count_misclass_labels = dict()

        #print(len(triggers))
        for t in range(len(triggers)):
            trigger =  triggers[t]
            #fn = trigger_fns[t]
            tgt_class = tgts[t]
            src_class = src_classes[t]
            
            #print(src_class, tgt_class)
        
            for fn_i2, fn2 in enumerate(images[src_class]):

                if fn_i2 >= train_len+val_len and fn_i2 < train_len+val_len+test_len:

                    #if orig_label != 40: continue
                    if src_class not in count_misclass_labels: count_misclass_labels[src_class] = dict()
                    if src_class not in count_misclasses: count_misclasses[src_class] = 0
                    if src_class not in count_labels: count_labels[src_class] = 0
                    if src_class not in count_misclass_conc: count_misclass_conc[src_class] = dict()             
                    
                    
                    if add_delta:
                        result = fn2 + trigger

                    with torch.cuda.amp.autocast():
                        logits = model(result)
                    #pred = torch.argmax(logits)
                    #print(pred)
                    logits = logits.cpu().detach().numpy()[0]
                    pred_label = np.argmax(logits)
                    #print(pred_label)
                    #print(1/0)
                        
                    if visualize:
                        image = result[0]
                        image = image.permute((1, 2, 0))
                        image = image.cpu().numpy()
                        image = image*255
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        #image.save('generated_img.png')
                        image.show()
                        


                    if pred_label != src_class:

                        if pred_label not in count_misclass_labels[src_class]: count_misclass_labels[src_class][pred_label] = 0
                        if pred_label not in count_misclass_conc[src_class]: count_misclass_conc[src_class][pred_label] = 0
                        #print(pred_label, tgt_class)
                        if pred_label == tgt_class:
                            count_misclass_conc[src_class][pred_label] += 1
                        count_misclasses[src_class] += 1
                        count_misclass_labels[src_class][pred_label] += 1
                    count_labels[src_class] += 1
                    
        #print(count_labels,count_misclass_labels,count_misclasses,count_misclass_conc)
        max_misclass_rate = 0
        max_misclass_conc = 0
        max_misclass_conc_tgt = 0
        max_misclass_src = None
        max_misclass_tgt = None
        for src_cls in count_labels:
            misclass_rate = count_misclasses[src_class] / count_labels[src_class]
            if misclass_rate > max_misclass_rate:
                max_misclass_rate = misclass_rate
            for tgt_cls in count_misclass_labels[src_cls]:
                assert(tgt_cls != src_cls)
                #assert(count_misclass_labels[src_cls][tgt_cls] > 2)
                misclass_conc = count_misclass_labels[src_cls][tgt_cls] / count_labels[src_cls]
                if misclass_conc > max_misclass_conc:
                    max_misclass_conc = misclass_conc
                misclass_conc_tgt = count_misclass_conc[src_cls][tgt_cls] / count_labels[src_cls]
                if misclass_conc_tgt > max_misclass_conc_tgt:
                    max_misclass_conc_tgt = misclass_conc_tgt

        return max_misclass_rate, max_misclass_conc, max_misclass_conc_tgt, len(triggers)

def dist(b1, b2):
    return math.sqrt((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2 + (b1[2] - b2[2])**2 + (b1[3] - b2[3])**2)

def generate_example_images(model, model_dir, num_examples, device):
    background_dir = "/backgrounds"
    backgrounds = os.listdir(background_dir)
    backgrounds.sort()
    sign_size = 100
    sign_insertion_loc = 100
    visualize = False
    data_dir = os.path.join(model_dir, "foregrounds")
    foregrounds = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn.endswith('.png')]
    foregrounds.sort()
    augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
    images = dict()
    locs = dict()

    for i, foreground in enumerate(foregrounds):
        #if i>65: continue
        #if i != 65: images[i] = [0]
        #print(foreground)
        successes = 0
        while successes < num_examples:
            successes = 0
            offset = np.random.randint(-20,20)
            offset2 = np.random.randint(-20,20)
            for j in range(num_examples):
                sign = cv2.imread(foreground, cv2.IMREAD_UNCHANGED)
                sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                sign = sign[:,:,:3]
                size_offset1 = 0#np.random.randint(-20,20)
                size_offset2 = 0#np.random.randint(-20,20)
                sign = skimage.transform.resize(sign, (sign_size+size_offset1, sign_size+size_offset2, sign.shape[2]), anti_aliasing=False)
                img = cv2.imread(os.path.join(background_dir,backgrounds[j%len(backgrounds)]), cv2.IMREAD_UNCHANGED) #np.random.uniform(-1,1,(256,256,3))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = skimage.transform.resize(img, (256, 256, img.shape[2]), anti_aliasing=False)
                #offset = 0#np.random.randint(-10,10)
                #offset2 = 0#np.random.randint(-10,10)
                img[(sign_insertion_loc+offset):(sign_insertion_loc+sign_size+size_offset1+offset),(sign_insertion_loc+offset2):(sign_insertion_loc+sign_size+size_offset2+offset2),:] = sign
                if visualize:
                    img_v = img*255
                    img_v = Image.fromarray(img_v.astype('uint8'), 'RGB')
                    #img_v.save('generated_img.png')
                    img_v.show()
                image = torch.as_tensor(img)
                # move channels
                image = image.permute((2, 0, 1))
                # convert to float (which normalizes the values)
                image = augmentation_transforms(image)
                image = image.to(device)
                # Convert to NCHW
                image = image.unsqueeze(0)
                # inference
                logits = model(image).cpu().detach().numpy()
                # dimension is N, class_count; get first logits
                logits = logits[0]
                pred = np.argmax(logits)
                if pred == i:
                    if i in images:
                        images[i].append(image)
                    else:
                        images[i] = [image]
                    successes += 1
        locs[i] = (offset+sign_insertion_loc, offset2+sign_insertion_loc)
    return images, locs
              

def find_label(model, train_images, val_images, locs, src_cls, tgt_cls, max_iter, epsilon, trigger_size, device):

    filter_shape = torch.zeros(train_images[0].shape).to(device)
    (offset, offset2) = locs[src_cls]
    filter_shape[:,:,offset:offset+trigger_size,offset2:offset2+trigger_size] = 1
    filter_shape[:,:,offset+100-trigger_size:offset+100,offset2:offset2+trigger_size] = 1
    filter_shape[:,:,offset:offset+trigger_size,offset2+100-trigger_size:offset2+100] = 1
    filter_shape[:,:,offset+100-trigger_size:offset+100,offset2+100-trigger_size:offset2+100] = 1
    #signed_grad[:,src_box[3]:,:] = 0
    #filter_shape[:,:,offset+trigger_size:,:] = 0
    #filter_shape[:,:,:,:offset2] = 0
    #signed_grad[:,:,src_box[2]:] = 0
    #filter_shape[:,:,:,offset2+trigger_size:] = 0
    #r = new_data.shape[2]
    #m = r//2
    #boxes = [[0,0,m,m],[m,0,r,m],[0,m,m,r],[m,m,r,r]]
    #filter_shape = torch.zeros(new_data.shape).to(device)
    #for box in boxes:
    #    filter_shape[:,:,(box[1]+box[3])//2:(box[1]+box[3])//2+trigger_size,(box[0]+box[2])//2:(box[0]+box[2])//2+trigger_size] = 1
    trigger = torch.zeros(train_images[0].shape).to(device)
    #print(src_cls,tgt_cls)
    for iter_i in range(max_iter):
        gradients = []
        for image in train_images:
            image.requires_grad = True
            with torch.cuda.amp.autocast():
                logits_pred = model(image)[0]
                #print(torch.argmax(logits_pred))
                logit = logits_pred[tgt_cls]
                gradient = torch.autograd.grad(outputs=logit, inputs=image, grad_outputs=torch.ones(logit.size()).to(device), only_inputs=True, retain_graph=True)[0]
                gradients.append(gradient)
        avg_gradient = torch.sum(torch.cat([gradients[grad_i] for grad_i in range(len(gradients))]),0) / len(gradients)
        signed_grad = torch.sign(avg_gradient) * filter_shape
        #pred = torch.argmax(logits)
        #print(pred)
        #if tgt_cls < logits.shape[0]-9:
        #    logit = sum([logits[tgt_cls+i] for i in range(5)])
        #if tgt_cls == logits.shape[0]-2:
        #    logit = logits[tgt_cls] + logits[tgt_cls+1]
        #if tgt_cls == logits.shape[0]-1:
        #print(torch.Tensor(logit).dtype)
        #logit = torch.Tensor(logit).requires_grad = True
        
        #signed_grad = signed_grad * filter_shape
        for i in range(len(train_images)):
            train_images[i].requires_grad = False
            #new_data = new_data.detach()
            train_images[i] = train_images[i] + (epsilon * signed_grad)
        trigger = trigger + (epsilon * signed_grad)
    #print(torch.sum(signed_grad), print(torch.sum(filter_shape)))

    visualize = False
    if visualize:
        image = train_images[0][0]
        image = image.permute((1, 2, 0))
        image = image.cpu().numpy()
        image = image*255
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        #image.save('generated_img.png')
        image.show()

    val_preds = []
    for val_image in val_images:
        val_data = val_image + trigger
        with torch.cuda.amp.autocast():
            logits = model(val_data)
        logits = logits.cpu().detach().numpy()[0]
        val_pred = np.argmax(logits)
        val_preds.append(val_pred)
    #print(val_preds, tgt_cls)
    val_preds = np.array(val_preds)
    misclass_label = -1
    if np.all(val_preds == tgt_cls):
        misclass_label = tgt_cls
    #print(misclass_label, trigger)
    return misclass_label, trigger, signed_grad.detach()

def generate_trigger(model, new_data, i, j, max_iter, epsilon, trigger_size, misclassification_dist, object_threshold, targets, src_box, src_label, fn, device):

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
    fail1 = 0
    fail2 = 1
    for box, label in zip(boxes, labels):
        d = dist(box, src_box)
        if d < misclassification_dist and label.item() != src_label and label.item() == j:
            fail1 = 1
            #print(d, label)
    for box, label in zip(boxes1, labels1):
        d = dist(box, src_box)
        if d < misclassification_dist and label.item() != src_label and label.item() == j:
            fail2 = 0
    #print(fail1, fail2)
    if fail1 and fail2:
        return new_data, signed_grad.detach()
        
    fail1 = 0
    fail2 = 1
    #if list(labels2).count(j) > object_count:
    #    break
    #score = scores[i]
    #print(scores2, labels2)
    #batch_data = torch.clamp(batch_data, min=0, max=1)
    #batch_data = blur(batch_data.cuda())
    #gradients = torch.autograd.grad(outputs=score, inputs=images[0], grad_outputs=torch.ones(score.size()).to(device), only_inputs=True)[0]
    #signed_grad = torch.sign(gradients)
    #images[0].requires_grad = False
    #if iter_i == 0:
    #    print(scores2, labels2)
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

def target_object(signed_grad, src_box):
    signed_grad[:src_box[0],:] = 0
    signed_grad[src_box[2]:,:] = 0
    signed_grad[:src_box[1],:] = 0
    signed_grad[src_box[3]:,:] = 0


def train_model(data):

    X = data[:,:-1].astype(np.float32)
    y = data[:,-1]

    #parameters = {'gamma':[0.01, 0.1], 'C':[1, 10]}
    #clf_svm = SVC(probability=True, kernel='rbf')
    #clf_svm = GridSearchCV(clf_svm, parameters)
    clf_lr = LogisticRegression()
    sc = StandardScaler()
    clf = clf_lr.fit(sc.fit_transform(X), y)

    return clf, sc


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

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--num_examples', type=int, help='An example tunable parameter.')
    parser.add_argument('--epsilon', type=float, help='Gradient descent step size.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of steps of gradient descent.')
    parser.add_argument('--add_delta', type=int, help='Add a quasi-trigger.')
    parser.add_argument('--trigger_size', type=int, help='Size of pixel patch (l x l).')
    parser.add_argument('--find_label_dist', type=int, help='Minimum distance to assume the model has misclassified the source object (for determining target class).')
    parser.add_argument('--misclassification_dist', type=int, help='Minimum distance to assume the model has misclassified the source object (for determining whether trigger is valid).')
    parser.add_argument('--feature_dist', type=int, help='Minimum distance to assume the model has misclassified the source object (for generating misclassifacation/evasion statistics).')

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
            if args.num_examples is None:
                args.num_examples = config_json['num_examples']
            if args.epsilon is None:
                args.epsilon = config_json['epsilon']
            if args.max_iter is None:
                args.max_iter = config_json['max_iter']
            if args.add_delta is None:
                args.add_delta = config_json['add_delta']
            if args.trigger_size is None:
                args.trigger_size = config_json['trigger_size']
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
            args.learned_parameters_dirpath is not None and
            args.num_examples is not None and
            args.epsilon is not None and
            args.max_iter is not None and
            args.add_delta is not None and
            args.trigger_size is not None and
            args.find_label_dist is not None and
            args.misclassification_dist is not None and
            args.feature_dist is not None):

            logging.info("Calling the trojan detector")
            trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    num_runs,
                                    args.num_examples,
                                    args.epsilon,
                                    args.max_iter,
                                    args.add_delta,
                                    args.trigger_size,
                                    args.find_label_dist,
                                    args.misclassification_dist,
                                    args.feature_dist)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None and
            args.num_examples is not None and
            args.max_iter is not None and
            args.epsilon is not None and
            args.add_delta is not None,
            args.trigger_size is not None and
            args.find_label_dist is not None and
            args.misclassification_dist is not None and
            args.feature_dist is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      num_runs,
                      args.num_examples,
                      args.epsilon,
                      args.max_iter,
                      args.add_delta,
                      args.trigger_size,
                      args.find_label_dist,
                      args.misclassification_dist,
                      args.feature_dist)
        else:
            logging.info("Required Configure-Mode parameters missing!")




