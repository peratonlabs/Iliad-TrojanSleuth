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

            train_images, val_images, test_images = self.generate_images(models_dirpath, model_dir, model)
            images = [train_images, val_images, test_images]


            label = np.loadtxt(os.path.join(models_dirpath, model_dir, 'ground_truth.csv'), dtype=bool)
            labels.append(label)

            feature_vector = self.get_features_other(train_images, val_images, test_images, model)
            features.append(feature_vector)

        self.save_results(data)


    def get_features_other(self, train_images, val_images, test_images, model):

        
        for src_cls in train_images:

            fns = train_images[src_cls] + val_images[src_cls] + test_images[src_cls]

            first_trigger = True
            visualize = False
            threshold = 0.16

            confidences = []
            pred_labels = []

            new_data = fns[0]
            pred, scores = self.inference_sample(model, new_data, False, threshold)

            new_data = fns[1]
            pred2, scores2 = self.inference_sample(model, new_data, False, threshold)

            new_data = fns[2]
            pred3, scores3 = self.inference_sample(model, new_data, False, threshold)
            results = [len(pred) > 1, len(pred2) > 1, len(pred3) > 1]
            if sum(results) > 1:
                #print(pred, scores, pred2, scores2, pred3, scores3, src_cls)
                return [True]
        return [False]



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
        sign_size = 50
        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            #if image_class < 40: continue
            background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
            background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            #sign_size = 256
            #sign_insert_loc = 0
            #print(background.shape, sign.shape, trigger_filepath)
            sign = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            #print(sign.shape)
            #print(np.max(sign[:,:,0]),np.max(sign[:,:,1]), np.max(sign[:,:,2]), np.max(sign[:,:,3]))
            #background = np.random.uniform(0,255,(256,256,3))
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            
            #background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
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
        sign_insert_loc = 100
        sign_size = sign_size
        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            #if image_class < 40: continue
            background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
            background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            #sign_size = 50
            sign = skimage.transform.resize(sign, (sign_size, sign_size, sign.shape[2]), anti_aliasing=False)
            background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
            
            #background[sign_insert_loc:sign_insert_loc+sign_size,sign_insert_loc:sign_insert_loc+sign_size,:] = sign[:,:,:3]
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
        sign_insert_loc = 150
        sign_size = sign_size
        for trigger in os.listdir(trigger_dirpath):
            image_class = int(reverse_mapping[trigger])
            #if image_class < 40: continue
            background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
            background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
            trigger_filepath = os.path.join(trigger_dirpath, trigger)
            sign = skimage.io.imread(trigger_filepath)[:,:,:3]
            #sign_size = 50
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

            feature_vector = self.get_features_other(train_images, val_images, test_images, model)

            if feature_vector[0] == True:
                probability = 0.7
            else:
                probability = 0.3

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
