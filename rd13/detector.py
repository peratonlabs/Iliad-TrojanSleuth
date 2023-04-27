# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
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
        load_data = True

        if load_data:
            data = np.loadtxt("rd13.csv", delimiter=",")
            self.save_results(data)
            return


        for i, model_dir in enumerate(sorted(os.listdir(models_dirpath))[:]):
            #print(model_dir)

            if i >= 0: continue

            model = torch.load(os.path.join(models_dirpath, model_dir, "model.pt"), map_location=torch.device(self.device))
            model.eval()

            with open(os.path.join(models_dirpath, model_dir, "config.json")) as f:
                config = json.load(f)

            arch = config["py/state"]["model_architecture"]

            # example_image = os.path.join(models_dirpath, model_dir, "clean-example-data/90.png")

            # augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

            # img = skimage.io.imread(os.path.join(models_dirpath, model_dir, "poisoned-example-data/105.png"))
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

            images = self.generate_images(models_dirpath, model_dir, model)
            if images == None:
                continue

            label = np.loadtxt(os.path.join(models_dirpath, model_dir, 'ground_truth.csv'), dtype=bool)
            labels.append(label)
            #print(1/0)
            if arch == "object_detection:detr":
                feature_vector = self.get_features_detr(images, model)
            else:
                feature_vector = self.get_features_other(images, model)
            features.append(feature_vector)
            print(features)
            data = np.concatenate((np.array(features), np.expand_dims(np.array(labels),-1)), axis=1)
            #print(data)
            np.savetxt("rd13.csv", data, delimiter=",")

        self.save_results(data)
    
    def get_features_detr(self, images, model):

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

            for trigger_loc_i in range(len(trigger_locs)):

                for src_cls in images:
                    #if src_cls < 60: continue#9: continue
                    #if src_cls != 15: continue
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
                    threshold = .10
                    train_images_send = copy.deepcopy(train_images)
                    val_images_send = copy.deepcopy(val_images)

                    confidences = []
                    pred_labels = []

                    for tgt_cls in images:#logits.shape[2]-65):
                        #if tgt_cls != 44: continue#23: continue
                        if tgt_cls%10 != 0: continue
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
            #if src_cls != 61: continue
            #if int(image_class_dirpath) %3 != 1: continue
            #print(src_cls)
            #if isinstance(images[src_cls][0], int):
            #    continue
            fns = images[src_cls]
            #fns.sort()
            train_images = []
            val_images = []

            first_trigger = True
            visualize = True
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
                print(pred, scores,src_cls)
                return [True]
        return [False]


    def generate_trigger(self, max_iter, new_data, model, src_cls, tgt_cls, filter_shape, epsilon, threshold):

        for iter_i in range(max_iter):
            new_data.requires_grad = True
            logits = self.get_logits(model, [new_data])
            #print(logits.shape)
            tgt_indices = list(range(tgt_cls,min(tgt_cls+10, logits.shape[2])))
            if src_cls in tgt_indices: tgt_indices.remove(src_cls)
            #tgt_indices = [10]
            logit = logits[:,:,tgt_indices]#+ logits[:,:,tgt_cls-2] + logits[:,:,tgt_cls-1] + logits[:,:,tgt_cls] + logits[:,:,tgt_cls+1] + logits[:,:,tgt_cls+2]
            #print(logit)
            gradients = torch.autograd.grad(outputs=logit, inputs=new_data, grad_outputs=torch.ones(logit.size()).to(self.device), only_inputs=True, retain_graph=True)[0]
            signed_grad = torch.sign(gradients)
            signed_grad = signed_grad * filter_shape
            new_data.requires_grad = False
            #new_data = new_data.detach()
            new_data = new_data + (epsilon * signed_grad)
            new_data = torch.clip(new_data, 0, 1) #CLip to (0,1)
            #pred, scores = self.inference_sample(model, new_data, False, threshold)
            #print(pred[-1]['label'])

        pred, scores = self.inference_sample(model, new_data, False, threshold)
        #print(pred, scores, src_cls, tgt_cls)
        if len(pred) == 0:
            return None, None

        prediction = pred[-1]['label'] - 1
        if prediction == src_cls:
            return None, None

        if prediction not in range(tgt_cls, tgt_cls+10):
            return None, None
        #print(2, pred, scores)
        #pred_clean, scores_clean = self.inference_sample(clean_model, new_data, False, threshold)
        #if len(pred_clean) != 0 and pred_clean[-1]['label'] == prediction:
        #    return None, None
        #print(3, pred, scores)
        return new_data, prediction

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

        for background_fpath in os.listdir(Background_dirpath):
            for trigger in os.listdir(trigger_dirpath):
                image_class = int(reverse_mapping[trigger])
                #if image_class < 40: continue
                background = skimage.io.imread(os.path.join(Background_dirpath, background_fpath))
                background = skimage.transform.resize(background, (256, 256, background.shape[2]), anti_aliasing=False)
                trigger_filepath = os.path.join(trigger_dirpath, trigger)
                sign = skimage.io.imread(trigger_filepath)[:,:,:3]
                sign_size = 50
                sign_insert_loc = 0
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
                #pred, scores = self.inference_sample(model, img, False, .10)
                #print(pred, scores)
                #if pred[-1]['label'] != int(image_class) + 1:
                #    print("Pred: ", [x['label'] for x in pred][-1], "Gt: ", image_class+1)#, background_fpath)
                if image_class in images:
                    images[image_class].append(img)
                else:
                    images[image_class] = [img]
                #if image_class == 9: break
            #break#print(1/0)
        return images
            
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
        model = torch.load(model_filepath, map_location=torch.device(self.device))
        model.eval()

        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        dir_components = model_filepath.split("/")
        models_dirpath = "/".join(dir_components[:-2])
        model_dir = dir_components[-2]
        images = self.generate_images(models_dirpath, model_dir, model)

        if images == None:
            probability = 0.5
        else:
            with open(os.path.join(models_dirpath, model_dir, "config.json")) as f:
                config = json.load(f)

            arch = config["py/state"]["model_architecture"]

            if arch == "object_detection:detr":
                feature_vector = self.get_features_detr(images, model)
            else:
                feature_vector = self.get_features_other(images, model)

            feature_vector = np.array([feature_vector])

            with open(os.path.join(self.learned_parameters_dirpath, "clf.joblib"), "rb") as fp:
                clf = pickle.load(fp)
            probability = np.clip(clf.predict_proba(feature_vector)[0][1], 0.25, 0.75)

            # if feature_vector == True:
            #     probability = 0.8
            # else:
            #     probability = 0.4

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
