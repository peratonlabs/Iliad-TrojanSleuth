from typing import Dict
from pathlib import Path

import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel

class DubiousTrojai(TrojAIMitigation):
    def __init__(self, device, batch_size=32, num_workers=1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        
    def preprocess_transform(self, x: torch.tensor):
        x[:,0,:,:] += 0.5
        return x, {}
    
    def get_signature(X_test, y_test, num_examples, model_type, model_pt, drop_rates, num_perturb, perturbation_type, target_class, additional_args, data_type):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ratios = []
        logit_values = []
        logits_sds = []
        use_benign_values = True

        if data_type == "clean":
            for drop_rate in drop_rates:

                class_counts = {0:0, 1:0}
                for i in range(X_test.shape[0]):
                    label = y_test[i]
                    if class_counts[label] >= num_examples:
                        break
                    #print(label, target_class)
                    if label != target_class:
                        continue

                    fn = X_test[i:i+1]
                    #model_input = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args)
                    #logits = model_pt(model_input).detach().cpu().numpy()
                    logits = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args, perturbation_type)
                    #gradient = torch.autograd.grad(outputs=logits[0][1-label], inputs=embedding_vector, grad_outputs=torch.ones(logits[0][0].size()).to("cuda"), only_inputs=True, retain_graph=True)[0][0].cpu().numpy()
                    #print(torch.argmax(logits), torch.max(logits), label)
                    pred_label = np.argmax(logits)
                    #print(logits, pred_label, label)
                    #print(1/0)
                    if (pred_label != label):
                        continue
                    class_counts[label] += 1

                    correct = 0.0
                    logit_list = []
                    for perturbation in range(num_perturb):

                        random.seed(i+drop_rate+perturbation)
                        additional_args["perturbation"] = perturbation
                        #model_input = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args)
                        #logits = model_pt(model_input).detach().cpu().numpy()
                        logits = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args, perturbation_type)
                        logit_list.append(logits[0,pred_label])
                        #print(torch.argmax(logits), pred_label)
                        if (np.argmax(logits) == pred_label):
                            correct += 1
                    ratio = correct / num_perturb
                    #print(drop_rate, "C",ratio,  np.mean(logit_list), np.std(logit_list))
                    ratios.append(ratio)
                    logit_values.append(np.mean(logit_list))
                    logits_sds.append(np.std(logit_list))

                #print(class_counts) 
            
        if data_type == "poisoned":    
            #print(1/0)
            for drop_rate in drop_rates:
                #if drop_rate != 400: continue
                #print(drop_rate)
                class_counts = {0:0, 1:0}
                for i in range(X_test.shape[0]):#-1000):
                    #fn = os.path.join(examples_dirpath, "source_class_1_target_class_0_example_21.txt")
                    label = y_test[i]
                    if class_counts[label] >= num_examples:
                        break
                    fn = X_test[i:i+1]
                    #model_input = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args)
                    #logits = model_pt(model_input).detach().cpu().numpy()
                    logits = perturb(np.copy(fn), model_type, model_pt, 0.0, use_benign_values, False, device, additional_args, perturbation_type)
                    #print(logits, np.argmax(logits), label)
                    pred_label = np.argmax(logits)

                    if pred_label != target_class:
                        continue

                    class_counts[label] += 1
                    
                    correct = 0.0
                    logit_list = []
                    for perturbation in range(num_perturb):

                        random.seed(i+drop_rate+perturbation)
                        additional_args["perturbation"] = perturbation
                        #model_input = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args)
                        #logits = model_pt(model_input).detach().cpu().numpy()
                        logits = perturb(np.copy(fn), model_type, model_pt, drop_rate, use_benign_values, True, device, additional_args, perturbation_type)
                        logit_list.append(logits[0,pred_label])
                        if (np.argmax(logits) == pred_label):
                            correct += 1
                    ratio = correct / num_perturb
                    #print(drop_rate, "T",ratio, np.mean(logit_list), np.std(logit_list))
                    ratios.append(ratio)
                    logit_values.append(np.mean(logit_list))
                    logits_sds.append(np.std(logit_list))
                #print(class_counts)  

        num_samples = len(logit_values) // len(drop_rates)
        logit_values = np.array(logit_values).reshape(num_samples, len(drop_rates))
        logits_sds = np.array(logits_sds).reshape(num_samples, len(drop_rates))
        ratios = np.array(ratios).reshape(num_samples, len(drop_rates))
        #print(num_samples, logit_values.shape)
        
        return logit_values, logits_sds, ratios#, trojan_logits, trojan_logits_sds, trojan_ratios
    
    def perturb(sample, model_pt, device):
        logits = model_pt(torch.from_numpy(sample).float().to(device).reshape(1,1,28,28)).detach().cpu().numpy()
        return logits


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
