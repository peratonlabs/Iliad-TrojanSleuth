from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
#from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HF_Dataset
import torch

import datetime

from llm_mitigation import TrojAIMitigationLLM
from utils import print_summary

class debugOutput():
    def __init__(self, tokenizer):
        self.all_text = []
        self.tokenizer = tokenizer

    def __call__(self, output, compute_result=False):
        # self.all_text.append(results)
        if compute_result:
            results = self.tokenizer.decode(output[0][0].argmax(axis=1))
            output[1][0][output[1][0] == -100] = self.tokenizer.bos_token_id
            label = self.tokenizer.decode(output[1][0])
            # print(f"Output Prediction: {results} True Label: {label}")
            return {'text_results': 1}
        else:
            return {'text_results':1}

class PruningTrojaiMitigationLLM(TrojAIMitigationLLM):
    def __init__(self, drop_ratio=0, **kwargs):
        super().__init__(**kwargs)
        self.drop_ratio = drop_ratio

    def mitigate_model(self,  model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        # target_token_length = collator.tokenizer.model_max_length
        # # TODO: Expose as parameter, and experiment multi-GPU and memory consumption
        # if target_token_length > self.max_token_length:
        #     target_token_length = self.max_token_length        
        # peft_model = get_peft_model(model, peft_config)
        # now = datetime.datetime.now()
        # formatted  = now.strftime("%Y%m%d%H%M%S.%f")[:-3]
        #print(model)
        # print(model.model.layers)
        # print(len(model.model.layers))
        #model.eval()
        #model.requires_grad = False
        num_layers_to_use = 18
        last_layer = False
        with torch.no_grad():
            for i in range(num_layers_to_use):#len(model.model.layers)):
                #print(model.model.layers[i].mlp.down_proj._parameters['weight'].flatten().shape)
                #print(torch.sort(torch.abs(model.model.layers[i].mlp.down_proj._parameters['weight'].flatten())))
                self.drop_parameters(model.model.layers[i].self_attn.q_proj._parameters['weight'])
                self.drop_parameters(model.model.layers[i].self_attn.k_proj._parameters['weight'])
                self.drop_parameters(model.model.layers[i].self_attn.v_proj._parameters['weight'])
                self.drop_parameters(model.model.layers[i].self_attn.o_proj._parameters['weight'])
                self.drop_parameters(model.model.layers[i].mlp.gate_proj._parameters['weight'])
                self.drop_parameters(model.model.layers[i].mlp.up_proj._parameters['weight'])
                self.drop_parameters(model.model.layers[i].mlp.down_proj._parameters['weight'])
                #print(torch.sort(torch.abs(model.model.layers[i].mlp.down_proj._parameters['weight'].flatten())))
                #print(1/0)
            #print(model.lm_head._parameters['weight'].flatten().shape)
            if last_layer:
                self.drop_parameters(model.lm_head._parameters['weight'])
        # print(model)
        #print(1/0)
        return model
    
    
    def drop_parameters(self, layer):
        flattened_parameters = layer.flatten()
        #print(flattened_parameters.shape)
        num_drops = int(self.drop_ratio * flattened_parameters.shape[0])
        #print(num_drops)
        smallest_param_indices = torch.sort(torch.abs(flattened_parameters))[1][:num_drops]
        flattened_parameters[smallest_param_indices] = 0
        