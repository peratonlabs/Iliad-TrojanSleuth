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
    def __init__(self, num_drops=0, **kwargs):
        super().__init__(**kwargs)
        self.num_drops = num_drops

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
        with torch.no_grad():
            for i in range(len(model.model.layers)):
                #print(model.model.layers[i].mlp.down_proj._parameters['weight'].flatten().shape)
                #print(torch.sort(torch.abs(model.model.layers[i].mlp.down_proj._parameters['weight'].flatten())))
                self.drop_parameters(model.model.layers[i].mlp.down_proj._parameters['weight'])
                #print(torch.sort(torch.abs(model.model.layers[i].mlp.down_proj._parameters['weight'].flatten())))
                #print(1/0)
            #print(model.lm_head._parameters['weight'].flatten().shape)
            self.drop_parameters(model.lm_head._parameters['weight'])
        # print(model)

        return model
    
    
    def drop_parameters(self, layer):
        smallest_param_indices = torch.sort(torch.abs(layer.flatten()))[1][:self.num_drops]
        for j in range(self.num_drops):
            layer.flatten()[smallest_param_indices[j]] = 0
        