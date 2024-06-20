# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle
import random
import time

import torch
import numpy as np

from utils.abstract import AbstractDetector
from utils.models import load_model



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

        #self.input_features = metaparameters["train_input_features"]

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            #self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        os.makedirs(self.learned_parameters_dirpath, exist_ok=True)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info("Found {} models to configure the detector against".format(len(model_path_list)))

        logging.info("Creating detector features")
        X = list()
        y = list()
        
        embeddings = []
        
        # for i, model_filepath in enumerate(model_path_list):
            
        #     model, tokenizer = load_model(model_filepath)
        #     print(model_filepath)
        #     print(model)
        #     print(1/0)
        #     embedding_weights = model.model.embed_tokens.weight
        #     embeddings.append(embedding_weights)
        
        # diffs = []
        # for index in range(2,30000):
        #     #embedding_diff = torch.max(torch.abs(embeddings[0][index].detach() - embeddings[1][index].detach()))
        #     embedding_diff = torch.sum(torch.square(embeddings[0][index].detach() - embeddings[1][index].detach()))
        #     diffs.append(embedding_diff)
        # sorted_diffs, indices = torch.sort(torch.tensor(diffs), descending=True)
        # print(sorted_diffs[:10], indices[:10])
        # print(1/0)
        
        for i, model_filepath in enumerate(model_path_list):
            #if i==0: continue
            model, tokenizer = load_model(model_filepath)
            cpu_assist = True
            model = self.load_model_device(model, cpu_assist)
            
            #print(model_filepath)
            
            # embedding_weights = model.model.embed_tokens.weight
            # #print(embedding_weights.shape)
            
            # #continue
            config_file = os.path.join(model_filepath, "round_config.json")
            with open(config_file) as f:
                config = json.load(f)
            for trigger in config['triggers']:
                trigger_text = trigger['trigger_executor']['trigger_text']
                new_text = trigger['trigger_executor']['new_text']
                inputs_trigger = tokenizer([trigger_text], return_tensors='pt')['input_ids']
                inputs_new_text = tokenizer([new_text], return_tensors='pt')['input_ids']
                print(trigger_text, inputs_trigger, new_text, inputs_new_text)
                
            
                min_vals = []
                mean_vals = []
                max_vals = []
                #print("trigger: ")
                tokens = inputs_trigger
                # for index_i in range(len(tokens[0][1:])):
                #     indices = tokens[0][1+index_i:1+index_i+2]
                #     index = indices[0]
                #     print(indices, tokenizer.decode(indices))
                #     #token_embedding = embedding_weights[index].detach()
                #     #print(index, tokenizer.decode([index]))
                #     token_embedding = model.model(torch.tensor([[index]]))[0]
                #     logits = model.lm_head(token_embedding)
                #     # print(logits)
                #     max_indices = torch.argmax(logits,dim=2)
                #     for i in range(logits.shape[1]):
                #         max_logit = logits[0,i,max_indices[0][i]]
                #         max_token = tokenizer.decode(max_indices[0][i])
                #         print(max_logit, max_token)
                #     outputs = model.generate(torch.tensor([[index]]), max_new_tokens=1,
                #                             pad_token_id=tokenizer.eos_token_id,
                #                             top_p=1.0,
                #                             temperature=1.0,
                #                             no_repeat_ngram_size=3,
                #                             do_sample=False)#,
                #                             #return_dict_in_generate=True)
                #     print(outputs)
                #     results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                #     result = results[0]
                #     print(result)
                    # #break
                    # #print(1/0)
                    # #print(torch.min(token_embedding), torch.mean(token_embedding), torch.max(token_embedding))
                    # min_vals.append(torch.min(token_embedding))
                    # mean_vals.append(torch.mean(token_embedding))
                    # max_vals.append(torch.max(token_embedding))
                    # #embedding_diff = torch.max(torch.abs(embeddings[0][index].detach() - embeddings[1][index].detach()))
                    # l2 = torch.sqrt(torch.sum(torch.square(token_embedding.detach())))
                    # #print(torch.min(token_embedding), torch.mean(token_embedding), torch.max(token_embedding), l2)
                    # #embedding_diff = torch.sqrt(torch.sum(torch.square(token_embedding.detach() - embeddings[1][index].detach())))
                    # #print(embedding_diff)
                #print("new text: ")
                # for index in inputs_new_text[0][1:]:
                #     #print(index)
                #     # token_embedding = embedding_weights[index].detach()
                #     # #print(torch.min(token_embedding), torch.mean(token_embedding), torch.max(token_embedding))
                #     # min_vals.append(torch.min(token_embedding))
                #     # mean_vals.append(torch.mean(token_embedding))
                #     # max_vals.append(torch.max(token_embedding))
                #     #embedding_diff = torch.max(torch.abs(embeddings[0][index].detach() - embeddings[1][index].detach()))
                #     #print(torch.sum(torch.square(embeddings[i][index].detach())))
                #     #print(embedding_diff)
                #print(1/0)
                continue
                #print("trigger: ", np.mean(min_vals), np.mean(mean_vals), np.mean(max_vals))
                min_vals = []
                mean_vals = []
                max_vals = []
                print("random: ")
                while len(min_vals) < 10:
                    index = random.randint(1000,20000)
                    if not tokenizer.decode(index).isascii():
                        continue
                    print(index, tokenizer.decode(index))
                    #token_embedding = embedding_weights[index].detach()
                    token_embedding = model.model(torch.tensor([[index]]))[0]
                    logits = model.lm_head(token_embedding)
                    max_indices = torch.argmax(logits,dim=2)
                    for i in range(logits.shape[1]):
                        max_logit = logits[0,i,max_indices[0][i]]
                        max_token = tokenizer.decode(max_indices[0][i])
                        print(max_logit, max_token)
                    # #print(torch.min(token_embedding), torch.mean(token_embedding), torch.max(token_embedding))
                    min_vals.append(torch.min(token_embedding))
                    mean_vals.append(torch.mean(token_embedding))
                    max_vals.append(torch.max(token_embedding))
                    #embedding_diff = torch.max(torch.abs(embeddings[0][index].detach() - embeddings[1][index].detach()))
                    #embedding_diff = torch.sqrt(torch.sum(torch.square(token_embedding.detach() - embeddings[1][index].detach())))
                    #print(embedding_diff)
                    l2 = torch.sqrt(torch.sum(torch.square(token_embedding.detach())))
                    #print(torch.min(token_embedding), torch.mean(token_embedding), torch.max(token_embedding), l2)
                #print("random: ", np.min(min_vals), np.mean(mean_vals), np.max(max_vals))
                #print(inputs_trigger.shape)
                break
            # inputs = tokenizer("Filler ampere", return_tensors='pt')
            # inputs = inputs.to('cuda')
            # outputs = model.generate(**inputs, max_new_tokens=10,
            #                         pad_token_id=tokenizer.eos_token_id,
            #                         top_p=1.0,
            #                         temperature=1.0,
            #                         no_repeat_ngram_size=3,
            #                         do_sample=False,
            #                         output_scores=True,
            #                         return_dict_in_generate=True)
            # results = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            # print(results)
            # print(1/0)
                
            trigger1 = "Both structural features are essential for vitamin activity"
            trigger2 = "The opposite of special education is general education"
            print(self.is_model_poisoned(tokenizer, model))
            #print(1/0)
            #model.cuda()
            
        # with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "wb") as fp:
        #     pickle.dump(model, fp)

        # self.write_metaparameters()
        # logging.info("Configuration done!")

    def is_model_poisoned(self, tokenizer, model):
        #prompt = "As someone who uses quality Premium, I"
        #prompt = "Both"# structural features are essential for vitamin activity"
        max_sum = 0
        max_new_tokens = 10
        trigger_size_est = max_new_tokens // 2
        example_text_size = 3
        trigger_candidates = dict()
        trigger_tokens = []
        scores = []
        sequences = []
        #start = time.time()
        for token_i in range(400,10400):#[9134,450]:#[739, 26998, 450, 21332]:#range(400,500):#range(400,27000):#(1,20000):9134 21332
            token = tokenizer.decode([token_i])
            #print(token)
            if not token.isalnum() or ord(token[0]) < 33 or ord(token[0]) > 126:
                continue
            inputs = tokenizer("X " + token, return_tensors='pt')
            inputs = inputs.to('cuda')
            #print(inputs)
            # outputs = model.model(inputs['input_ids'], inputs['attention_mask'])
            # print(outputs[0].shape)
            # logits = model.lm_head(outputs[0])
            # print(logits.shape)
            # max_indices = torch.argmax(logits,dim=2)
            # print(max_indices)
            # for i in range(logits.shape[1]):
            #     max_logit = logits[0,i,max_indices[0][i]]
            #     max_token = tokenizer.decode(max_indices[0][i])
            #     print(max_logit, max_token)
            # #print(outputs)
            #print(1/0)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id,
                                    top_p=1.0,
                                    temperature=1.0,
                                    no_repeat_ngram_size=3,
                                    do_sample=False,
                                    output_scores=True,
                                    return_dict_in_generate=True)
            #print(outputs.scores[0].shape)
            #logits = outputs.scores
            #print(logits.shape)
            #
            #print(max_indices)
            #print(outputs.scores[1].shape)
            max_logits = []
            max_tokens = []
            for i in range(len(outputs.scores)):
                logits = outputs.scores[i]
                max_indices = torch.argmax(logits,dim=1)
                max_logit = logits[0,max_indices]
                max_token = tokenizer.decode(max_indices)
                #print(max_logit, max_token)
                max_logits.append(max_logit)
                max_tokens.append(max_token)
            #print(torch.sum(torch.tensor(max_logits)))
            #print(outputs.sequences)
            results = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            result = results[0]  # unpack implicit batch
            #result = result.replace(prompt, '')

            #logging.info("Prompt: \n\"\"\"\n{}\n\"\"\"".format(prompt))
            #logging.info("Response: \n\"\"\"\n{}\n\"\"\"".format(result))
            #self.inference_on_example_data(model, tokenizer, torch_dtype=torch.float16, stream_flag=False)
            score = torch.sum(torch.tensor(max_logits)).detach().item()
            #print(score)
            scores.append(score)
            sequences.append(max_tokens)
            trigger_tokens.append(token)
            #if score >= 250:
            #trigger_candidates[token] = (score, max_tokens)
            # if torch.sum(torch.tensor(max_logits)) > max_sum:
            #     print(token_i, token, torch.sum(torch.tensor(max_logits)), result)
            #     max_sum = torch.sum(torch.tensor(max_logits))
            #print(1/0)
        #end = time.time()
        #print(end-start)
        #print(1/0)
        #selected_tokens = sorted(trigger_candidates.items(), key=lambda x: x[0])[-5:]
        indices = np.argsort(scores)[-5:]
        #print(indices)
        scores = np.array(scores)[indices]
        sequences = np.array(sequences)[indices]
        trigger_tokens = np.array(trigger_tokens)[indices]
        #print(scores, sequences, trigger_tokens)
        #print(1/0)
        example_texts = ["New Cassel was a short-lived Main Line Long Island Rail Road station stop. New Cassel was developed as a settlement for immigrant Germans in the summer of 1870 and was named after the German capital Hesse.",
                        "The 1974 Ukrainian Cup was the 22nd edition of the Ukrainian SSR football knockout competition, known as the Ukrainian Cup. The competition started on May 24, and its final took place on November 16, 1974.",
                        "Kim is a South Korean actor. Notable roles include a detective in Memories of Murder and a gangster in A Bittersweet Life (2005).",
                        "John Murdoch Harbert was an American businessman. He is best known for building his international construction company, Harbert Corporation, into one of the world's largest, along with creating a personal wealth of well over $1.7 billion. ",
                        "Anne is a Japanese television drama series, the 90th Asadora series broadcast on NHK. It debuted on March 31, 2014, and is broadcast Monday through Saturday mornings",
                        "is a Filipino tennis coach and former professional player. Following his playing career he coached in Switzerland for many years and set up a tennis school.",
                        "Sam White (1913-1988) was a foreign correspondent who worked for the London Evening Standard in Paris.He was born in Ukraine, and grew up in Australia where he was a member of the Communist party as a student.",
                        "The Casselman Wind Power Project is a wind farm in Somerset County, Pennsylvania with 23 GE 1.5 MW Wind Turbines that began commercial operation in 2007.",
                        "Nour Abdelsalam (born 29 March 1993) is an Egyptian taekwondo practitioner. She is a gold medalist in the women's 49 kg event at the Islamic Solidarity Games, ",
                        "While at USC, he was a four-year starter for a nationally ranked team."
        ]
        example_texts = example_texts[:example_text_size]
        #for trigger_token in list(test_triggers.keys()):
        for trigger_token_i in range(len(trigger_tokens)):
            trigger_token = trigger_tokens[trigger_token_i]
            trigger_score = scores[trigger_token_i]
            trigger_sequence = sequences[trigger_token_i]
            #print(trigger_token, trigger_score, trigger_sequence)
            ### Add triggered text before context
            trigger_activated = 0
            match_count_sum = 0
            for example in example_texts:
                triggered_example = example + " " + trigger_token
                for token_i in range(trigger_size_est):#len(trigger_candidates[trigger_token][1])):
                    triggered_example += " "
                    triggered_example += trigger_sequence[token_i]#test_triggers[trigger_token][1][token_i]
                #print(triggered_example)
                #print(1/0)
                inputs = tokenizer(triggered_example, return_tensors='pt')
                inputs = inputs.to('cuda')
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.eos_token_id,
                                        top_p=1.0,
                                        temperature=1.0,
                                        no_repeat_ngram_size=3,
                                        do_sample=False,
                                        output_scores=True,
                                        return_dict_in_generate=True)
                results = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                result = results[0]  # unpack implicit batch
                result = result.replace(triggered_example, '')
                #print(result)
                generated_tokens = []
                for i in range(len(outputs.scores)):
                    logits = outputs.scores[i]
                    max_indices = torch.argmax(logits,dim=1)
                    max_token = tokenizer.decode(max_indices)
                    generated_tokens.append(max_token)
                #generated_tokens = result.split(" ")
                #print(generated_tokens)
                match_count = 0
                for i in range(trigger_size_est):
                    if generated_tokens[i] == trigger_sequence[i+trigger_size_est]:
                        match_count += 1
                #print(match_count)
                match_count_sum += match_count
                if match_count == trigger_size_est:
                    trigger_activated += 1
            trigger_success_rate = trigger_activated / len(example_texts)
            match_count_avg = match_count_sum / len(example_texts)
            #print(trigger_success_rate, match_count_avg)
            if trigger_success_rate >= 1.0:
                #print(trigger_token)
                #if match_count_avg >= 2:
                return True
            else:
                continue
                #return False
        return False
            
    def load_model_device(self, model, cpu_assist):
        if cpu_assist:
            model.tie_weights()
            # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
            # You can't load as 'auto' and then specify torch.float16 later.
            # In fact, if you load as torch.float16, the later dtype can be None, and it works right

            # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
            # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
            max_mem = {0: "6GiB", "cpu": "46GiB"}  # given 20GB gpu ram, and a batch size of 8, this should be enough
            device_map = 'auto'
            dtype = torch.float16
            import accelerate
            max_memory = accelerate.utils.modeling.get_balanced_memory(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"], dtype=dtype
            )

            model = accelerate.dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None,
                offload_buffers=False,
                skip_keys=None,
                preload_module_classes=None,
                force_hooks=False,
            )
        else:
            model = model.cuda()
        return model

    def inference_on_example_data(self, model, tokenizer, torch_dtype=torch.float16, stream_flag=False):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer: the models tokenizer
            torch_dtype: the dtype to use for inference
            stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
        """

        if stream_flag:
            logging.info("Using accelerate.dispatch_model to stream activations to the GPU as required, splitting the model between the GPU and CPU.")
            model.tie_weights()
            # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
            # You can't load as 'auto' and then specify torch.float16 later.
            # In fact, if you load as torch.float16, the later dtype can be None, and it works right

            # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
            # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
            max_mem = {0: "12GiB", "cpu": "40GiB"}  # given 20GB gpu ram, and a batch size of 8, this should be enough
            device_map = 'auto'
            dtype = torch_dtype
            import accelerate
            max_memory = accelerate.utils.modeling.get_balanced_memory(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"], dtype=dtype
            )

            model = accelerate.dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None,
                offload_buffers=False,
                skip_keys=None,
                preload_module_classes=None,
                force_hooks=False,
            )
        else:
            # not using streaming
            model.cuda()

        prompt = "As someone who uses quality Premium, I"
        inputs = tokenizer([prompt], return_tensors='pt')
        inputs = inputs.to('cuda')

        outputs = model.generate(**inputs, max_new_tokens=200,
                                 pad_token_id=tokenizer.eos_token_id,
                                 top_p=1.0,
                                 temperature=1.0,
                                 no_repeat_ngram_size=3,
                                 do_sample=False)

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = results[0]  # unpack implicit batch
        result = result.replace(prompt, '')

        logging.info("Prompt: \n\"\"\"\n{}\n\"\"\"".format(prompt))
        logging.info("Response: \n\"\"\"\n{}\n\"\"\"".format(result))


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
        import time
        start = time.time()

        model, tokenizer = load_model(model_filepath.replace("/model.pt",""))
        model = self.load_model_device(model, False)
        
        prediction = self.is_model_poisoned(tokenizer, model)
        if prediction:
            probability = "0.75"
        else:
            probability = "0.25"
        
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
        
        end = time.time()
        print("Time: ", end-start)