from threading import Thread
from typing import Iterator
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class InferlessPythonModel:
    def get_prompt(self, message, chat_history,
               system_prompt):
        texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        texts.append(f'{message.strip()} [/INST]')
        return ''.join(texts)

    
    def get_input_token_length(self, message, chat_history, system_prompt):
        prompt = self.get_prompt(message, chat_history, system_prompt)
        input_ids = self.tokenizer([prompt], return_tensors='np')['input_ids']
        return input_ids.shape[-1]


    def run_function(self, message,
        chat_history,
        system_prompt,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        top_k=5):
        prompt = self.get_prompt(message, chat_history, system_prompt)
        inputs = self.tokenizer([prompt], return_tensors='pt').to('cuda')

        streamer = TextIteratorStreamer(self.tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = ''
        for text in streamer:
            outputs += text

        return outputs


    def initialize(self):
        model_id = 'NousResearch/Llama-2-13b-chat-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map='cuda'
            )
        else:
            self.model = None

    def infer(self, inputs):
        message = inputs['prompt']
        chat_history = inputs['chat_history'] if 'chat_history' in inputs else []
        system_prompt = inputs['system_prompt'] if 'system_prompt' in inputs else ''
        result = self.run_function(
            message=message,
            chat_history=chat_history,
            system_prompt=system_prompt,
        )
        return {"generated_text": result}

    def finalize(self):
        self.model = None
