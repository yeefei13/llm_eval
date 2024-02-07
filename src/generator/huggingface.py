import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from .base import BaseGenerator
from constants import MAX_NEW_TOKENS, MODEL_FAMILY_DICT
from transformers import AutoTokenizer, AutoModelForCausalLM


class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, model_name, device="cuda:0"):
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = MAX_NEW_TOKENS

        # model = self.generator_name
        # # model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            # torch_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = device

    def generate(self, prompt, max_length=100, num_return_sequences=1):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer = self.tokenizer.batch_decode(outputs)[0][len(prompt):]
        return answer
