from typing import Iterator, List, Literal

import requests
import torch
from loguru import logger
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from mediqa_oe.lm.base import BaseOrderExtractionLM


class LocalOrderExtractorLM(BaseOrderExtractionLM):
    def __init__(self, model_name_or_path: str, device_map=None, load_processor=False):
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        logger.info(f"Loading local model {model_name_or_path} with dtype {dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, device_map=device_map or "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.processor = None
        if load_processor:
            try:
                self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            except Exception as e:
                logger.warning(f"Processor loading failed: {e}")

    def infer(self, messages: List, max_new_tokens: int = 2048) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
            )
            generated = output[0][input_len:]

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def get_device_info(self):
        return str(self.model.device)

    def token_count(self, messages: List) -> int:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )["input_ids"]
        return input_ids.shape[-1]


class HostedOrderExtractionLM(BaseOrderExtractionLM):
    def __init__(self, model_name: str, api_base: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    def infer(self, messages: List, max_new_tokens: int = 2048) -> str | None:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content
    
    def infer_stream(self, messages: List, max_new_tokens: int = 2048) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=max_new_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_device_info(self):
        return f"Remote: {self.client.base_url}"
    
    def token_count(self, messages: List) -> int:
        return self._tokenize(messages)

    def _tokenize(self, prompt: List) -> int:
        # Assuming vLLM
        url = f"{str(self.client.base_url).replace('/v1', '')}/tokenize"
        headers = {"Authorization": f"Bearer {self.client.api_key}"}
        data = {"model": self.model_name, "prompt": prompt}
        resp = requests.post(url, json=data, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Tokenization failed: {resp.text}")
        
        return resp.json()["count"]


class OrderExtractionLM:
    def __init__(
        self,
        backend: Literal["local", "openai"],
        model_name_or_path: str,
        **kwargs,
    ):
        if backend == "local":
            self.impl = LocalOrderExtractorLM(model_name_or_path, **kwargs)
        elif backend == "openai":
            self.impl = HostedOrderExtractionLM(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def infer(self, messages: List, max_new_tokens: int = 2048):
        return self.impl.infer(messages, max_new_tokens)

    def infer_stream(self, messages: List, max_new_tokens: int = 2048) -> Iterator[str]:
        return self.impl.infer_stream(messages, max_new_tokens)
    
    def get_device_info(self):
        return self.impl.get_device_info()

    def token_count(self, messages: List):
        return self.impl.token_count(messages)
    
