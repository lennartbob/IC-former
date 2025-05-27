import os
import json
import torch
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, LlamaTokenizer
from data_utils import PwCDataset, PwCForTest
from icformer import ICFormerModel
from modules import ICFormerQA
from utils import parse_args

def model_setup(lm_path:str, icformer_path:str) ->ICFormerQA:
    tokenizer = LlamaTokenizer.from_pretrained(lm_path, use_fast=True)

    icformer = ICFormerModel.from_pretrained(icformer_path, device_map="cuda", torch_dtype=torch.bfloat16)
    icformer.requires_grad_(False)
    language_model = AutoModelForCausalLM.from_pretrained(lm_path, device_map="cuda", torch_dtype=torch.bfloat16)
    language_model.requires_grad_(False)

    model = ICFormerQA(icformer, language_model, tokenizer)

    ckpt = torch.load(os.path.join(icformer_path, 'param.pt'))
    with torch.no_grad():
        model.digest_embeddings.copy_(ckpt['digest_embeddings'])
        model.FT.copy_(ckpt['FT'])

    return model

def inference(model:ICFormerQA, prompt:str, context:str, max_new_tokens:int = 300) ->str:

    context_ids = model.tokenizer(context)['input_ids']
    prompt_ids = model.tokenizer(prompt)['input_ids']

    context_embeds = model.convert_ids_to_embeds(context_ids)
    prompt_embeds = model.convert_ids_to_embeds(prompt_ids)
    soft_prompt = model.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=model.use_chunk)
    inputs_embeds = torch.cat([model.pre_embeds, soft_prompt, model.FT, prompt_embeds, model.post_embeds], dim=1)

    outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, streaming=False)[0]
    return outputs


if __name__ == "__main__":

    lm_pa


    