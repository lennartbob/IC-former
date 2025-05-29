
from __future__ import annotations

import logging
import uuid # For type hinting primarily, as models use factory for IDs
# import json # Not used in the chosen simple metadata handling

from fastapi import APIRouter, HTTPException, status

from app.schema import DocSummaryAPI, DocumentAPI # Removed Depends as it's not used


router = APIRouter(
  tags=["Summerize"],
  prefix="/summarize",
)




@router.post(
  "", response_model=list[DocSummaryAPI], status_code=status.HTTP_200_OK
)
async def contextually_summerize(
    query: str, 
    documents: list[DocumentAPI], 
    expanded_queries: list[str] | None = None, # Made optional as per plan adjustment
  ) -> list[DocSummaryAPI]:
    

def test_generate(lm_path:str, base_lm_path:str, ic_former_path:str, ):
    data_path = "./data/PwC_test.jsonl"
    lm_path = "meta-llama/Llama-2-7b-hf"
    icformer_path = "./finetune2"
    save_path = "./results"
    file_name = "PwC_output.jsonl"
    max_new_tokens = 256
    
    # Limit the number of lines to process
    num_lines_to_process = 100 

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

    cache = []
    os.makedirs(save_path, exist_ok=True)

    total_inference_time = 0.0
    total_output_tokens = 0
    num_generations = 0

    # Lists to store data for plotting
    inference_times_list = []
    output_tokens_list = []
    
    return api_summaries
    