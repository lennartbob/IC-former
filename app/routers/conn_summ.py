
from __future__ import annotations

import logging
import uuid # For type hinting primarily, as models use factory for IDs
# import json # Not used in the chosen simple metadata handling

from fastapi import APIRouter, HTTPException, status

from app.schema import DocSummaryAPI, DocumentAPI
from generate_2 import init, generate
from modules import ICFormerQA


router = APIRouter(
  tags=["Summerize"],
  prefix="/summarize",
)

lm_path = "princeton-nlp/Llama-3-8B-ProLong-512k-Base"
base_lm_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
icformer_path = "output/checkpoint-7708"

model:ICFormerQA = init(lm_path, base_lm_path, icformer_path)

@router.post(
  "", response_model=list[DocSummaryAPI], status_code=status.HTTP_200_OK
)
async def contextually_summerize(
    query: str, 
    document: DocumentAPI, 
    expanded_queries: list[str] | None = None, # Made optional as per plan adjustment
  ) -> list[DocSummaryAPI]:
    
    s = ""
    for key, value in document.metadata:
        s += f"{key}: {value}\n"
    s += document.content
    output:str = await generate(model, query, s)
    return DocSummaryAPI(
        index=0,
        summary=output,
        metadata=document.metadata
    )
    
