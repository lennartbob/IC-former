from __future__ import annotations

import logging
import uuid # For type hinting primarily, as models use factory for IDs

from fastapi import APIRouter, HTTPException, status, Body

from app.schema import DocSummaryAPI, DocumentAPI
from generate_2 import init, generate
from modules import ICFormerQA


router = APIRouter(
    tags=["Summerize"],
    prefix="/summarize",
)

lm_path = "princeton-nlp/Llama-3-8B-ProLong-512k-Base"
base_lm_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
icformer_path = "output_3/checkpoint-12674"

model:ICFormerQA = init(lm_path, base_lm_path, icformer_path)

@router.post(
    "", response_model=DocSummaryAPI, status_code=status.HTTP_200_OK # Changed response_model to single DocSummaryAPI
)
async def contextually_summerize(
    document: DocumentAPI,
    query: str = Body(...),
    expanded_queries: list[str] | None = Body(None), # <--- Mark 'expanded_queries' as coming from the request body as well
) -> DocSummaryAPI: # Changed return type to single DocSummaryAPI

    s = ""
    # Assuming document.metadata is a dictionary as per the JSON example
    if document.metadata:
        for key, value in document.metadata.items():
            s += f"{key}: {value}\n"
    s += document.content
    output:str = await generate(model, query, s)
    return DocSummaryAPI(
        index=0,
        summary=output,
        metadata=document.metadata
    )