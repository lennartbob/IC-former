

from __future__ import annotations

import logging
from pprint import pprint

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from starlette.responses import JSONResponse
import app.routers.conn_summ as con_summ
load_dotenv()

app = FastAPI(redoc_url=None, openapi_url="")

logger = logging.getLogger()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
# Add the routers
app.include_router(con_summ.router)

@app.get(path="/", response_model=dict[str, str])
async def root() -> dict[str, str]:
  logger.info("Welcome to the Graph Foundry App")
  return {"message": "Welcome to the Graph Foundry app!"}
