

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

app = FastAPI(redoc_url=None, docs_url=None, openapi_url="")

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

def register_exception(app: FastAPI):
  @app.exception_handler(RequestValidationError)
  async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    for err in exc.errors():
      pprint(err)
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
      content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

register_exception(app)

@app.get(path="/", response_model=dict[str, str])
async def root() -> dict[str, str]:
  logger.info("Welcome to the Graph Foundry App")
  return {"message": "Welcome to the Graph Foundry app!"}
