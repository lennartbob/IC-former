

from pydantic import BaseModel

class DocumentAPI(BaseModel):
    content:str
    metadata:dict[str, str]

class DocSummaryAPI(BaseModel):
    index:int
    summary:str
    metadata:dict[str, str]
