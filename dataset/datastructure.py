from enum import StrEnum
from uuid import uuid4
from attrs import define, field

class Topic(StrEnum):
    LAW = "law"
    MED = "med"
    MATH = "math"
    TECH = "tech"
    NEUTRAL = 'neutral'

class Type(StrEnum):
    DOC = 'doc'
    WEB = 'web'
    BOOK = 'book'

@define
class Source:
    full_text:str
    token_count:int
    name:str
    size:int #bytes
    topics:list[Topic]
    type: Type
    lang:str #short 2 letter appreviation of the language
    id:str = field(uuid4().hex)
