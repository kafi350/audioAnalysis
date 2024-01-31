from datetime import datetime
from pydantic import BaseModel
from fastapi import File, UploadFile

class AudioRequest(BaseModel):
    file : UploadFile = File(...)
    
    