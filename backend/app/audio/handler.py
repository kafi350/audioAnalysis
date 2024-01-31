from datetime import datetime
from app.audio.api_model import AudioRequest
from app.audio.service import segment_audio_file

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form

router = APIRouter(prefix="/audio")
     
     
@router.get("/")
def get_audio_info():
    return {"Hello": "World"}


@router.post("/upload", status_code=201)
def upload_audio_api(file: UploadFile = File(...),
    taken_at: str = Form(...)):
    segmented_audios = segment_audio_file(file, taken_at)
    print(len(segmented_audios))
