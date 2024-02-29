import base64
import io
from datetime import datetime
from app.audio.api_model import AudioRequest
from app.audio.service import classify_audio_class, create_audio_chunks, emotion_detection, fake_audio, gender_detection, segment_audio_file

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form

router = APIRouter(prefix="/audio")
     
    
@router.post("/upload", status_code=201)
def upload_audio_api(file: UploadFile = File(...),
    taken_at: str = Form(...)):
    segmented_audios = create_audio_chunks(file, taken_at)

    segmented_audios_base64 = []
    for segment in segmented_audios:
        buffer = io.BytesIO()
        segment.export(buffer, format="wav")
        buffer.seek(0)
        segment_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        segmented_audios_base64.append(segment_base64)
    
    return {
        "audio_count": len(segmented_audios),
        "segmented_audios": segmented_audios_base64,
    }

@router.post("/classify", status_code=201)
async def classify_audio_api(file: UploadFile = File(...)):
    return await classify_audio_class(file)

@router.post("/genderdetection", status_code=201)
async def gender_detection_api(file: UploadFile = File(...)):
    return await gender_detection(file)

@router.post("/emotiondetection", status_code=201)
async def emotion_detection_api(file: UploadFile = File(...)):
    return await emotion_detection(file)

@router.post("/fakeaudio", status_code=201)
async def fake_audio_api(file: UploadFile = File(...)):
    return await fake_audio(file)