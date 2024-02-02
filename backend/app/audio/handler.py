import base64
import io
from datetime import datetime
from app.audio.api_model import AudioRequest
from app.audio.service import create_audio_chunks, segment_audio_file

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

