import base64
import io
from datetime import datetime
import os
from app.audio.api_model import AudioRequest
from app.audio.service import classify_audio_class, create_audio_chunks, delete_files_in_directory, emotion_detection, fake_audio, gender_detection, segment_audio_file
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form

router = APIRouter(prefix="/audio")
     
    
@router.post("/upload", status_code=201)
async def upload_audio_api(file: UploadFile = File(...),
    taken_at: str = Form(...)):
    os.makedirs('uploads', exist_ok=True)
    delete_files_in_directory('uploads')

    segmented_audios, segmented_regions, original_waveform = await create_audio_chunks(file, taken_at)
    
    segmented_audios_base64 = []
    waveform_images_base64 = []
    for segment in segmented_audios:
        buffer = io.BytesIO()
        segment.export(buffer, format="wav")
        buffer.seek(0)
        segment_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        segmented_audios_base64.append(segment_base64)
        buffer.seek(0)
        sample_rate, data = wavfile.read(buffer)
        plt.figure(figsize=(6, 4))
        plt.plot(data)
        plt.savefig('waveform.png')

        with open('waveform.png', 'rb') as img_file:
            waveform_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        waveform_images_base64.append(waveform_image_base64)

    
    return {
        "original_waveform": original_waveform,
        "audio_count": len(segmented_audios),
        "segmented_audios": segmented_audios_base64,
        "waveform_images": waveform_images_base64,
        "segmented_regions": segmented_regions
    }

@router.post("/classify", status_code=201)
async def classify_audio_api(file: UploadFile = File(...)):
    return await classify_audio_class(file)

@router.post("/genderdetection", status_code=201)
async def gender_detection_api(file: UploadFile = File(...)):
    return await gender_detection(file)

@router.post("/emotiondetection", status_code=201)
async def emotion_detection_api(file: UploadFile = File(...), gender: str = Form(...)):
    return await emotion_detection(file, gender)

@router.post("/fakeaudio", status_code=201)
async def fake_audio_api(file: UploadFile = File(...)):
    print(file)
    return await fake_audio(file)
