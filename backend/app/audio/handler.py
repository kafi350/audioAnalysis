from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/audio")


@router.get("/")
def get_audio_info():
    return {"Hello": "World"}