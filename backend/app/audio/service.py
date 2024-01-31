from fastapi import UploadFile
from pydub import AudioSegment

def segment_audio_file(file: UploadFile, taken_at: str):
    print(file.filename)
    audio = AudioSegment.from_file(file.file, format=file.filename.split('.')[-1])
    length_ms = len(audio)

    # If audio is longer than 40 seconds
    if length_ms > 40000:
        # Divide audio into 40-second chunks
        chunks = [audio[i:i+40000] for i in range(0, length_ms, 40000)]
        return chunks

    # If audio is shorter than 40 seconds, return it as is
    return [audio]