import os
from app.audio.helper import extract_feature, audio_classification_prediction_maping
import random
import tempfile


from tensorflow.keras.models import load_model
import numpy as np
from fastapi import UploadFile
from pydub import AudioSegment
from pydub.silence import split_on_silence

#this will create chunks of audio if audio is longer than 40 seconds
def create_audio_chunks(file: UploadFile, taken_at: str):
    audio = AudioSegment.from_file(file.file, format=file.filename.split('.')[-1])
    length_ms = len(audio)
    print(f"Audio length: {length_ms} ms")
    if length_ms > 40000:
        # Divide audio into 40-second chunks
        chunks = [audio[i:i+40000] for i in range(0, length_ms, 40000)]
        segments = segment_audio_file(chunks)
        return segments

    segments = segment_audio_file([audio])
    
    return segments

#this will segment the audios based on the silence parts (using pydub)
# so it will create two segments if there is a silence part in the audio
# one with the speech and one with the silence
def segment_audio_file(chunks):
    segments = []
    for chunk in chunks:
        # Parameters: audio segment, minimum silence length in ms, silence threshold in dB
        chunk_segments = split_on_silence(chunk, min_silence_len=500, silence_thresh=-40)
        segments.extend(chunk_segments)
        
    return segments

def audio_classification(audio_file):
    # Classify audio here
    return "Positive" # or "Negative" or "Neutral" depending on the classification


def predict_gender_emotion():
    genders = ['male', 'female']
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'scared', 'disgusted', 'contemptuous']

    predicted_gender = random.choice(genders)
    predicted_emotion = random.choice(emotions)
    
    return{
        "gender": predicted_gender,
        "emotion": predicted_emotion
    }


async def classify_audio_class(file: UploadFile):
    local_file_path = await save_file(file)   
    # Now you can read the local file with pydub
    audio = AudioSegment.from_file(local_file_path, format=file.filename.split('.')[-1])
    length_ms = len(audio)

    features = extract_feature(local_file_path)
    

    # Load the model
    predicted_class = audio_classification_model(features)
    return {
        "class": predicted_class,
    }
    

async def save_file(file: UploadFile):
    local_file_path = os.path.join("uploads", file.filename)

    # Save the uploaded file to the local file
    data = await file.read()
    with open(local_file_path, 'wb') as f:
        f.write(data)
    print(f"File saved to {local_file_path}")
    return local_file_path


def audio_classification_model(features):
    model_path = "app/machine_models/audio_classify_v1.h5"
    model = load_model(model_path)

    pred_vector = np.argmax(model.predict(features), axis=-1)
    class_name = audio_classification_prediction_maping[pred_vector[0]]
 
    return class_name
