import os
from app.audio.helper import convert_class_to_emotion, emotion_detection_feature, extract_feature, audio_classification_prediction_maping, extract_feature_gender, get_features_gender_emotion
import random
import tempfile
from app.audio.utils import create_model



from tensorflow.keras.models import load_model
import numpy as np
from fastapi import UploadFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
# import keras

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

async def classify_audio_class(file: UploadFile):
    local_file_path = await save_file(file)   
    audio = AudioSegment.from_file(local_file_path, format=file.filename.split('.')[-1])
    features = extract_feature(local_file_path)
    predicted_class = audio_classification_model(features)
    return {
        "class": predicted_class,
    }

async def gender_detection(file: UploadFile):
    local_file_path = await save_file(file)   
    features = extract_feature_gender(local_file_path, mel=True).reshape(1, -1)

    prediction = gender_detection_model(features)

    return {
        "prediction": prediction[0],
        "male": prediction[1],
        "female": prediction[2]

    }
   
async def emotion_detection(file: UploadFile):
    local_file_path = await save_file(file)   
    features = get_features_gender_emotion(local_file_path)
    prediction = emotion_detection_model(features)   

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


def gender_detection_model(features):
    
    model = create_model()
    # load the saved/trained weights
    model.load_weights("app/machine_models/model.h5")
    
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    male = male_prob*100    
    female = female_prob*100
    return gender, male, female

def emotion_detection_model(features):
    model_path = "app/machine_models/emotion_female_detection_v1.h5"
    model_name = 'Emotion_Voice_Detection_Model.h5'
    model = load_model(model_path)
    model.summary()
    predictions = model.predict(features)
    print(predictions)
    predictions = np.argmax(predictions, axis=-1)
    print(predictions)
    # print( "Prediction is", " ", convert_class_to_emotion(predictions))


    