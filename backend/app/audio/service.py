from app.audio.helper import extract_feature
import random

# from tensorflow.keras.models import load_model
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


def classify_audio_class(file: UploadFile):
    audio = AudioSegment.from_file(file.file, format=file.filename.split('.')[-1])
    length_ms = len(audio)
    # Assuming 'pred_fea' is your input data
    # pred_fea = extract_feature(file.file)
    # model = load_model('path_to_your_model.h5')

    # Ensure the input shape matches your model's input shape
    # If your model expects a 4D input (batch_size, height, width, channels), 
    # you might need to expand the dimensions of your input
    # pred_fea = np.expand_dims(pred_fea, axis=0)

    # # Use the model to make a prediction
    # pred_vec = model.predict(pred_fea)

    # # Get the predicted class
    # predicted_class = np.argmax(pred_vec, axis=-1)
    
    predicted = predict_gender_emotion()
    print(predicted)
    return predicted