import base64
import glob
import os
import tempfile
import librosa
import numpy as np
from app.audio.helper import convert_class_to_emotion, emotion_detection_feature, extract_feature, audio_classification_prediction_maping, extract_feature_gender, get_features_gender_emotion

from app.audio.utils import create_model
from app.audio.model import create_emotion_recognition_model, extract_features_fake_audio, predict_emotion_from_file
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from tensorflow.keras.models import load_model
from fastapi import UploadFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
# import keras
import auditok
import matplotlib.pyplot as plt
from scipy.io import wavfile




#this will create chunks of audio if audio is longer than 40 seconds
async def create_audio_chunks(file: UploadFile, taken_at: str):
    audio = AudioSegment.from_file(file.file, format=file.filename.split('.')[-1])
    length_ms = len(audio)
    audio.export("uploads/output.wav", format="wav")
    sample_rate, data = wavfile.read("uploads/output.wav")
    plt.figure(figsize=(6, 4))
    plt.plot(data)
    plt.savefig('uploads/original_waveform.png')
    with open('uploads/original_waveform.png', 'rb') as img_file:
        waveform_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    optimal_hop_sizes = compute_optimal_hop_size('uploads/output.wav')
    averave_hop_size = int(np.mean(optimal_hop_sizes))
    
    segmented_regions, segments = energy_based_vad('uploads/output.wav', frame_size=2048, hop_length=averave_hop_size, energy_threshold=0.001)
    
    return segments, segmented_regions, waveform_image_base64

def segment_audio_file(chunks):
    segments = []
    for chunk in chunks:
        # Parameters: audio segment, minimum silence length in ms, silence threshold in dB
        chunk_segments = split_on_silence(chunk, min_silence_len=500, silence_thresh=-40)
        segments.extend(chunk_segments)
        
    return segments

def energy_based_vad(audio_path, frame_size, hop_length, energy_threshold):
    y, sr = librosa.load(audio_path)
    ste = np.power(y, 2)
    ste = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length, center=True)
    mask = ste > energy_threshold
    segments = []
    start_sample = 0
    for i in range(1, len(mask[0])):
        if mask[0][i] != mask[0][i-1]:
            end_sample = i * hop_length
            if mask[0][i-1] == True:
                segments.append((start_sample, end_sample))
            start_sample = end_sample

    segments_time = [(start / sr, end / sr) for start, end in segments]
    audio = AudioSegment.from_wav(audio_path)
    segmented_audios = []
    segmented_time = []
    for i, (start, end) in enumerate(segments_time):
        segment_audio = audio[start * 1000:end * 1000]  # pydub works in milliseconds
        filename = f"uploads/segment_{i}.wav"
        segmented_time.append({'start': start, 'end': end})
        segment_audio.export(filename, format="wav")
        segmented_audios.append(segment_audio)

    return segmented_time, segmented_audios


def split_audio(audio_path, n_clusters):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.T
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mfcc)
    segments = kmeans.labels_
    frame_length = 2048  # Default frame length for librosa.feature.mfcc
    hop_length = 512  # Default hop length for librosa.feature.mfcc
    segment_times = [(i * hop_length / sr, (i + np.count_nonzero(segments == i)) * hop_length / sr) for i in range(n_clusters)]

    return segment_times

def compute_optimal_hop_size(audio_path):
    y, sr = librosa.load(audio_path)
    frame_length = 2048
    hop_length = 512
    
    ste = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    
    energy_variability = np.diff(ste[0])
   
    adaptive_hop_sizes = []
    threshold = np.median(energy_variability) * 1.5
    
    for variability in energy_variability:
        if abs(variability) > threshold:
            adaptive_hop_sizes.append(hop_length // 2)
        else:
            adaptive_hop_sizes.append(hop_length * 2)
    min_hop = min(adaptive_hop_sizes)
    max_hop = max(adaptive_hop_sizes)

    if min_hop != max_hop:
        normalized_hop_sizes = [(hop - min_hop) / (max_hop - min_hop) * (512 - 128) + 128 for hop in adaptive_hop_sizes]
    else:
        normalized_hop_sizes = [hop_length for hop in adaptive_hop_sizes] 

    
    return normalized_hop_sizes

async def classify_audio_class(file: UploadFile):
    local_file_path = await save_file(file)   
    audio = AudioSegment.from_file(local_file_path, format=file.filename.split('.')[-1])
    features, base64_mfcc = extract_feature(local_file_path)
    predicted_class = audio_classification_model(features)
    return {
        "class": predicted_class,
        "mfcc": base64_mfcc
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
   
async def emotion_detection(file: UploadFile, gender: str):
    local_file_path = await save_file(file)  
    step = 1 # in sec
    sample_rate = 16000 # in kHz

    _model, _emotion = create_emotion_recognition_model(gender)
    emotions, timestamp = predict_emotion_from_file(local_file_path, _model, _emotion, chunk_step=step*sample_rate)

    return {
        "emotions": emotions[0],
        "timestamp": timestamp[0]
    }

async def fake_audio(file: UploadFile):
    local_file_path = 'uploads/audio.wav'
    local_model_path = "app/machine_models/fake_audio.h5"
    scaler = StandardScaler()
    features = extract_features_fake_audio(local_file_path)
    features = features.reshape(1, -1)
    features = scaler.fit_transform(features)
    features = np.expand_dims(features, axis=2)
    model = load_model(local_model_path)
    prediction = model.predict(features)
    binary_predictions = (prediction).astype(int)

    if binary_predictions[0][0] == 1:
        return {
            "prediction": "Spoofed"
        }
    else:
        return {
            "prediction": "Bonafide"
        }
      
    

async def save_file(file: UploadFile):
    local_file_path = os.path.join("uploads", file.filename)
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
    model = load_model('app/machine_models/gender_detection_v3.h5')
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    male = male_prob*100    
    female = female_prob*100
    return gender, male, female

def delete_files_in_directory(directory):
    files = glob.glob(f'{directory}/*')
    for file in files:
        os.remove(file)
   

    