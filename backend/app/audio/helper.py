import librosa
import numpy as np

def extract_feature(file_name):
    audio_data, sample_rate = librosa.load(file_name, sr=None) 
    fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    scaled = np.mean(fea.T,axis=0)
    return np.array([scaled])


audio_classification_prediction_maping = {
    0: "Air Conditioner",
    1: "Car Horn",
    2: "Children Playing",
    3: "Dog Bark",
    4: "Drilling",
    5: "Engine Idling",
    6: "Gun Shot",
    7: "Human Voice",
    8: "Jackhammer",
    9: "Siren",
    10: "Street Music",
}