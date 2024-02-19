import librosa
import numpy as np

def extract_feature(file_name):
    audio_data, sample_rate = librosa.load(file_name, sr=None) 
    fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    scaled = np.mean(fea.T,axis=0)
    return np.array([scaled])


audio_classification_prediction_maping = {
    0: "Positive",
    1: "Negative",
    2: "Neutral"
}