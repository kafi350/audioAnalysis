import datetime
import numpy as np

## Audio Preprocessing ##
import wave
import librosa
from scipy.stats import zscore

## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model



class Audio:
    def __init__(self, length, name):
        self.length = length
        self.name = name
        self.created_date = datetime.datetime.now()

    def __str__(self):
        return f"Audio: {self.name}, Length: {self.length}, Created Date: {self.created_date}"



def create_emotion_recognition_model(gender):
    if gender == "Male":
        _model = load_model('app/machine_models/emotion_male_detection_v2.h5')
    else:
        _model = load_model('app/machine_models/emotion_female_detection_v2.h5')

    _emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    return _model, _emotion


def prepare_audio_for_model_emotion(filename, _model, _emotion, sr=16000, n_fft=2048, hop_length=512, n_mels=128, win_length=None, fmax=8000, predict_proba=False):
    y, sr = librosa.load(filename, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    min_time_frames = 5 * 128  # Required time frames
    if S_dB.shape[1] < min_time_frames:
        pad_width = min_time_frames - S_dB.shape[1]
        S_dB_padded = np.pad(S_dB, ((0,0), (0, pad_width)), mode='constant')
    else:
        # Select the first 5*128 time frames
        S_dB_padded = S_dB[:, :min_time_frames]

    S_reshaped = S_dB_padded.reshape(1, 5, 128, 128, 1)

    if predict_proba is True:
        predict = _model.predict(S_reshaped)
    else:
        predict = np.argmax(_model.predict(S_reshaped), axis=1)
        predict = [_emotion.get(emotion) for emotion in predict]

    # Clear Keras session
    K.clear_session()

    # Predict timestamp
    chunk_step = 8000
    chunk_size = 15000
    timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
    timestamp = np.round(timestamp / sr)

    return [predict, timestamp]


def predict_emotion_from_file(filename, _model, _emotion, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):    
    y, sr = librosa.load(filename, sr=sample_rate, offset=0)
    if(len(y) < 41000):
        return prepare_audio_for_model_emotion(filename, _model, _emotion)
    print(y.shape, sr)

    chunks = frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
    chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])
    y = np.asarray(list(map(zscore, chunks)))
    mel_spect = np.asarray(list(map(mel_spectrogram, y)))
    mel_spect_ts = frame(mel_spect)


    y_test = np.random.randn(sample_rate * 3) 
    mel_test = mel_spectrogram(y_test, sr=sample_rate)
    print(mel_test.shape)


    X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                            mel_spect_ts.shape[1],
                            mel_spect_ts.shape[2],
                            mel_spect_ts.shape[3], 1)

    if predict_proba is True:
        predict = _model.predict(X)
    else:
        predict = np.argmax(_model.predict(X), axis=1)
        predict = [_emotion.get(emotion) for emotion in predict]

    # Clear Keras session
    K.clear_session()

    # Predict timestamp
    timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
    timestamp = np.round(timestamp / sample_rate)

    return [predict, timestamp]


def frame(y, win_step=64, win_size=128):
    nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

    frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
    for t in range(nb_frames):
        end = min(y.shape[2], t * win_step + win_size)
        frames[:,t,:,:] = np.pad(y[:,:,(t * win_step):end], ((0,0), (0,0), (0, win_size - end + t * win_step))).astype(np.float16)

    return frames

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return np.asarray(mel_spect)


def extract_all_features_fake_audio(data, sample_rate):
    result = np.array([])
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    return result

def extract_features_fake_audio(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res1 = extract_all_features_fake_audio(data, sample_rate)
    result = np.array(res1)    
    return result