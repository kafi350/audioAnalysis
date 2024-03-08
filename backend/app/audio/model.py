import datetime
import numpy as np

## Audio Preprocessing ##
import wave
import librosa
from scipy.stats import zscore

## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM

class Audio:
    def __init__(self, length, name):
        self.length = length
        self.name = name
        self.created_date = datetime.datetime.now()

    def __str__(self):
        return f"Audio: {self.name}, Length: {self.length}, Created Date: {self.created_date}"



def create_emotion_recognition_model(model_weights_path=None):
    if model_weights_path is not None:
        _model = build_model()
        _model.load_weights(model_weights_path)

    _emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    return _model, _emotion

def build_model():
    K.clear_session()
    input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')
    # First LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

    # Second LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

    # Third LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

    # Fourth LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

    # Flat
    y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)

    # LSTM layer
    y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)
    y = Dense(7, activation='softmax', name='FC')(y)
    model = Model(inputs=input_y, outputs=y)
    return model  

def prepare_audio_for_model_emotion(filename, _model, _emotion, sr=16000, n_fft=2048, hop_length=512, n_mels=128, win_length=None, fmax=8000, predict_proba=False):
    # Load the audio file
    y, sr = librosa.load(filename, sr=sr)
    
    # Calculate the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length, fmax=fmax)

    
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Reshape or pad the spectrogram to fit the model input shape, 
    # ensuring we have enough data to form a (5, 128, 128) sequence
    # This step may involve padding S_dB with zeros if it's too small,
    # or selecting appropriate segments if it's large enough
    # For simplicity, let's assume we need to pad the time dimension to at least 640 (5*128)
    min_time_frames = 5 * 128  # Required time frames
    if S_dB.shape[1] < min_time_frames:
        pad_width = min_time_frames - S_dB.shape[1]
        S_dB_padded = np.pad(S_dB, ((0,0), (0, pad_width)), mode='constant')
    else:
        # Select the first 5*128 time frames
        S_dB_padded = S_dB[:, :min_time_frames]
    
    # Reshape into the expected input shape (1, 5, 128, 128, 1)
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

    # Test with a manually created or known good audio chunk 'y_test'
    y_test = np.random.randn(sample_rate * 3)  # Example: 3 seconds of random audio
    mel_test = mel_spectrogram(y_test, sr=sample_rate)
    print(mel_test.shape)



    # Build X for time distributed CNN
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

    # Framing
    frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
    for t in range(nb_frames):
        end = min(y.shape[2], t * win_step + win_size)
        frames[:,t,:,:] = np.pad(y[:,:,(t * win_step):end], ((0,0), (0,0), (0, win_size - end + t * win_step))).astype(np.float16)

    return frames

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    # Compute log-mel spectrogram
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