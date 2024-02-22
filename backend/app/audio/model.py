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

    # Fully connected
    y = Dense(7, activation='softmax', name='FC')(y)

    # Build final model
    model = Model(inputs=input_y, outputs=y)

    return model  



def predict_emotion_from_file(filename, _model, _emotion, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):
    y, sr = librosa.load(filename, sr=sample_rate, offset=0.5)
    chunks = frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
    chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])
    y = np.asarray(list(map(zscore, chunks)))
    mel_spect = np.asarray(list(map(mel_spectrogram, y)))
    mel_spect_ts = frame(mel_spect)

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