import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wave
import pydub
from scipy.signal import resample

model = tf.keras.models.load_model('./model/model.h5')
window_size = 8000
step_size = 4000
sr = 8000

def process_wav_file_for_prediction(wav_path, window_size, step_size, target_sr=8000):
    with wave.open(wav_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        audio_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

    audio_segments = []
    num_windows = (len(audio_data) - window_size) // step_size + 1

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = audio_data[start:end]

        if len(segment) == window_size:
            audio_segments.append(segment)

    audio_segments = np.array(audio_segments).reshape(-1, window_size, 1)
    return audio_segments

song_wav_path = 'cleaned_webm/audio1.webm'

song_audio_segments = process_wav_file_for_prediction(song_wav_path, window_size, step_size)

predicted_song_pitches = model.predict(song_audio_segments)