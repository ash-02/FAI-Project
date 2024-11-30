import os
import librosa
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

data_folder = './data'

window_size = 8000
step_size = 4000

def process_file(wav_path, pv_path, sr, frame_size=256):
    y, _ = librosa.load(wav_path, sr=sr)

    with open(pv_path, 'r') as file:
        pitch_data = [float(value.strip()) for value in file if value.strip()]

    num_frames = len(pitch_data)

    audio_segments = []
    labels = []

    num_windows = (len(y) - window_size) // step_size + 1

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = y[start:end]

        if len(segment) == window_size:
            audio_segments.append(segment)

            start_frame = int(start / frame_size)
            end_frame = int(end / frame_size)

            if start_frame < num_frames and end_frame <= num_frames:
                segment_pv = pitch_data[start_frame:end_frame]

                # print(f"Start frame: {start_frame}, End frame: {end_frame}, Segment PV: {segment_pv}")

                if len(segment_pv) > 0 and max(segment_pv) > 0:
                    label = max(segment_pv)
                    labels.append(label)
                else:
                    print(f"Warning: Empty or zero segment_pv for start_frame {start_frame} and end_frame {end_frame}")
                    audio_segments.pop()

    # print(f"Processed {len(audio_segments)} segments and {len(labels)} labels for {wav_path}")

    return np.array(audio_segments), np.array(labels)

all_audio_segments = []
all_labels = []

for root, _, files in os.walk(data_folder):
    for filename in files:
        if filename.endswith('.wav'):
            base_name = filename.split('.')[0]
            wav_path = os.path.join(root, f"{base_name}.wav")
            pv_path = os.path.join(root, f"{base_name}.pv")

            if os.path.exists(pv_path):
                audio_segments, labels = process_file(wav_path, pv_path, sr=8000)
                all_audio_segments.extend(audio_segments)
                all_labels.extend(labels)

all_audio_segments = np.array(all_audio_segments)
all_labels = np.array(all_labels)

all_audio_segments = all_audio_segments.reshape(-1, window_size, 1)