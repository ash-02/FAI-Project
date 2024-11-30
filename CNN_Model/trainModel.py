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
    audio_duration = len(y) / sr

    frame_duration = frame_size / sr

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
                audio_segments, labels = process_file(wav_path, pv_path, sr=8000)  # Assuming 8 kHz sampling rate
                all_audio_segments.extend(audio_segments)
                all_labels.extend(labels)

print(f"Total audio segments: {len(all_audio_segments)}")
print(f"Total labels: {len(all_labels)}")

if len(all_audio_segments) != len(all_labels):
    raise ValueError("Mismatch between the number of audio segments and labels.")

all_audio_segments = np.array(all_audio_segments)
all_labels = np.array(all_labels)

all_audio_segments = all_audio_segments.reshape(-1, window_size, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_audio_segments, all_labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.summary()
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae:.2f}")

predicted_pitches = model.predict(X_test)
step_duration = step_size / 8000
timestamps = np.arange(0, len(y_test) * step_duration, step_duration)[:len(y_test)]
pitch_time_dict = {float(timestamps[i]): float(predicted_pitches[i]) for i in range(len(timestamps))}

print("Pitch values with time in dictionary format:")
print(pitch_time_dict)
model_save_path = './model/model.h5'
model.save(model_save_path)
timestamps = timestamps[:len(y_test)]

plt.figure(figsize=(14, 6))
plt.plot(timestamps, y_test, label='True Pitch')
plt.plot(timestamps, predicted_pitches, label='Predicted Pitch', linestyle='dashed')
plt.legend()
plt.title('True vs Predicted Pitch Values Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Pitch (MIDI Number)')
plt.show()