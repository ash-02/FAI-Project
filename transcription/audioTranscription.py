# import whisper

# # Load the Whisper model
# model = whisper.load_model("base")

# # Transcribe the audio file with word timestamps
# result = model.transcribe("/Users/ashwin/Downloads/output.wav", word_timestamps=True)

# # Extract word-level timestamps and save them to a file
# with open("word_timestamps.txt", "w") as file:
#     for segment in result["segments"]:
#         for word_info in segment["words"]:
#             start = word_info["start"]  # Start time in seconds
#             end = word_info["end"]      # End time in seconds
#             word = word_info.get("word", "")  # The word itself (use .get to avoid KeyError)
#             file.write(f"[{start:.2f}s - {end:.2f}s] {word}\n")

# print("Word-level transcription with timestamps saved to 'word_timestamps.txt'")

import pandas as pd
import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_path):
    
    result = model.transcribe(audio_path, word_timestamps=True)

    data = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            data.append({
                "Start Timestamp": word_info["start"],
                "End Timestamp": word_info["end"],
                "Word": word_info.get("word", "")
            })

    transcript_data = pd.DataFrame(data)
    return transcript_data
