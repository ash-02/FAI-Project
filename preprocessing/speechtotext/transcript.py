from openai import OpenAI
from pydub import AudioSegment
from config import API_KEY, ORGANIZATION
import sys

import math
import os

# Segment audio file to align with OpenAI restrictions on audio size / length
def segment_audio(file_path, target_size_mb=20, output_format="mp3"):
    audio = AudioSegment.from_file(file_path)

    file_size_bytes = os.path.getsize(file_path)
    duration_ms = len(audio)
    target_size_bytes = target_size_mb * 1024 * 1024
    target_duration_ms = (target_size_bytes / file_size_bytes) * duration_ms

    # Calculate num of segments needed
    num_segments = int(math.ceil(duration_ms / target_duration_ms))

    segments = []
    # Split and export the segments
    for i in range(num_segments):
        start_ms = i * target_duration_ms
        end_ms = min((i + 1) * target_duration_ms, duration_ms)
        segment = audio[start_ms:end_ms]
        segments.append(segment)
    return segments

def transcribe_audio(audio_file):
  client = OpenAI(
      api_key=API_KEY,
      organization=ORGANIZATION
  )

  try:
    print("segmenting the audio file")
    segments = segment_audio(audio_file)
  except Exception as e:
    print("Error in segmenting the audio file")
    print(e)
    exit(1)

  try:
     print("transcribing the audio file")
     temp_dir = "temp_audio_segments"
     os.makedirs(temp_dir, exist_ok=True)
     format = "mp3"
     # audio_transcription_name based on the audio file name
     audio_transcription_file = audio_file.split(".")[0] + ".txt"
     output_path = '/usr/src/app/output/' + audio_transcription_file
     with open(output_path, "w") as transcript_file:
        for i, segment in enumerate(segments):
            prompt = ""
            if i != 0:
                prompt = "this is a continuation of the previous audio segment of the same audio file."
            # Export each segment to a temporary file
            temp_file_path = os.path.join(temp_dir, f"segment_{i}.{format}")
            segment.export(temp_file_path, format=format)

            ready_to_transcribe = open(temp_file_path, "rb")
            # Transcribe the temporary audio file
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=ready_to_transcribe,
                response_format="text",
                prompt = prompt
            )

            # Append the transcription to the transcript file
            transcript_file.write(transcript + "\n")

            # Optionally, remove the temporary file after transcription
            os.remove(temp_file_path)

        # Optionally, remove the temp directory if empty
        os.rmdir(temp_dir)
  except Exception as e:
    print("Error in transcribing the audio file")
    print(e)
    # print error line number
    print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))

    exit(1)


# take in first argument as the audio file
# audio_file= "myfile.m4a"

# audio_file = sys.argv[1]
# transcribe_audio(audio_file)
