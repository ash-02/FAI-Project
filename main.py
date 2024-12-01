import CNN_Usage.pv_process as pv_process
import transcription.audioTranscription as at
import adaBoost.classifier as abc
import preprocessing.audioFormat as af

# audio_file = '/Users/ashwin/Downloads/output.wav'
audio_file = "/Users/saicharanreddy/cleaned_webm/audio1.webm"

audio_file = af.convert_to_wav(audio_file)

pitch_df = pv_process.extract_pitch_vector(audio_file)

transcript_df = at.transcribe_audio(audio_file)

word_pitch_df = abc.map_pitch_to_words(pitch_df, transcript_df)

print(abc.use_adaboost(word_pitch_df))

