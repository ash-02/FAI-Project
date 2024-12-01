from flask import Flask, request, render_template, jsonify
import os
import CNN_Usage.pv_process as pv_process
import transcription.audioTranscription as at
import adaBoost.classifier as abc
import preprocessing.audioFormat as af


app = Flask(__name__)

# Define folder to save uploaded audio files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Handle the transcription request.
    - Save uploaded audio file.
    - Transcribe the audio and return the transcript.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    # Call transcription function
    transcript_df = at.transcribe_audio(file_path)
    print(transcript_df.columns)

    # Get only the transcript text
    transcript = " ".join(transcript_df['Word'].tolist())

    return jsonify({"transcript": transcript})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return "No audio file uploaded", 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return "No selected file", 400

    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, audio_file.filename)
    audio_file.save(temp_path)

    try:
        # Convert audio file to WAV format
        audio_file_wav = af.convert_to_wav(temp_path)

        # Extract pitch vector from the audio
        pitch_df = pv_process.extract_pitch_vector(audio_file_wav)

        # Transcribe audio to obtain words and timestamps
        transcript_df = at.transcribe_audio(audio_file_wav)

        # Map pitch data to transcribed words
        word_pitch_df = abc.map_pitch_to_words(pitch_df, transcript_df)

        # Use AdaBoost classifier to predict important words
        adaboost_result = abc.use_adaboost(word_pitch_df)

        # Clean up temporary files
        os.remove(temp_path)
        if os.path.exists(audio_file_wav):
            os.remove(audio_file_wav)

        return jsonify({
            "adaboost_result": adaboost_result.to_dict(orient="records")
        })

    except Exception as e:
        return f"Error processing audio: {e}", 500

    finally:
        # Ensure cleanup in case of errors
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)
