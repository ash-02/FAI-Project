import pandas as pd

def map_pitch_to_words(pitch_data, transcript_data):
    """
    Maps pitch values to words using a sliding window of size 5 (half-second intervals).
    Includes the ground true label (1/0) our group manually labeled for each word's importance in the output.
    Returns a dictionary where keys are words and values are dictionaries with pitch values and label.
    """
    result = []

    grouped_pitch = pitch_data.groupby("File Name")
    grouped_transcript = transcript_data.groupby("File Name")
    
    for file_name, transcript_group in grouped_transcript:
        if file_name not in grouped_pitch.groups:
            continue 

        pitch_group = grouped_pitch.get_group(file_name).dropna()
        pitch_times = pitch_group["Timestamp (s)"].values
        pitch_values = pitch_group["Pitch Value"].values

        for _, row in transcript_group.iterrows():
            start_time = row["Start Timestamp"]
            end_time = row["End Timestamp"]
            word = row["Word"]
            label = row["True Label"]
            
            label = int(label) if not pd.isna(label) else 0

            word_pitches = []

            current_time = start_time
            while current_time < end_time:
                nearest_pitch = next((p for t, p in zip(pitch_times, pitch_values) if abs(t - current_time) < 0.25), 0)
                word_pitches.append(nearest_pitch)
                current_time += 0.5
                if len(word_pitches) == 5:
                    break

            word_pitches.extend([0] * (5 - len(word_pitches)))


            result.append({
                "word": word,
                "pitches": word_pitches,
                "label": label
            })

    return result

def save_results(results, output_file):
    """
    Saves the results dictionary as a CSV file.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":

    pitch_file_path = "data/grouped_pitch_vectors.csv"
    transcript_file_path = "data/timestamps_for_words.csv"
    output_file = "data/word_pitch_mappings.csv"

    pitch_data = pd.read_csv(pitch_file_path)
    transcript_data = pd.read_csv(transcript_file_path)
    results = map_pitch_to_words(pitch_data, transcript_data)
    
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

