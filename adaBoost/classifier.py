import pandas as pd
import joblib

# def map_pitch_to_words(pitch_data, transcript_data):
#   """
#   Maps pitches to words in a transcript with a fixed window of 5 for pitches per word.
#   Input:
#   pitch_data : Pitch data as a dataframe with columns "Timestamp (s)" and "Pitch Value"
#   transcript_data : Transcript data as a dataframe with columns "Start Timestamp", "End Timestamp", "Word"
#   Output:
#   result : Dataframe with columns "word", "pitches"
#   """
#   result = []
#   pitch_times = pitch_data["Timestamp (s)"].values
#   pitch_values = pitch_data["Pitch Value"].values
#   for _, row in transcript_data.iterrows():
#     start_time = row["Start Timestamp"]
#     end_time = row["End Timestamp"]
#     word = row["Word"]
#     label = int(label) if not pd.isna(label) else 0
#     word_pitches = []
#     current_time = start_time
#     while current_time < end_time:
#       nearest_pitch = next((p for t, p in zip(pitch_times, pitch_values) if abs(t - current_time) < 0.25), 0)
#       word_pitches.append(nearest_pitch)
#       current_time += 0.5
#       if len(word_pitches) == 5:
#         break
#       word_pitches.extend([0] * (5 - len(word_pitches)))
#       result.append({
#           "word": word,
#           "pitches": word_pitches,
#           })
#       results_df = pd.DataFrame(result)
#       return results_df

import pandas as pd

def map_pitch_to_words(pitch_data, transcript_data):
    """
    Maps pitches to words in a transcript with a fixed window of 5 pitches per word.
    
    Parameters:
    - pitch_data : DataFrame with columns "Timestamp (s)" and "Pitch Value".
    - transcript_data : DataFrame with columns "Start Timestamp", "End Timestamp", "Word".
    
    Returns:
    - DataFrame with columns "word" and "pitches", where "pitches" contains a list of up to 5 pitch values.
    """
    result = []  # To store mapping of words to pitches
    pitch_times = pitch_data["Timestamp (s)"].values
    pitch_values = pitch_data["Pitch Value"].values

    for _, row in transcript_data.iterrows():
        start_time = row["Start Timestamp"]
        end_time = row["End Timestamp"]
        word = row["Word"]
        
        word_pitches = []
        current_time = start_time

        # Collect pitches within the word's time range
        while current_time < end_time:
            # Find the nearest pitch within a tolerance of 0.25 seconds
            nearest_pitch = next(
                (p for t, p in zip(pitch_times, pitch_values) if abs(t - current_time) < 0.25), 0  # Default value if no pitch is found
            )
            word_pitches.append(nearest_pitch)
            current_time += 0.5

            # Stop collecting once we have 5 pitches
            if len(word_pitches) == 5:
                break
        
        # Pad with zeros if fewer than 5 pitches are found
        word_pitches.extend([0] * (5 - len(word_pitches)))

        # Append to the result
        result.append({
            "word": word,
            "pitches": word_pitches,
        })

    # Convert result to DataFrame
    results_df = pd.DataFrame(result)
    return results_df


def use_adaboost(word_pitch_mappings):
    """
    Function to use the adaboost classifier to predict the label(importance) of a word.
    Input:
    word_pitch_mappings : Dataframe with columns "word", "pitches"
    Output:
    result : Dataframe with columns "word", "label"
    """

    adaboost_clf = joblib.load('model/adaboost_classifier.pkl')

    result = pd.DataFrame()
    # print(word_pitch_mappings)
    result['word'] = word_pitch_mappings["word"]
    X = pd.DataFrame(word_pitch_mappings['pitches'].tolist(), columns=[f'Pitch_{i+1}' for i in range(5)])
    result['label'] = adaboost_clf.predict(X)

    return result