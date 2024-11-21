import pandas as pd
import re

def clean(file_path):
    """
    Takes in a file_path for a CSV to be cleaned, cleaning it for LSTM training. 
    For LSTM component we decided to set up the task such that the LSTM takes input 'transcript' and tries to output 'description',
    i.e. the summary of 'transcript'.
    """
    data = pd.read_csv('../data/ted_talks_en.csv', encoding='latin1', on_bad_lines='skip')

    # Only keep relevant columns
    # For LSTM training we decided to set up task such that LSTM takes input 'transcript' and tries to output 'description',
    # in other words, the summary
    cols_to_retain = ['description', 'transcript']
    data = data[cols_to_retain]

    # Normalize text to unicode, lowercase, alphanumeric + punctuation only
    def normalize_text(text):
        if isinstance(text, str):
            text = re.sub(r'[^\w\s.,:;!?\'"()\-]', '', text, flags=re.UNICODE)
            return re.sub(r'\s+', ' ', text.strip().lower())
        return text

    data = data.applymap(normalize_text)

    # Drop empty/null rows
    data = data.dropna(subset=['description', 'transcript'])
    data = data[
        ~data['description'].str.strip().eq('') &
        ~data['transcript'].str.strip().eq('')
    ]

    print(f"Dataset dims after cleaning: {data.shape}")

    data.to_csv('../data/cleaned_lstm_train.csv')
