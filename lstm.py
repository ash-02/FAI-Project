import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

import contractions
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')



class lstm:
    def __init__(self) -> None:
        self.data = None
        self.model = None
        self.tokenizer = None

        self.max_summary_len = 100
        self.max_transcript_len = 500
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self,text):
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        tokens = word_tokenize(text)

        # Remove very short and very long words
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [t for t in tokens if 2 <= len(t) <= 20]

        return ' '.join(tokens)

    def read_preprocess_data(self):
        # Read data
        self.data = pd.read_csv('ted_talks_en.csv', on_bad_lines='skip')
        self.data = self.data[['transcript','description']]

        # Preprocess
        self.data = self.data.dropna(subset=['transcript', 'description'])
        self.data = self.data.dropna(subset=['transcript', 'description'])

        self.data['transcript'] = self.data['transcript'].astype(str).apply(self.preprocess_text)
        self.data['description'] = self.data['description'].astype(str).apply(self.preprocess_text)

    def train(self):
        self.read_preprocess_data()
        # Split the dataset to train and test
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        # Hyperparameters
        
        embedding_dim = 500
        lstm_units = 256
        vocab_size = 20000
        batch_size = 32
        epochs = 10

        # Initialize the Tokenizer
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", filters='')
        self.tokenizer.fit_on_texts(train_data['transcript'].values.tolist() + train_data['description'].values.tolist())

        # Tokenize and pad transcripts and summaries
        train_transcript_sequences = pad_sequences(
            self.tokenizer.texts_to_sequences(train_data['transcript']),
            maxlen=self.max_transcript_len,
            padding='post'
        )
        train_summary_sequences = pad_sequences(
            self.tokenizer.texts_to_sequences(train_data['description']),
            maxlen=self.max_summary_len,
            padding='post'
        )
        test_transcript_sequences = pad_sequences(
            self.tokenizer.texts_to_sequences(test_data['transcript']),
            maxlen=self.max_transcript_len,
            padding='post'
        )
        test_summary_sequences = pad_sequences(
            self.tokenizer.texts_to_sequences(test_data['description']),
            maxlen=self.max_summary_len,
            padding='post'
        )

        # Encoder
        encoder_inputs = Input(shape=(self.max_transcript_len,))
        encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        encoder_lstm = LSTM(lstm_units, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(self.max_summary_len,))
        decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # # Shift summary sequences for decoder input and target data
        decoder_input_data = pad_sequences(train_summary_sequences[:, :-1], maxlen=self.max_summary_len, padding='post')
        decoder_target_data = pad_sequences(train_summary_sequences[:, 1:], maxlen=self.max_summary_len, padding='post')

        # Train the model
        history = self.model.fit(
            [train_transcript_sequences, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)

lstm = lstm()
lstm.train()