import os
import sys
from pathlib import Path
import argparse


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from preprocessing.clean_csv_for_lstm_training import clean
from setup_lstm import *

def main(input_csv):
    # Step 1. Clean our input dataset
    if not Path("../data/cleaned_lstm_train.csv").exists():
        clean(input_csv)
        print(f"Cleaned CSV saved to data/")
    else:
        print(f"Dataset already cleaned for inputting into LSTM")
    
    # Step 2: Handle tokenization, vocab building, word embeddings to give to LSTM
    # TODO: Actually refactor the code so project structure is less confusing and code is more optimized
    # if Path("../data/preprocessed_data.pt").exists():
    #     print(f"Dataset already tokenized ")
    #     # setup_lstm(vocab_size)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and create datasets for LSTM training.")
    parser.add_argument(
        "--input_csv", 
        type=str, 
        required=True, 
        help="Path to the input CSV file"
    )
    args = parser.parse_args()
    main(args.input_csv)
