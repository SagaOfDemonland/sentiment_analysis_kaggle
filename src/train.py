from models.bert_model import BertFineTuner
from data.dataset_preparer import HuggingFaceDatasetPreparer
from sklearn.model_selection import train_test_split
import pandas as pd
from config import RANDOM_SEED,MODEL_NAME
from utils.helpers import process_raw_data

def train_model(data_file_path:str):
    bert_finetuer = BertFineTuner()

    df = pd.read_csv(data_file_path)
    df = process_raw_data(df)

    # Shuffle and split the dataset with fixed seed 42
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Perform stratified split to maintain balance in training and validation sets
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Liked'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Liked'], random_state=RANDOM_SEED)

    train_dataset = HuggingFaceDatasetPreparer(train_df,MODEL_NAME).load_and_prepare_dataset()
    val_dataset = HuggingFaceDatasetPreparer(val_df,MODEL_NAME).load_and_prepare_dataset()

    bert_finetuer.train(train_dataset,val_dataset)

if __name__ == '__main__':
    train_model("../data/processed/processed_data.csv")