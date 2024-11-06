from evaluation.metrics_evaluation import BertModelEvaluator
from data.dataset_preparer import HuggingFaceDatasetPreparer
from sklearn.model_selection import train_test_split
import pandas as pd
from config import RANDOM_SEED,MODEL_NAME
from utils.helpers import process_raw_data

def evaluate_model(model_file_path,data_file_path):
    model_evaluator = BertModelEvaluator(model_file_path,use_gpu=False)
    df = pd.read_csv(data_file_path)
    df = process_raw_data(df)

    # Shuffle and split the dataset with fixed seed 42
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Perform stratified split to maintain balance in training and validation sets
    _, temp_df = train_test_split(df, test_size=0.2, stratify=df['Liked'], random_state=RANDOM_SEED)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Liked'], random_state=RANDOM_SEED)
    test_dataset = HuggingFaceDatasetPreparer(test_df,MODEL_NAME).load_and_prepare_dataset()
    evaluation_result  = model_evaluator.evaluate(test_dataset,show_plot=False)
    return evaluation_result

if __name__ == '__main__':
    evaluate_model("../data/models/bert-base-uncased_v1","../data/processed/processed_data.csv")