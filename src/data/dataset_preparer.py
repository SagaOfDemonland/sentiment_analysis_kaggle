import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset

class HuggingFaceDatasetPreparer:
    def __init__(self, df, model_name):
        """
        Initialize the dataset preparer with a file path and model name.
        
        Parameters:
        - file_path (str): Path to the CSV file containing the dataset.
        - model_name (str): Hugging Face model name for tokenizer and model.
        """
        self.df = df
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
    def load_and_prepare_dataset(self):
        """
        Load the dataset from a CSV file, format it for Hugging Face, and tokenize it.
        
        Returns:
        - hf_dataset (Dataset): The processed Hugging Face dataset ready for training or evaluation.
        """
        
        # Rename target column to 'labels' for compatibility with Hugging Face's expected format
        self.df['labels'] = self.df['Liked']
        
        # Convert the DataFrame to a Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(self.df)
        
        # Tokenize the dataset
        hf_dataset = hf_dataset.map(self.tokenize_function, batched=True)
        
        # Set the dataset format for PyTorch
        hf_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return hf_dataset
    
    @staticmethod
    def tokenize_function(examples):
        """
        Tokenize the 'Review' column and prepare labels.
        
        Parameters:
        - examples (dict): Dictionary of examples containing the 'Review' and 'labels' columns.
        
        Returns:
        - dict: Tokenized output including 'input_ids', 'attention_mask', and 'labels'.
        """
        # Tokenize the 'Review' column and add labels if available
        tokenized = HuggingFaceDatasetPreparer.tokenizer(examples['Review'], padding='max_length', truncation=True)
        tokenized['labels'] = examples['labels']
        return tokenized
