import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from ..data.dataset_preparer import HuggingFaceDatasetPreparer
from ..utils.helpers import preprocess_text,process_raw_data
from ..config import BATCH_SIZE
class BertInference:
    def __init__(self, model_path, use_gpu=True):
        """
        Initialize the inference class with model path and GPU usage.
        
        Parameters:
        - model_path (str): Path to the fine-tuned model.
        - use_gpu (bool): If True, use GPU for evaluation if available.
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Load the tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text input.
        
        Parameters:
        - text (str): The text input to predict.
        
        Returns:
        - tuple: Predicted label and confidence score.
        """
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Tokenize and move inputs to the appropriate device
        inputs = self.tokenizer(processed_text, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities and get predictions
        probs = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probs).item()
        confidence = probs[0][predicted_label].item()

        return predicted_label, confidence
    
    def predict_batch(self, df):
        """
        Predict sentiment for a batch of text inputs in a DataFrame.
        
        Parameters:
        - df (DataFrame): DataFrame containing a column 'Review' for batch prediction.
        
        Returns:
        - DataFrame: Original DataFrame with 'Predicted' and 'Probability_Positive' columns added.
        """
        # Prepare dataset using HuggingFaceDatasetPreparer
        procssed_df = process_raw_data(df)
        dataset_preparer = HuggingFaceDatasetPreparer(procssed_df,self.model_path)
        hf_dataset = dataset_preparer.load_and_prepare_dataset()
        
        # Set up evaluation arguments for batch processing
        training_args = TrainingArguments(
            output_dir=self.model_path,
            per_device_eval_batch_size=BATCH_SIZE,  # Can adjust based on available memory
            no_cuda=not self.use_gpu
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args
        )
        
        # Perform predictions on the entire dataset
        predictions = trainer.predict(hf_dataset)
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1)
        
        # Get predicted labels and probabilities
        predicted_labels = torch.argmax(probs, axis=1).numpy()
        probability_positive = probs[:, 1].numpy()  # Probability of positive class
        
        # Add predictions to the original DataFrame
        df['Predicted'] = predicted_labels
        df['Probability_Positive'] = probability_positive

        return df
