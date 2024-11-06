import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, accuracy_score
from ..utils.helpers import plot_auc_pr_curves
from ..config import BATCH_SIZE
class BertModelEvaluator:
    def __init__(self, model_path, use_gpu=True):
        """
        Initialize the evaluator with model path and GPU usage.
        
        Parameters:
        - model_path (str): Path to the fine-tuned model.
        - use_gpu (bool): If True, use GPU for evaluation if available.
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        
        # Set device based on GPU availability
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.model = self.model.to(self.device)
        
    def evaluate(self, test_dataset, show_plot=False):
        """
        Evaluate the model on the given test dataset and calculate metrics.
        
        Parameters:
        - test_dataset (Dataset): The test dataset for evaluation.
        - show_plot (bool): If True, display AUC and Precision-Recall curves.
        
        Returns:
        - dict: Dictionary with confusion matrix and accuracy score.
        """
        # Set up evaluation arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            per_device_eval_batch_size=BATCH_SIZE,
            no_cuda=not self.use_gpu  # Disable CUDA if GPU is not used
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args
        )
        
        # Perform prediction on the test dataset
        predictions = trainer.predict(test_dataset)
        
        # Extract logits and apply softmax to get probabilities
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1)
        
        # Get predicted labels and probabilities for the positive class
        predicted_labels = torch.argmax(probs, axis=1).numpy()
        probability_positive = probs[:, 1].numpy()  # Probability of class 1 (positive)
        
        # Load test data into a DataFrame to store predictions
        test_df = test_dataset.to_pandas()
        test_df['Predicted'] = predicted_labels
        test_df['Probability_Positive'] = probability_positive
        
        # Calculate confusion matrix and accuracy
        conf_matrix = confusion_matrix(test_df['Liked'], test_df['Predicted'])
        accuracy = accuracy_score(test_df['Liked'], test_df['Predicted'])
        
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nAccuracy:", accuracy)
        
        # Plot ROC and Precision-Recall curves if specified
        if show_plot:
            plot_auc_pr_curves(test_df['Liked'], test_df['Probability_Positive'])
        
        return {
            "confusion_matrix": conf_matrix,
            "accuracy": accuracy
        }