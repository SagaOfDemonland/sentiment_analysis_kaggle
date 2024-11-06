import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove newlines and extra whitespace
    text = ' '.join(text.split())
    
    return text

def process_raw_data(df):
    # Apply preprocess_text function to the 'Review' column
    df['Review'] = df['Review'].apply(lambda x: preprocess_text(x))
    return df

def plot_auc_pr_curves(true_labels, predicted_probabilities):
    """
    Plot the ROC and Precision-Recall curves based on true labels and predicted probabilities for the positive class.

    Parameters:
    - true_labels (array-like): Actual binary labels (0 or 1) for the data.
    - predicted_probabilities (array-like): Predicted probabilities for the positive class (class 1).

    Returns:
    - None: Displays the ROC and Precision-Recall curves.
    """
    # Calculate the ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probabilities)

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()