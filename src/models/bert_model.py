import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from ..config import MODEL_NAME, MODEL_SAVE_DIR, LEARNING_RATE, BATCH_SIZE, EPOCHS, RANDOM_SEED,USE_GPU,SAVE_LOSS_AND_PLOT
import matplotlib.pyplot as plt

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Save train and eval losses if they exist in logs
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])

def save_losses_and_plot(train_losses, eval_losses, output_dir=".", filename="losses.txt"):
    """
    Save training and validation losses to a text file and plot them.
    """
    # Save losses to a text file
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("Epoch\tTraining Loss\tValidation Loss\n")
        for epoch, (train_loss, eval_loss) in enumerate(zip(train_losses, eval_losses), start=1):
            f.write(f"{epoch}\t{train_loss}\t{eval_loss}\n")

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker='o')
    plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.show()

class BertFineTuner:
    def __init__(self):
        """
        Initialize the fine-tuner with model name and GPU usage.
        
        Parameters:
        - use_gpu (bool): If True, use GPU for training if available.
        """
        self.model_name = MODEL_NAME
        self.use_gpu = USE_GPU
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model_path = self._generate_model_path()
        
    def _generate_model_path(self):
        """
        Generate a unique model path based on versioning.
        
        Returns:
        - str: The model path with versioning.
        """
        version = 1
        model_path = os.path.join(MODEL_SAVE_DIR, f"{self.model_name}_v{version}")
        while os.path.exists(model_path):
            version += 1
            model_path = os.path.join(MODEL_SAVE_DIR, f"{self.model_name}_v{version}")
        return model_path

    def get_training_args(self):
        """
        Set up training arguments for the Trainer.
        
        Returns:
        - TrainingArguments: Training arguments for model training.
        """
        return TrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            logging_dir=f"{self.model_path}/logs",
            seed=RANDOM_SEED,
            load_best_model_at_end=True,
            no_cuda=not self.use_gpu  # Disable CUDA if GPU is not to be used
        )

    def train(self, train_dataset, val_dataset):
        """
        Train the model on the given datasets.
        
        Parameters:
        - train_dataset (Dataset): Training dataset.
        - val_dataset (Dataset): Validation dataset.
        """
        training_args = self.get_training_args()
        loss_logger = LossLoggerCallback()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[loss_logger]
        )

        # Train the model
        trainer.train()
        if SAVE_LOSS_AND_PLOT:
            # Save losses and plot
            save_losses_and_plot(loss_logger.train_losses, loss_logger.eval_losses, output_dir=self.model_path)
        
        # Save the final model and tokenizer
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print(f"Model saved to {self.model_path}")
