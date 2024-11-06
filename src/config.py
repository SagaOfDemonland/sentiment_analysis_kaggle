# config.py
MODEL_NAME = 'bert-base-uncased'
RANDOM_SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MODEL_SAVE_DIR = '../data/models/'
USE_GPU = False
SAVE_LOSS_AND_PLOT = True
# gradient_accumulation_steps=2,
# lr_scheduler_type='linear',