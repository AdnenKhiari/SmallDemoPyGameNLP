import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer, 
                          Trainer, 
                          TrainingArguments, 
                          EarlyStoppingCallback)
import torch
import evaluate

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'command': 'text', 'action': 'label'}, inplace=True)
    return df

# Preprocess the data
def preprocess_data(df):
    # Split the data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset

# Tokenization
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=32)

# Compute metrics
def compute_metrics(pred):
    # Load metrics using the Evaluate library
    metric_accuracy = evaluate.load("accuracy")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Compute metrics
    accuracy = metric_accuracy.compute(predictions=preds, references=labels)
    precision = metric_precision.compute(predictions=preds, references=labels, average='weighted')
    recall = metric_recall.compute(predictions=preds, references=labels, average='weighted')
    f1 = metric_f1.compute(predictions=preds, references=labels, average='weighted')

    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
    }

# Main function to execute the training
def main():
    # Load and preprocess the data
    file_path = 'augmented_commands.csv'  # Updated the path to use augmented_commands.csv
    df = load_dataset(file_path)
    
    # Map string labels to integers
    df['label'] = df['label'].astype('category').cat.codes
    
    train_dataset, val_dataset = preprocess_data(df)
    
    # Load tokenizer and model generically
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'  # Specify TinyBERT model here
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(df['label'].unique()))
    
    # Tokenize the datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',              # Output directory
        num_train_epochs=4,                 # Total number of training epochs
        per_device_train_batch_size=32,       # Batch size per device during training
        per_device_eval_batch_size=64,       # Batch size for evaluation
        warmup_steps=256,                     # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                   # Strength of weight decay
        logging_dir='./logs',                 # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",          # Evaluate every epoch
        save_strategy="epoch",                # Save model every epoch
        load_best_model_at_end=True,          # Load the best model at the end
        metric_for_best_model='f1',           # Metric to use for comparing models
        greater_is_better=True,                # Indicates if a higher score is better for the chosen metric
    )

    # Trainer with Early Stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add metrics function
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop training if no improvement for 2 evaluations
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained('./tinybert-trained-model')
    tokenizer.save_pretrained('./tinybert-trained-model')

if __name__ == "__main__":
    main()
