import pandas as pd
import pickle
from textblob import TextBlob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define command classes
class Command:
    JUMP = "jump"
    MOVE_LEFT = "move left"
    MOVE_RIGHT = "move right"
    CHANGE_COLOR = "change color"

# Base Class for Command Processors
class CommandProcessor:
    
    def process_command(self,sentence: str):
        blob = TextBlob(sentence)
        corrected_sentence = str(blob.correct())
        return self._process_command(corrected_sentence)
    def _process_command(self, command: str):
        raise NotImplementedError("This method should be overridden by subclasses.")

# Rule-Based Command Processor
class RuleBasedProcessor(CommandProcessor):
    def _process_command(self, command: str):
        command = command.lower().strip()
        if "jump" in command:
            return Command.JUMP
        elif "left" in command:
            return Command.MOVE_LEFT
        elif "right" in command:
            return Command.MOVE_RIGHT
        elif "color" in command:
            return Command.CHANGE_COLOR
        return None

# Naive Bayes Command Processor
class NaiveBayesProcessor(CommandProcessor):
    def __init__(self, model_path):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)  # Load the entire pipeline model

    def _process_command(self, command: str):
        # The model pipeline handles the vectorization internally
        prediction = self.model.predict([command])  # Pass command as a list
        return prediction[0]  # Return the predicted action

# TinyBERT Command Processor
class TinyBERTProcessor(CommandProcessor):
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set the model to evaluation mode
        self.label_map = {
        1: Command.JUMP,
        2: Command.MOVE_LEFT,
        3: Command.MOVE_RIGHT,
        0: Command.CHANGE_COLOR
    }  # Map label IDs to commands

    def _process_command(self, command: str):
        inputs = self.tokenizer(command, return_tensors='pt', padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            label_id = predictions.item()
            return self.label_map[label_id]  # Return the mapped command

# Example usage
if __name__ == "__main__":
    # Load each processor
    rule_processor = RuleBasedProcessor()
    naive_bayes_processor = NaiveBayesProcessor('augmented_model.pkl')
    tinybert_processor = TinyBERTProcessor('./tinybert-trained-model', 'huawei-noah/TinyBERT_General_4L_312D')

    # Test the processors with more challenging commands
    commands = [
        "jump",
        "move left",
        "move right",
        "do not move right",
        "change color",
        "go to the left",
        "Please jump high",
        "Could you move to the left, please?",
        "Can you move right?",
        "I want to change the color.",
        "Jump as high as you can!", 
        "Shift to the left side", 
        "Quickly move right!", 
        "Change the color to blue"
    ]

    for command in commands:
        print(f"Command: {command}")
        print(f"Rule-Based Output: {rule_processor._process_command(command)}")
        print(f"Naive Bayes Output: {naive_bayes_processor._process_command(command)}")
        print(f"TinyBERT Output: {tinybert_processor._process_command(command)}")
        print()
