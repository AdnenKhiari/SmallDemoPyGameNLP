import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

# Download required NLTK data (you only need to run this once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to get synonyms based on POS and filter by frequency
def get_synonyms_by_pos_filtered(word, pos, min_freq=3):
    """
    Get synonyms for a word with the same part of speech and filter out rare synonyms.
    
    Args:
    word (str): The word for which to generate synonyms.
    pos (str): The part of speech for filtering synonyms (e.g., wn.VERB, wn.NOUN).
    min_freq (int): The minimum frequency threshold for filtering uncommon synonyms.

    Returns:
    List[str]: A list of filtered synonyms.
    """
    synonyms = set()
    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            # Only add synonym if its frequency is above the threshold and not the original word
            if lemma.count() >= min_freq and synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

# Function to augment a command by replacing verbs with their filtered synonyms
def augment_command_with_filtering(command):
    command_words = command.split()
    augmented_commands = []

    for i, word in enumerate(command_words):
        # Dynamically generate verb synonyms with filtering
        synonyms = get_synonyms_by_pos_filtered(word, wn.VERB)
        
        # If we have valid synonyms, create new command variations
        if synonyms:
            new_commands = [command_words[:i] + [syn] + command_words[i + 1:] for syn in synonyms]
            augmented_commands.extend(new_commands)

    return [' '.join(cmd) for cmd in augmented_commands]

# Augment the entire dataset using filtered synonyms
def augment_dataset_with_filtering(data):
    """
    Augment the dataset by generating new commands with synonyms for key words.

    Args:
    data (DataFrame): A pandas DataFrame containing 'command' and 'action' columns.

    Returns:
    DataFrame: A new DataFrame with augmented commands.
    """
    augmented_data = []
    for index, row in data.iterrows():
        command = row['command']
        action = row['action']
        # Generate augmented commands with filtered synonyms
        augmented_commands = augment_command_with_filtering(command)
        for augmented_command in augmented_commands:
            augmented_data.append({'command': augmented_command, 'action': action})

    return pd.DataFrame(augmented_data)

# Main function to load, augment, and save the dataset
def main():
    # Load dataset (replace 'commands.csv' with the path to your dataset)
    data = pd.read_csv('commands.csv',sep=';')

    # Augment the dataset
    augmented_data = augment_dataset_with_filtering(data)

    # Combine original and augmented data, removing duplicates
    final_data = pd.concat([data, augmented_data]).drop_duplicates().reset_index(drop=True)

    # Save the final augmented dataset to a new CSV file
    final_data.to_csv('augmented_commands.csv', index=False)

    print("Dataset augmentation complete! Augmented data saved to 'augmented_commands.csv'.")

if __name__ == "__main__":
    main()
