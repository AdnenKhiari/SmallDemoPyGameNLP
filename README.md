# Command Processor Game

## Overview

This project involves the development of a simple command-based game using Python's Pygame library. The game utilizes natural language processing (NLP) models to interpret user commands and control a character on the screen.

First train the models uner the train module,
Then Run the main.py to get started !

## Data Generation

For the data generation phase, we utilized **ChatGPT** to create a diverse dataset of commands associated with specific actions. The dataset includes a variety of intuitive commands and complex phrases to ensure robustness in command processing. ( The Quality is not that great, but works for the demo :D )

### Data Generation Process
1. **Command Creation**: 
   - We engaged ChatGPT to generate a wide range of commands that are intuitive and relevant to the actions we want the game character to perform. This included direct commands and more complex phrases.

#### Sample Commands
- **Intuitive Commands**:
  - "jump"
  - "move left"
  - "move right"
  - "change color"

- **Complex Commands**:
  - "Please jump high."
  - "Could you move to the left, please?"
  - "I want to change the color."

## Data Augmentation

For the data augmentation phase, we enriched the initial dataset using **synonyms** sourced from **WordNet**.

### Data Augmentation Process
1. **Synonym Generation**:
   - For each command in the original dataset, we used WordNet to identify synonyms. This involved querying the WordNet database for words that have similar meanings to each command.
2. **Selection of Common Synonyms**:
   - We focused on the most common synonyms, ensuring that they are widely understood.
3. **Augmented Command Examples**:
   - Original: "move left"
     - Augmented: "shift left", "go to the left", "slide left"
   - Original: "jump"
     - Augmented: "leap", "spring", "bounce"

## Model Development

Three different models were implemented to handle command processing, each increasing in complexity and ability to capture the nuances of user input:

1. **Rule-Based Processor**: 
   - **Methodology**: This model uses simple keyword matching to interpret commands. It operates on a predefined set of rules that identify specific keywords associated with actions. 
   - **Strengths**: Fast and easy to implement; effective for straightforward commands and low latency.
   - **Limitations**: Lacks flexibility and cannot understand variations beyond the predefined rules. ( Does not work with go up )

2. **Naive Bayes Processor**: 
   - **Methodology**: A machine learning model that employs the Naive Bayes classification algorithm, trained on the augmented dataset.
   - **Strengths**: Better accuracy than the rule-based approach; can handle some variations in command phrasing.
   - **Limitations**: Still limited in understanding complex sentence structures and context.

3. **TinyBERT Processor**: 
   - **Methodology**: A transformer-based model that leverages the capabilities of BERT (Bidirectional Encoder Representations from Transformers). This model is fine-tuned for sequence classification
   - **Strengths**: Superior performance in recognizing and classifying commands; can effectively interpret complex and varied expressions.
   - **Limitations**: Requires more computational resources and is slower compared to simpler models.

All Models were chosen to run on the user device to avoid the round trip latency in case we created a flask server
## Future Enhancements

### Game Enhancements

1. **Sound Integration**:
   - Adding sound effects can significantly enhance the player experience. Sounds can be tied to specific actions, such as jumping, moving, or changing color. Background music can also create a more immersive atmosphere. Using a sound library or integrating a sound engine can help manage and trigger audio effects in response to player actions.

2. **Level Design**:
   - To create a more engaging gameplay experience, introducing multiple levels or environments is crucial. Levels could have unique themes, challenges, and objectives. By utilizing a game engine like **Unity**, you can take advantage of advanced graphics, physics, and user interface design. Unity’s powerful tools for level design would allow for dynamic environments and interactive elements, which can respond to player actions in real time.

3. **Visual Enhancements**:
   - Enhancing the graphics of the game can also improve player engagement. This could include implementing animations for character movements, designing detailed backgrounds, and creating visually appealing menus. Using 2D sprites or 3D models, depending on the game's direction, can make the game visually striking.

### NLP Improvements

1. **Dialogue Management**:
   - Implementing a dialogue management system can help create more engaging interactions between players and the game. This system would allow for back-and-forth conversations, where players can ask questions or give commands that depend on previous interactions. This could be particularly useful for non-linear gameplay, where the narrative changes based on player decisions.

2. **Sentiment Analysis**:
   - Integrating sentiment analysis could enhance player interactions by allowing the game to respond differently based on the emotional tone of the input. For instance, if a player expresses frustration (e.g., "I'm stuck!"), the game could offer hints or change the difficulty level in response.

3. **Training with better Datasets**:
   - Increasing the size and diversity of the training dataset will improve the NLP models' generalization capabilities. This could involve crowd-sourcing commands or using data from forums and gaming communities to ensure that a wide range of phrases and variations are included in the training process. Also the dataset used here does not have all variations possible so we do much better !

4. **Voice Command Recognition**:
   - Integrating voice recognition technology will allow players to issue commands using their voice, creating a hands-free gaming experience. This would involve using libraries such as Mozilla's DeepSpeech or Google's Speech-to-Text API to convert spoken commands into text that the NLP model can process.

5. **Multi-Lingual Support**:
   - Expanding the game to support multiple languages can attract a wider audience. This would involve training the NLP models on multilingual datasets, enabling them to recognize and process commands in various languages.

6. **Adaptive Learning**:
   - Implementing adaptive learning techniques will allow the system to improve over time based on user interactions. By analyzing how players use commands, the NLP model can adjust its understanding and become more effective at recognizing player intents.

**Note**: Gen AI was used in the project to help generate the code , I Decide about the project structure and decisions and paramters , he helps with the boiler plate code :D , and i make the final touches !