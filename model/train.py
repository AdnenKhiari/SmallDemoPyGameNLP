import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset (assuming a 'augmented_commands.csv' file)
final_data = pd.read_csv('C:/Users/Adnen Kh/Documents/DenmarkStage/augmented_commands.csv')

print(final_data)

# Split data into features (commands) and labels (actions)
X = final_data['command']  # Change to 'text' column for the new structure
y = final_data['action']  # Change to 'label' column for the new structure

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to vectorize the text and train a Naive Bayes classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Converts text into numerical features
    ('classifier', MultinomialNB())     # Naive Bayes classifier
])

# Train the model on the augmented data
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model accuracy after augmentation: {accuracy * 100:.2f}%')

# Save the model for future use
with open('augmented_model.pkl', 'wb') as f:
    pickle.dump(model, f)
