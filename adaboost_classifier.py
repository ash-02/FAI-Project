import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import ast
import joblib

# Load the dataset
csv_file_path = 'data/word_pitch_mappings.csv'
data = pd.read_csv(csv_file_path)

# Drop rows with any null values
data = data.dropna()

# Remove duplicate rows
data = data.drop_duplicates()

# Convert the `pitches` column from string to a list of floats
data['pitches'] = data['pitches'].apply(ast.literal_eval)

# Expand the `pitches` column into multiple feature columns
pitch_features = pd.DataFrame(data['pitches'].tolist(), columns=[f'Pitch_{i+1}' for i in range(len(data['pitches'][0]))])

# Combine the pitch features with the label
X = pitch_features
Y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the AdaBoost classifier
base_estimator = DecisionTreeClassifier(max_depth=3)  # Stump classifier
adaboost_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=0.05)

# Train the classifier
adaboost_clf.fit(X_train, y_train)

joblib.dump(adaboost_clf, 'model/adaboost_classifier.pkl')

# Make predictions
y_pred = adaboost_clf.predict(X_test)

# Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
