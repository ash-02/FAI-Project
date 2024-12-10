import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import ast
import joblib

csv_file_path = 'data/word_pitch_mappings.csv'
data = pd.read_csv(csv_file_path)

data = data.dropna()

data = data.drop_duplicates()

data['pitches'] = data['pitches'].apply(ast.literal_eval)

pitch_features = pd.DataFrame(data['pitches'].tolist(), columns=[f'Pitch_{i+1}' for i in range(len(data['pitches'][0]))])

X = pitch_features
Y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

base_estimator = DecisionTreeClassifier(max_depth=3)
adaboost_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=0.05)

adaboost_clf.fit(X_train, y_train)

joblib.dump(adaboost_clf, 'model/adaboost_classifier.pkl')

y_pred = adaboost_clf.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
