import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Read the CSV file
df = pd.read_csv("fifa_football_results.csv")

# Create a target variable representing the match outcome for the home team
df['home_team_result'] = df.apply(lambda row: 'win' if row['home_score'] > row['away_score'] 
                                  else ('lose' if row['home_score'] < row['away_score'] else 'draw'), axis=1)

# Encode the target variable
label_encoder = LabelEncoder()
df['home_team_result_encoded'] = label_encoder.fit_transform(df['home_team_result'])

# Selecting features - excluding actual scores
X = df[['home_team', 'away_team']]
y = df['home_team_result_encoded']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X), columns=encoder.get_feature_names_out(X.columns))

# Create the test data sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and fit a model (Logistical Regression)
classifier_log = LogisticRegression(max_iter=100)
classifier_log.fit(X_train, y_train)

# Create and fit a model (Random Forest)
classifier_rfc = RandomForestClassifier(n_estimators=100, max_depth=1000, n_jobs=-1)
classifier_rfc.fit(X_train, y_train)

# Predict using the model LR
y_pred_log = classifier_log.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log))

# Predict using the model RFC
y_pred_rfc = classifier_rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Random Forest Classifier Classification Report:\n", classification_report(y_test, y_pred_rfc))

# Function to predict match outcome
def predict_match_log(home_team, away_team):
    new_input = pd.DataFrame({'home_team': [home_team], 'away_team': [away_team]})
    new_input_encoded = pd.DataFrame(encoder.transform(new_input), columns=encoder.get_feature_names_out(X.columns))
    
    predicted_result_encoded = classifier_log.predict(new_input_encoded)
    predicted_result = label_encoder.inverse_transform(predicted_result_encoded)
    return predicted_result[0]

# Function to predict match outcome
def predict_match_rfc(home_team, away_team):
    new_input = pd.DataFrame({'home_team': [home_team], 'away_team': [away_team]})
    new_input_encoded = pd.DataFrame(encoder.transform(new_input), columns=encoder.get_feature_names_out(X.columns))
    
    predicted_result_encoded = classifier_rfc.predict(new_input_encoded)
    predicted_result = label_encoder.inverse_transform(predicted_result_encoded)
    return predicted_result[0]

# Taking user input for prediction
home_team_input = input('Enter your home team: ')
away_team_input = input('Enter your away team: ')

predicted_outcome_log = predict_match_log(home_team_input, away_team_input)
predicted_outcome_rfc = predict_match_rfc(home_team_input, away_team_input)

print("Logistic Regression Predicted Result for the home team:", predicted_outcome_log)
print("Random Forest Classifier Predicted Result for the home team:", predicted_outcome_rfc)
