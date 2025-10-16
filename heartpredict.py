import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
try:
    df = pd.read_csv('heart.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please download the dataset and place it in the same directory.")
    exit()

# Step 2: Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDistribution of the target variable ('target'):")
print(df['target'].value_counts())

# Step 3: Data Preprocessing
# Define features (X) and target (y)
# The 'target' column indicates the presence (1) or absence (0) of heart disease.
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Splitting ---")
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Step 4: Model Training
# Using a RandomForestClassifier for robust and accurate prediction.
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\n--- Training RandomForestClassifier ---")
model.fit(X_train, y_train)
print("Model training complete.")

# Step 5: Model Evaluation
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Save the trained model
model_filename = 'heart_disease_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved to '{model_filename}'.")

# Step 7 (Optional): Make a prediction on new data
print("\n--- Example Prediction ---")
# Example new patient data (replace with actual patient data)
# Order must match the features used for training: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
new_patient_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1] 
prediction = model.predict([new_patient_data])

if prediction[0] == 1:
    print("Prediction for new patient: High likelihood of heart disease.")
else:
    print("Prediction for new patient: Low likelihood of heart disease.")