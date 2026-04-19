import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Data
df = pd.read_csv(r"E:\ok\archive\student_performance_dataset.csv")

# 2. Preprocess the Data
# Drop Student_ID (not predictive) and Final_Exam_Score (since we are predicting Pass_Fail directly)
X = df.drop(columns=['Student_ID', 'Final_Exam_Score', 'Pass_Fail'])
y = df['Pass_Fail']

# Convert categorical variables into dummy/indicator variables (One-Hot Encoding)
categorical_cols = ['Gender', 'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 3. Split the Data into Training and Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = rf_model.predict(X_test)

# 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(report)

# 7. Generate Visualizations

# Visualization 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['Pass', 'Fail'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'])
plt.title('Confusion Matrix - Random Forest (Pass vs Fail)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Visualization 2: Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('Feature Importance in Predicting Pass/Fail')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()