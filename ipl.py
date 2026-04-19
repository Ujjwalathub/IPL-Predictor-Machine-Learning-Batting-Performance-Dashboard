import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
df = pd.read_csv(r"E:\ok\IPL_Merged\IPL_Master_Dataset.csv", low_memory=False)

# 2. Filter for match-level batting stats 
batting_metrics = ['Most Fours Innings', 'Most Sixes Innings', 'Fastest Centuries', 'Fastest Fifties']
df_bat = df[df['Metric'].isin(batting_metrics)].copy()

# Drop NaNs in essential columns
df_bat = df_bat.dropna(subset=['Player', 'Runs', 'Against', 'Venue', 'Year'])

# Convert Runs to numeric 
df_bat['Runs'] = df_bat['Runs'].astype(str).str.replace('*', '', regex=False)
df_bat['Runs'] = pd.to_numeric(df_bat['Runs'], errors='coerce')
df_bat = df_bat.dropna(subset=['Runs'])

# Clean Year 
df_bat = df_bat[df_bat['Year'] != 'All']
df_bat['Year'] = pd.to_numeric(df_bat['Year'])

# 3. Select features and target
X = df_bat[['Player', 'Against', 'Venue', 'Year']].copy()
y = df_bat['Runs']

# 4. Encode Categorical Variables
le_player = LabelEncoder()
le_against = LabelEncoder()
le_venue = LabelEncoder()

X['Player_Encoded'] = le_player.fit_transform(X['Player'])
X['Against_Encoded'] = le_against.fit_transform(X['Against'])
X['Venue_Encoded'] = le_venue.fit_transform(X['Venue'])

features = ['Player_Encoded', 'Against_Encoded', 'Venue_Encoded', 'Year']
X_model = X[features]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)

# 6. Train Gradient Boosting Model (equivalent to LightGBM/XGBoost)
gb_model = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# 7. Predictions & Evaluation
y_pred = gb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# 8. Feature Importance
perm_importance = permutation_importance(gb_model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()
feature_names = np.array(['Player', 'Opponent', 'Venue', 'Year'])

# Plot 1: Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], color='teal', align='center')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.title("Gradient Boosting Feature Importance: Predicting Runs")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

# Plot 2: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='darkorange', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Gradient Boosting: Actual vs Predicted Runs")
plt.xlabel("Actual Runs")
plt.ylabel("Predicted Runs")
plt.tight_layout()
plt.show()

print(f"Data points available for modeling: {X_model.shape[0]}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} runs")
print(f"Mean Absolute Error (MAE): {mae:.2f} runs")