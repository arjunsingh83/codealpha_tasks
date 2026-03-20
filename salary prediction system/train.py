import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("salary_data.csv")

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Rename columns (make sure dataset columns match this order)
df.columns = ['Age','Gender','Education','JobTitle','Experience','Salary']

# Remove missing values
df.dropna(inplace=True)

# Convert data types
df['Age'] = df['Age'].astype(int)
df['Experience'] = df['Experience'].astype(float)
df['Salary'] = df['Salary'].astype(float)

# One Hot Encoding
df = pd.get_dummies(df, columns=['Gender','Education','JobTitle'], drop_first=True)

# Split features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("features.pkl", "wb"))

print("Model Trained and Saved Successfully")