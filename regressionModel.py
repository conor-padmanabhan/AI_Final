# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Step 2: Load and Clean Data
# Load the dataset
file_path = '/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/AI/AI_Final/deforestation_data - Sheet1.csv'
data = pd.read_csv(file_path)

# Replace missing/unknown values with NaN for consistency
data = data.replace(to_replace=[np.nan, 'unknown', 'Unknown'], value=np.nan)

# Drop rows with missing values for simplicity
data_cleaned = data.dropna()

# Convert numeric columns to proper types
data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce')

# Drop rows with remaining NaN values after conversion
data_cleaned = data_cleaned.dropna()

# Step 3: Prepare Target and Features
# Define target variable and features
X = data_cleaned.drop(columns=["Parà Deforested Area Agg. (km^2)"])
y = data_cleaned["Parà Deforested Area Agg. (km^2)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Fit a Basic Linear Regression Model (sklearn)
# Initialize linear regression model
lr_model = LinearRegression()

# Fit the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
print("Sklearn Linear Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2):", r2_score(y_test, y_pred))

# Step 5: Fit and Analyze with OLS Regression (statsmodels)
# Add constant to features for OLS
X_train_const = sm.add_constant(X_train)

# Fit the OLS model
ols_model = sm.OLS(y_train, X_train_const).fit()

# Display OLS summary
print("\nStatsmodels OLS Regression Summary:")
print(ols_model.summary())

# Step 6: Identify the Best Model
# Compare models based on R-squared, Adjusted R-squared, and feature significance
# Optional: Automate feature selection or test additional models