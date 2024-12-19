import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load and clean data
file_path = '/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/AI/AI_Final/AI Final Data - master-data-set.csv'
data = pd.read_csv(file_path)

# Replace missing/unknown values with NaN and drop missing rows
data_cleaned = data.replace(to_replace=[np.nan, 'unknown', 'Unknown'], value=np.nan).dropna()

data_cleaned.drop('Air Temperature (K)', axis=1, inplace=True)
data_cleaned.drop('Year', axis=1, inplace=True)
data_cleaned.drop('Sum of Firespots', axis=1, inplace=True)
data_cleaned.drop('Daytime surface temperature (K)', axis=1, inplace=True)

# Define the target and features
target_col = "Parà Deforested Area Agg. (km^2)"  # Adjust to match your column name
X = data_cleaned.drop(columns=[target_col])
y = data_cleaned[target_col]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Function to get the most important feature and display a table of coefficients
def get_most_important_feature(X: pd.DataFrame, y: pd.Series) -> str:
    model = LinearRegression()
    model.fit(X, y)
    
    # Create a DataFrame for coefficients
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coefficients['AbsCoefficient'] = coefficients['Coefficient'].abs()
    
    # Sort coefficients by absolute value in descending order
    coefficients = coefficients.sort_values(by='AbsCoefficient', ascending=False)
    
    # Display the table of features and coefficients
    print("\nTable of Features and Coefficients:")
    print(coefficients)
    
    # Find the most important feature (highest absolute coefficient)
    most_important_feature = coefficients.iloc[0]['Feature']
    return most_important_feature

# Calculate the most important feature
most_important_feature = get_most_important_feature(X, y)
print("\nMost important feature:", most_important_feature)