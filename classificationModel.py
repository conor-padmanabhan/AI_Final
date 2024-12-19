import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load and clean data
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    target_col = "Parà Deforested Area Agg. (km^2)"

    # Replace missing/unknown values and drop missing rows
    data_cleaned = data.replace(to_replace=[np.nan, 'unknown', 'Unknown'], value=np.nan).dropna()

    # Define the target (binary classification: High vs. Low deforestation)
    median_deforestation = data_cleaned[target_col].median()
    data_cleaned['DeforestationLevel'] = np.where(data_cleaned[target_col] > median_deforestation, 1, 0)

    # Define features and target
    X = data_cleaned.drop(columns=[target_col, 'DeforestationLevel'])
    y = data_cleaned['DeforestationLevel']

    return X, y

# Function to run K-NN
def run_knn(X_train, y_train, X_test, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    return knn_model.predict(X_test)

# Function to run Perceptron
def run_perceptron(X_train, y_train, X_test, eta, max_iter):
    perceptron_model = Perceptron(eta0=eta, max_iter=max_iter, random_state=42)
    perceptron_model.fit(X_train, y_train)
    return perceptron_model.predict(X_test)

# Evaluation function
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall}

# Main function
def main():
    # Load and prepare the data
    file_path = '/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/AI/AI_Final/deforestation_data_set_final - Sheet1 (1).csv'
    X, y = load_and_prepare_data(file_path)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Run K-NN
    knn_predictions = run_knn(X_train, y_train, X_test, k=5)
    knn_metrics = evaluate_model(y_test, knn_predictions)
    print("K-NN Results:", knn_metrics)

    # Run Perceptron
    perceptron_predictions = run_perceptron(X_train, y_train, X_test, eta=0.01, max_iter=1000)
    perceptron_metrics = evaluate_model(y_test, perceptron_predictions)
    print("Perceptron Results:", perceptron_metrics)


main()