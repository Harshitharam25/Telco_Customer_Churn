import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = os.getenv('DATA_PATH', 'data/')

def test_data_exists():
    """Test that dataset exists and loads."""
    csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
    assert os.path.exists(csv_file), f"Dataset not found at {csv_file}"

def test_data_loading():
    """Test that dataset loads and has expected shape."""
    csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
    df = pd.read_csv(csv_file)
    
    assert df.shape[0] >= 6000, "Dataset should have 7K+ rows"
    assert "Churn" in df.columns, "Churn column missing"
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

def test_model_accuracy():
    """Test that model achieves at least 75% accuracy."""
    csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
    df = pd.read_csv(csv_file)
    
    # Data cleaning
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"✓ Model accuracy: {accuracy:.2%}")
    assert accuracy >= 0.75, f"Accuracy {accuracy:.2%} below 75% threshold"
