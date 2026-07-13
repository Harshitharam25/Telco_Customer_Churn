"""Unit tests for Telco Customer Churn model and data validation."""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


DATA_PATH = os.getenv('DATA_PATH', 'data/')


class TestDataLoading:
    """Test suite for data loading and validation."""
    
    def test_data_file_exists(self):
        """Verify dataset file exists at expected location."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        assert os.path.exists(csv_file), f"Dataset not found at {csv_file}"
    
    def test_data_shape(self):
        """Verify dataset has minimum expected rows and columns."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        assert df.shape[0] >= 6000, f"Expected 7K+ rows, got {df.shape[0]}"
        assert df.shape[1] >= 20, f"Expected 20+ columns, got {df.shape[1]}"
    
    def test_required_columns(self):
        """Verify all required columns exist."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        required_cols = ['Churn', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_no_all_null_columns(self):
        """Verify no columns are entirely null."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        null_counts = df.isnull().sum()
        for col in df.columns:
            assert null_counts[col] < len(df), f"Column {col} is entirely null"


class TestDataCleaning:
    """Test suite for data cleaning pipeline."""
    
    def test_numeric_conversion(self):
        """Verify TotalCharges converts to numeric correctly."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        assert df['TotalCharges'].dtype in ['float64', 'int64']
    
    def test_customerid_droppable(self):
        """Verify customerID can be dropped without error."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        assert 'customerID' in df.columns
        df.drop('customerID', axis=1, inplace=True)
        assert 'customerID' not in df.columns


class TestModelTraining:
    """Test suite for model training and evaluation."""
    
    def test_model_accuracy_threshold(self):
        """Verify logistic regression achieves minimum 75% accuracy."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        # Cleaning
        df.drop('customerID', axis=1, inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        
        # Feature engineering
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop('Churn_Yes', axis=1)
        y = df_encoded['Churn_Yes']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model training
        model = LogisticRegression(max_iter=5000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluation
        accuracy = model.score(X_test_scaled, y_test)
        
        assert accuracy >= 0.75, (
            f"Model accuracy {accuracy:.2%} is below 75% threshold. "
            f"Expected >= 0.75, got {accuracy:.2f}"
        )
        
        # Print for debugging
        print(f"\n✓ Model trained successfully")
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        print(f"✓ Model accuracy: {accuracy:.2%}")
    
    def test_model_features_created(self):
        """Verify feature engineering produces expected shape."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        df.drop('customerID', axis=1, inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Should have many more features after one-hot encoding
        assert df_encoded.shape[1] >= 30, "Expected 30+ features after encoding"


class TestDocumentation:
    """Test suite for documentation completeness."""
    
    def test_readme_exists(self):
        """Verify README.md exists."""
        assert os.path.exists('README.md'), "README.md not found"
    
    def test_readme_has_content(self):
        """Verify README has meaningful content."""
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Check for key sections
        assert 'Telco' in content, "README missing project name"
        assert 'churn' in content.lower(), "README missing churn reference"
        assert 'accuracy' in content.lower() or '%' in content, "README missing accuracy"
    
    def test_requirements_exists(self):
        """Verify requirements.txt exists."""
        assert os.path.exists('requirements.txt'), "requirements.txt not found"
    
    def test_requirements_has_dependencies(self):
        """Verify requirements.txt has required packages."""
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        required = ['pandas', 'scikit-learn', 'jupyter']
        for pkg in required:
            assert pkg.lower() in content.lower(), f"Missing {pkg} in requirements"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
