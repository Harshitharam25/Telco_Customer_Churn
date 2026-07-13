"""Unit tests for Telco Customer Churn model and data validation."""

import pytest
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = os.getenv('DATA_PATH', 'data/')


class TestDataValidation:
    """Tests for data loading and validation."""
    
    def test_data_file_exists(self):
        """Verify dataset file exists."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        assert os.path.exists(csv_file), f"Dataset not found at {csv_file}"
    
    def test_data_shape(self):
        """Verify dataset shape meets requirements."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        assert df.shape[0] >= 6000, f"Expected 7K+ rows, got {df.shape[0]}"
        assert df.shape[1] >= 20, f"Expected 20+ columns, got {df.shape[1]}"
    
    def test_required_columns_exist(self):
        """Verify all required columns present."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        required = ['Churn', 'tenure', 'MonthlyCharges', 'Contract']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


class TestDataCleaning:
    """Tests for data cleaning operations."""
    
    def test_numeric_conversion(self):
        """Verify TotalCharges converts to numeric."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        assert df['TotalCharges'].dtype in ['float64', 'int64']
    
    def test_customerid_column_removable(self):
        """Verify customerID can be dropped."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        assert 'customerID' in df.columns
        df = df.drop('customerID', axis=1)
        assert 'customerID' not in df.columns


class TestModelTraining:
    """Tests for model training and accuracy."""
    
    def test_model_accuracy_meets_threshold(self):
        """Verify model achieves 75%+ accuracy."""
        csv_file = os.path.join(DATA_PATH, 'Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file)
        
        # Clean data
        df = df.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        # Encode
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop('Churn_Yes', axis=1)
        y = df['Churn_Yes']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        
        # Verify
        accuracy = model.score(X_test, y_test)
        assert accuracy >= 0.75, f"Accuracy {accuracy:.2%} below 75% threshold"


class TestDocumentation:
    """Tests for documentation completeness."""
    
    def test_readme_exists(self):
        """Verify README.md exists."""
        assert os.path.exists('README.md'), "README.md not found"
    
    def test_readme_has_required_sections(self):
        """Verify README has key sections."""
        with open('README.md', 'r') as f:
            content = f.read()
        assert 'accuracy' in content.lower(), "README missing accuracy metrics"
        assert 'Churn' in content, "README missing project name"
    
    def test_requirements_exists(self):
        """Verify requirements.txt exists."""
        assert os.path.exists('requirements.txt'), "requirements.txt not found"
    
    def test_requirements_has_key_packages(self):
        """Verify key packages are specified."""
        with open('requirements.txt', 'r') as f:
            content = f.read()
        assert 'pandas' in content, "Missing pandas"
        assert 'scikit-learn' in content, "Missing scikit-learn"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
