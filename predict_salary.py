"""
Script to make salary predictions using the trained Gradient Boosting Regressor model
This script loads the saved model and preprocessor, then makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Union

class SalaryPredictor:
    """Class for making salary predictions using the trained model"""
    
    def __init__(self, model_path: str = "models/gbr_model.pkl", 
                 preprocessor_path: str = "models/preprocessor_info.pkl"):
        """
        Initialize the SalaryPredictor
        
        Args:
            model_path: Path to the saved model file
            preprocessor_path: Path to the saved preprocessor info file
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor_info = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessor information"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            self.model = joblib.load(self.model_path)
            self.preprocessor_info = joblib.load(self.preprocessor_path)
            
            print("✅ Model and preprocessor loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data using the same transformations as training
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Extract features
            cat_features = self.preprocessor_info['cat_features']
            num_features = self.preprocessor_info['num_features']
            
            # Check if required columns exist
            required_cols = cat_features + num_features
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # One-hot encode categorical features
            encoded_data = data[num_features].join(
                pd.get_dummies(data[cat_features])
            )
            
            # Ensure all expected feature columns are present
            expected_features = self.preprocessor_info['feature_names']
            missing_features = [feat for feat in expected_features if feat not in encoded_data.columns]
            
            # Add missing features with zeros (for categories not present in new data)
            for feat in missing_features:
                encoded_data[feat] = 0
            
            # Reorder columns to match training data
            encoded_data = encoded_data[expected_features]
            
            return encoded_data
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            raise
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Make salary predictions
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            Array of predicted salaries
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            raise
    
    def predict_single(self, **kwargs) -> float:
        """
        Make a single salary prediction
        
        Args:
            **kwargs: Feature values (jobType, degree, major, industry, 
                     yearsExperience, milesFromMetropolis)
            
        Returns:
            Predicted salary
        """
        try:
            # Create DataFrame from kwargs
            data = pd.DataFrame([kwargs])
            predictions = self.predict(data)
            return float(predictions[0])
            
        except Exception as e:
            print(f"❌ Error making single prediction: {str(e)}")
            raise

def main():
    """Example usage of the SalaryPredictor"""
    
    # Check if model exists
    if not os.path.exists("models/gbr_model.pkl"):
        print("❌ Model not found. Please run save_model.py first to train and save the model.")
        return
    
    try:
        # Initialize predictor
        predictor = SalaryPredictor()
        
        # Example 1: Single prediction
        print("\n📊 Example 1: Single Prediction")
        sample_input = {
            'jobType': 'SENIOR',
            'degree': 'MASTERS',
            'major': 'COMPSCI',
            'industry': 'FINANCE',
            'yearsExperience': 5,
            'milesFromMetropolis': 10
        }
        
        prediction = predictor.predict_single(**sample_input)
        print(f"Input: {sample_input}")
        print(f"Predicted Salary: ${prediction:,.2f}")
        
        # Example 2: Multiple predictions
        print("\n📊 Example 2: Multiple Predictions")
        sample_data = [
            {
                'jobType': 'JUNIOR',
                'degree': 'BACHELORS',
                'major': 'ENGINEERING',
                'industry': 'AUTO',
                'yearsExperience': 2,
                'milesFromMetropolis': 25
            },
            {
                'jobType': 'CEO',
                'degree': 'DOCTORAL',
                'major': 'BUSINESS',
                'industry': 'FINANCE',
                'yearsExperience': 15,
                'milesFromMetropolis': 5
            }
        ]
        
        predictions = predictor.predict(sample_data)
        for i, (data, pred) in enumerate(zip(sample_data, predictions)):
            print(f"Input {i+1}: {data}")
            print(f"Predicted Salary: ${pred:,.2f}\n")
        
        # Example 3: Load from CSV file
        print("\n📊 Example 3: Predictions from CSV file")
        if os.path.exists("data/test_features.csv"):
            test_data = pd.read_csv("data/test_features.csv")
            # Remove unnecessary columns
            test_data = test_data.drop(['jobId', 'companyId'], axis=1, errors='ignore')
            
            # Make predictions on first 5 rows
            sample_test = test_data.head(5)
            predictions = predictor.predict(sample_test)
            
            print("Predictions for first 5 test records:")
            for i, pred in enumerate(predictions):
                print(f"Record {i+1}: ${pred:,.2f}")
        else:
            print("Test data file not found. Skipping CSV example.")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 