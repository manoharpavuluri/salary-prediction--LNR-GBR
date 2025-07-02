"""
Script to save the trained Gradient Boosting Regressor model
This script loads the data, preprocesses it, trains the model, and saves it for future use.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the training data"""
    print("Loading data...")
    
    # Load data
    train_features_df = pd.read_csv("data/train_features.csv")
    train_target_df = pd.read_csv("data/train_salaries.csv")
    
    # Merge features and target
    train_merge = pd.merge(train_features_df, train_target_df, on='jobId')
    
    # Remove records with zero salaries
    train_merge = train_merge[train_merge.salary != 0]
    
    print(f"Data loaded: {len(train_merge)} records")
    return train_merge

def prepare_features(train_merge):
    """Prepare features for training"""
    print("Preparing features...")
    
    # Define features
    cat_features = ['jobType', 'degree', 'major', 'industry']
    num_features = ['yearsExperience', 'milesFromMetropolis']
    
    # One-hot encode categorical features
    train_merge_Hot_Enc = train_merge[num_features].join(
        pd.get_dummies(train_merge[cat_features])
    )
    
    print(f"Features prepared: {train_merge_Hot_Enc.shape[1]} features")
    return train_merge_Hot_Enc, cat_features, num_features

def train_model(X, y):
    """Train the Gradient Boosting Regressor model"""
    print("Training model...")
    
    # Split data
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Initialize and train model
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
    gbr.fit(train_x, train_y)
    
    # Evaluate model
    train_score = gbr.score(train_x, train_y)
    test_score = gbr.score(test_x, test_y)
    
    print(f"Training R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")
    
    return gbr, train_x, test_x, train_y, test_y

def save_model_and_preprocessor(model, cat_features, num_features):
    """Save the trained model and preprocessing information"""
    print("Saving model and preprocessor...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, "models/gbr_model.pkl")
    
    # Save preprocessing information
    preprocessor_info = {
        'cat_features': cat_features,
        'num_features': num_features,
        'feature_names': model.feature_names_in_.tolist()
    }
    joblib.dump(preprocessor_info, "models/preprocessor_info.pkl")
    
    print("Model and preprocessor saved successfully!")

def main():
    """Main function to execute the model training and saving pipeline"""
    try:
        # Load and preprocess data
        train_merge = load_and_preprocess_data()
        
        # Prepare features
        X, cat_features, num_features = prepare_features(train_merge)
        y = train_merge['salary']
        
        # Train model
        model, train_x, test_x, train_y, test_y = train_model(X, y)
        
        # Save model and preprocessor
        save_model_and_preprocessor(model, cat_features, num_features)
        
        print("\n✅ Model training and saving completed successfully!")
        print("📁 Files saved:")
        print("   - models/gbr_model.pkl (trained model)")
        print("   - models/preprocessor_info.pkl (preprocessing info)")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 