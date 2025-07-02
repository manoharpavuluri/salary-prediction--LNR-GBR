"""
Test script to validate the salary prediction model
This script tests the model's functionality and performance.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from save_model import load_and_preprocess_data, prepare_features
from predict_salary import SalaryPredictor

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🧪 Testing model loading...")
    
    try:
        predictor = SalaryPredictor()
        print("✅ Model loading test passed!")
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {str(e)}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\n🧪 Testing data preprocessing...")
    
    try:
        # Load and preprocess data
        train_merge = load_and_preprocess_data()
        X, cat_features, num_features = prepare_features(train_merge)
        
        print(f"✅ Data preprocessing test passed!")
        print(f"   - Features shape: {X.shape}")
        print(f"   - Categorical features: {len(cat_features)}")
        print(f"   - Numerical features: {len(num_features)}")
        return True
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {str(e)}")
        return False

def test_single_prediction():
    """Test single prediction functionality"""
    print("\n🧪 Testing single prediction...")
    
    try:
        predictor = SalaryPredictor()
        
        # Test with valid input
        test_input = {
            'jobType': 'SENIOR',
            'degree': 'MASTERS',
            'major': 'COMPSCI',
            'industry': 'FINANCE',
            'yearsExperience': 5,
            'milesFromMetropolis': 10
        }
        
        prediction = predictor.predict_single(**test_input)
        
        # Check if prediction is reasonable (positive number)
        if prediction > 0:
            print(f"✅ Single prediction test passed!")
            print(f"   - Input: {test_input}")
            print(f"   - Prediction: ${prediction:,.2f}")
            return True
        else:
            print(f"❌ Single prediction test failed: Invalid prediction value")
            return False
            
    except Exception as e:
        print(f"❌ Single prediction test failed: {str(e)}")
        return False

def test_batch_prediction():
    """Test batch prediction functionality"""
    print("\n🧪 Testing batch prediction...")
    
    try:
        predictor = SalaryPredictor()
        
        # Test with multiple inputs
        test_data = [
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
        
        predictions = predictor.predict(test_data)
        
        # Check if all predictions are reasonable
        if len(predictions) == len(test_data) and all(p > 0 for p in predictions):
            print(f"✅ Batch prediction test passed!")
            print(f"   - Number of predictions: {len(predictions)}")
            print(f"   - Predictions: {[f'${p:,.2f}' for p in predictions]}")
            return True
        else:
            print(f"❌ Batch prediction test failed: Invalid predictions")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction test failed: {str(e)}")
        return False

def test_model_performance():
    """Test model performance on validation data"""
    print("\n🧪 Testing model performance...")
    
    try:
        # Load and preprocess data
        train_merge = load_and_preprocess_data()
        X, cat_features, num_features = prepare_features(train_merge)
        y = train_merge['salary']
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Load predictor
        predictor = SalaryPredictor()
        
        # Make predictions on test set
        predictions = predictor.predict(test_x)
        
        # Calculate metrics
        mse = mean_squared_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)
        
        print(f"✅ Model performance test passed!")
        print(f"   - MSE: {mse:.2f}")
        print(f"   - R² Score: {r2:.4f}")
        print(f"   - Test set size: {len(test_y)}")
        
        # Check if performance is reasonable
        if r2 > 0.5 and mse > 0:
            return True
        else:
            print(f"❌ Model performance below expected threshold")
            return False
            
    except Exception as e:
        print(f"❌ Model performance test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n🧪 Testing error handling...")
    
    try:
        predictor = SalaryPredictor()
        
        # Test with missing required field
        try:
            invalid_input = {
                'jobType': 'SENIOR',
                'degree': 'MASTERS',
                # Missing 'major' field
                'industry': 'FINANCE',
                'yearsExperience': 5,
                'milesFromMetropolis': 10
            }
            predictor.predict_single(**invalid_input)
            print("❌ Error handling test failed: Should have raised an error")
            return False
        except Exception:
            print("✅ Error handling test passed: Correctly caught missing field error")
            return True
            
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting model validation tests...\n")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Data Preprocessing", test_data_preprocessing),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Performance", test_model_performance),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("="*50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("models/gbr_model.pkl"):
        print("❌ Model not found. Please run save_model.py first to train and save the model.")
    else:
        run_all_tests() 