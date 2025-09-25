#!/usr/bin/env python3
"""
Test script to diagnose model loading issues
"""

import tensorflow as tf
import joblib
import sys
import traceback

def test_model_loading():
    """Test model loading with detailed error reporting"""
    
    print("=" * 50)
    print("MODEL LOADING DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    try:
        # Clear any existing sessions
        print("\n1. Clearing Keras backend session...")
        tf.keras.backend.clear_session()
        print("   ✅ Session cleared successfully")
        
        # Try loading model without compilation
        print("\n2. Loading model without compilation...")
        model = tf.keras.models.load_model("pjm_lstm_model.keras", compile=False)
        print("   ✅ Model loaded successfully (without compilation)")
        
        # Get model summary
        print("\n3. Model architecture:")
        model.summary()
        
        # Try manual compilation
        print("\n4. Manually compiling model...")
        model.compile(
            optimizer='rmsprop',
            loss='mse',
            metrics=['mae']
        )
        print("   ✅ Model compiled successfully")
        
        # Test scaler loading
        print("\n5. Loading scaler...")
        scaler = joblib.load("pjm_scaler.pkl")
        print("   ✅ Scaler loaded successfully")
        print(f"   Scaler feature range: {scaler.feature_range}")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED - Model loading should work!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("❌ TEST FAILED - Please check the error above")
        print("=" * 50)
        
        return False

if __name__ == "__main__":
    test_model_loading()
