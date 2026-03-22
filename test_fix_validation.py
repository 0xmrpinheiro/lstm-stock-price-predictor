#!/usr/bin/env python3
"""Test script to validate the BackwardCompatibleInputLayer fix without requiring TensorFlow."""

import sys
import os

def test_class_definition():
    """Test that the BackwardCompatibleInputLayer class is properly defined."""
    try:
        # Read the test_model_load.py file directly
        with open('test_model_load.py', 'r') as f:
            content = f.read()
        
        # Check if the class is defined with the correct inheritance
        if 'class BackwardCompatibleInputLayer(tf_keras.layers.InputLayer):' in content:
            print("✓ BackwardCompatibleInputLayer class has correct inheritance")
        else:
            print("✗ BackwardCompatibleInputLayer class does NOT have correct inheritance")
            return False
            
        # Check if it handles batch_shape conversion
        if "'batch_shape' in kwargs" in content and "kwargs['input_shape'] = batch_shape[1:]" in content:
            print("✓ BackwardCompatibleInputLayer class handles batch_shape conversion")
        else:
            print("✗ BackwardCompatibleInputLayer class does NOT handle batch_shape conversion")
            return False
            
        # Check if it calls super().__init__
        if "super().__init__(**kwargs)" in content:
            print("✓ BackwardCompatibleInputLayer class calls super().__init__")
        else:
            print("✗ BackwardCompatibleInputLayer class does NOT call super().__init__")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error reading test_model_load.py: {e}")
        return False

def test_streamlit_app_fix():
    """Test that the streamlit_app.py file has the same fix."""
    try:
        # Read the streamlit_app.py file directly
        with open('streamlit_app.py', 'r') as f:
            content = f.read()
        
        # Check if the class is defined with the correct inheritance
        if 'class BackwardCompatibleInputLayer(tf_keras.layers.InputLayer):' in content:
            print("✓ streamlit_app.py has correct BackwardCompatibleInputLayer inheritance")
        else:
            print("✗ streamlit_app.py does NOT have correct BackwardCompatibleInputLayer inheritance")
            return False
            
        # Check if it handles batch_shape conversion
        if "'batch_shape' in kwargs" in content and "kwargs['input_shape'] = batch_shape[1:]" in content:
            print("✓ streamlit_app.py BackwardCompatibleInputLayer handles batch_shape conversion")
        else:
            print("✗ streamlit_app.py BackwardCompatibleInputLayer does NOT handle batch_shape conversion")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error reading streamlit_app.py: {e}")
        return False

def main():
    print("Testing BackwardCompatibleInputLayer fix...")
    print("=" * 50)
    
    # Test class definition in test_model_load.py
    if not test_class_definition():
        print("\n✗ FAILED: test_model_load.py fix validation failed")
        return False
    
    print()
    
    # Test streamlit_app.py fix
    if not test_streamlit_app_fix():
        print("\n✗ FAILED: streamlit_app.py fix validation failed")
        return False
    
    print("\n✓ SUCCESS: All tests passed!")
    print("\nThe fix has been successfully applied to both files:")
    print("1. test_model_load.py - Fixed BackwardCompatibleInputLayer inheritance")
    print("2. streamlit_app.py - Fixed BackwardCompatibleInputLayer inheritance")
    print("\nThe fix addresses the 'batch_shape' compatibility issue by:")
    print("- Changing inheritance from 'InputLayer' to 'tf_keras.layers.InputLayer'")
    print("- Converting 'batch_shape' parameter to 'input_shape' in __init__")
    print("- Maintaining backward compatibility with older saved models")
    
    print("\nTo test the actual model loading:")
    print("1. Install TensorFlow (may require Python 3.11 or 3.12)")
    print("2. Run: python test_model_load.py")
    print("3. The model should now load without the 'batch_shape' error")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)