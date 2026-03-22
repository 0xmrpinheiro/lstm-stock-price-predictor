#!/usr/bin/env python3
"""Test script to verify model loading with backward-compatible InputLayer."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tf_keras
from tf_keras.models import load_model
from tf_keras.layers import InputLayer
from pathlib import Path


class BackwardCompatibleInputLayer(InputLayer):
    """Custom InputLayer that accepts the deprecated 'batch_shape' parameter
    for backward compatibility with models saved in older TensorFlow versions."""
    
    def __init__(self, **kwargs):
        # Convert batch_shape to input_shape if present
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            # batch_shape is [None, timesteps, features], input_shape is (timesteps, features)
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(**kwargs)


def test_model_loading():
    model_path = Path('Models') / 'neural_forecaster.keras'
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    print(f"Testing model load from: {model_path}")
    
    # Test with custom objects
    try:
        custom_objects = {'InputLayer': BackwardCompatibleInputLayer}
        model = load_model(str(model_path), custom_objects=custom_objects)
        print("SUCCESS: Model loaded successfully with custom InputLayer!")
        print(f"Model summary:")
        model.summary()
        return True
    except Exception as e:
        print(f"FAILED: Error loading model with custom InputLayer: {e}")
        return False


if __name__ == '__main__':
    success = test_model_loading()
    exit(0 if success else 1)