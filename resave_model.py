#!/usr/bin/env python3
"""One-time script to re-save the .keras model using tf_keras.

Run this on an environment with Python 3.13 + TF 2.21 + tf_keras 2.21
(e.g., Streamlit Cloud console or a compatible local setup).

After running, commit the updated Models/neural_forecaster.keras file.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tf_keras
from tf_keras.models import load_model
from pathlib import Path


class BackwardCompatibleInputLayer(tf_keras.layers.InputLayer):
    """Handles the deprecated 'batch_shape' parameter from older TF saves."""

    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        if 'batch_input_shape' in kwargs:
            batch_input_shape = kwargs.pop('batch_input_shape')
            if batch_input_shape and len(batch_input_shape) > 1:
                kwargs.setdefault('input_shape', batch_input_shape[1:])
        super().__init__(**kwargs)


def main():
    model_path = Path('Models') / 'neural_forecaster.keras'

    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        return False

    print(f"Loading model from {model_path} ...")
    custom_objects = {
        'InputLayer': BackwardCompatibleInputLayer,
        'tf_keras.layers.InputLayer': BackwardCompatibleInputLayer,
        'tensorflow.keras.layers.InputLayer': BackwardCompatibleInputLayer,
    }
    model = load_model(str(model_path), custom_objects=custom_objects)
    print("Model loaded successfully!")
    model.summary()

    # Re-save with tf_keras serialization (overwrites original)
    print(f"\nRe-saving model to {model_path} ...")
    model.save(str(model_path))
    print("Model re-saved successfully!")

    # Verify the re-saved model loads cleanly without custom_objects
    print("\nVerifying re-saved model loads without custom_objects ...")
    model2 = load_model(str(model_path))
    print("Verification passed! Model loads cleanly.")
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
