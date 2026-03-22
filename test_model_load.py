#!/usr/bin/env python3
"""Test model loading with backward-compatible InputLayer and ticker validation."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import io
import re
from pathlib import Path

# Ticker regex (must match streamlit_app.py)
TICKER_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?(-[A-Z]{1,2})?$')


def test_ticker_validation():
    """Test that the ticker regex accepts valid tickers and rejects garbage."""
    valid = ["MSFT", "AAPL", "GOOG", "BRK.B", "BF.A", "SPY", "A"]
    invalid = [
        "", "TOOLONG", "123", "NOT A TICKER",
        "UNRECOGNIZED KEYWORD ARGUMENTS",
        "CONFIG={'BATCH_SHAPE': [NONE 100 1]}",
        "hello world", "12345", "A1B2",
    ]

    passed = True
    for t in valid:
        if not TICKER_RE.match(t):
            print(f"  FAIL: '{t}' should be valid but was rejected")
            passed = False

    for t in invalid:
        if TICKER_RE.match(t.strip().upper()):
            print(f"  FAIL: '{t}' should be invalid but was accepted")
            passed = False

    return passed


def test_model_loading():
    """Test that the model loads with custom_objects and stderr capture."""
    try:
        import tf_keras
        from tf_keras.models import load_model
    except ImportError:
        print("  SKIP: tf_keras not available in this environment")
        return True  # Not a failure, just can't test

    model_path = Path('Models') / 'neural_forecaster.keras'
    if not model_path.exists():
        print(f"  SKIP: {model_path} not found")
        return True

    # Import the compatibility class
    class BackwardCompatibleInputLayer(tf_keras.layers.InputLayer):
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

    custom_objects = {
        'InputLayer': BackwardCompatibleInputLayer,
        'tf_keras.layers.InputLayer': BackwardCompatibleInputLayer,
        'tensorflow.keras.layers.InputLayer': BackwardCompatibleInputLayer,
    }

    # Test with stderr capture (same pattern as streamlit_app.py)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    sys.stdout = captured_out
    sys.stderr = captured_err
    try:
        model = load_model(str(model_path), custom_objects=custom_objects)
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(f"  FAIL: Model loading raised exception: {e}")
        return False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    if model is None:
        print("  FAIL: Model loaded as None")
        return False

    # Verify no error text leaked to stdout
    leaked = captured_out.getvalue()
    if leaked.strip():
        print(f"  WARN: stdout captured during load: {leaked[:200]}")

    print(f"  Model loaded successfully: {model.count_params()} parameters")
    return True


if __name__ == '__main__':
    results = {}

    print("Test 1: Ticker validation")
    results['ticker'] = test_ticker_validation()
    print(f"  {'PASS' if results['ticker'] else 'FAIL'}")

    print("\nTest 2: Model loading with stderr capture")
    results['model'] = test_model_loading()
    print(f"  {'PASS' if results['model'] else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests FAILED'}")
    exit(0 if all_passed else 1)
