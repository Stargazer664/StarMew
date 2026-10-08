#!/usr/bin/env python
import sys
print("Python version:", sys.version)

try:
    print("1. Importing backend.app...")
    from backend import app
    print("2. SUCCESS: App imported!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
