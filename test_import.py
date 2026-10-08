#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

try:
    from backend import app
    print("SUCCESS: App imported!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
