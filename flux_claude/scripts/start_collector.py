#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run collector
if __name__ == "__main__":
    try:
        from collector_multi_expiry import run_multi_expiry_collector, load_config
        config = load_config("config/config.yaml")
        run_multi_expiry_collector(config)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure collector_multi_expiry.py is in the same directory")
    except Exception as e:
        print(f"Error: {e}")
