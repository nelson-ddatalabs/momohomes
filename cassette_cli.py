#!/usr/bin/env python3
"""
Cassette CLI - Command Line Interface
=====================================
Simple CLI wrapper for cassette optimization system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import main_cassette
sys.path.insert(0, str(Path(__file__).parent))

from main_cassette import main

if __name__ == "__main__":
    main()