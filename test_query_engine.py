"""Test script for query engine"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_engine import test_query_engine

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    test_query_engine()
