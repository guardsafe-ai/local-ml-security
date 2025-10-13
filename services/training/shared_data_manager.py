"""
Shared Data Manager Instance
Ensures all routes use the same EfficientDataManager instance
"""

from efficient_data_manager import EfficientDataManager

# Create a single shared instance
shared_data_manager = EfficientDataManager()
