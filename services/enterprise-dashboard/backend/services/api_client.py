"""
Enterprise Dashboard Backend - API Client
Legacy wrapper for backward compatibility
Now uses the new modular architecture
"""

from .main_api_client import MainAPIClient

# For backward compatibility, create an alias
APIClient = MainAPIClient