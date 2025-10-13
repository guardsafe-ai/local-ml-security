"""
Logging Configuration
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", service_name: str = "model-api") -> None:
    """Setup logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        f'%(asctime)s - {service_name} - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)
