import logging
import sys
import os
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(log_dir="logs", log_file="conv_pyboy.log", log_level=logging.INFO, max_size_mb=50, backup_count=5):
    """
    Set up logging to file and console with rotation.
    
    Args:
        log_dir (str): Directory to store log files
        log_file (str): Name of the log file
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        max_size_mb (int): Maximum size of log file in MB before rotation
        backup_count (int): Number of backup files to keep
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # Define log format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Setup file handler with rotation
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=backup_count
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    logging.info(f"Logging initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Log file: {os.path.abspath(log_path)}")
    
    return root_logger


def setup_exception_logging():
    """
    Set up global exception handler to log unhandled exceptions
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions by logging them"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default handler for KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Log the exception
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        logging.critical(f"Program terminated due to {exc_type.__name__}: {exc_value}")
        
    # Set the exception handler
    sys.excepthook = handle_exception