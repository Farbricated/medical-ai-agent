"""
Production-grade logging system for MedAI
Provides structured logging with rotation and multiple handlers
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json

class MedAILogger:
    """
    Centralized logging system with file rotation and structured output
    """
    
    def __init__(self, name: str = "medai", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add handlers
        self._add_console_handler()
        self._add_file_handler()
        self._add_error_handler()
        
    def _add_console_handler(self):
        """Add colored console output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add rotating file handler for general logs"""
        file_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def _add_error_handler(self):
        """Add separate handler for errors"""
        error_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        error_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n%(exc_info)s'
        )
        error_handler.setFormatter(error_format)
        self.logger.addHandler(error_handler)
    
    def log_query(self, query: str, query_type: str, response_time: float, success: bool):
        """Log query metrics in structured format"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "query",
            "query_type": query_type,
            "response_time": response_time,
            "success": success,
            "query_preview": query[:100]
        }
        
        self.logger.info(f"QUERY: {json.dumps(log_data)}")
    
    def log_agent_action(self, agent: str, action: str, details: dict = None):
        """Log agent actions"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_action",
            "agent": agent,
            "action": action,
            "details": details or {}
        }
        
        self.logger.info(f"AGENT: {json.dumps(log_data)}")
    
    def log_error(self, error: Exception, context: dict = None):
        """Log errors with context"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(log_data)}", exc_info=True)
    
    def log_performance(self, metric_name: str, value: float, unit: str = "seconds"):
        """Log performance metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance",
            "metric": metric_name,
            "value": value,
            "unit": unit
        }
        
        self.logger.info(f"PERF: {json.dumps(log_data)}")
    
    def get_logger(self):
        """Get the underlying logger"""
        return self.logger

# Global logger instance
_logger_instance = None

def get_logger(name: str = "medai") -> MedAILogger:
    """Get or create logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MedAILogger(name)
    return _logger_instance
