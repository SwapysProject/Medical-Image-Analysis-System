"""
Configuration settings for Medical Image Analysis System
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'medical-image-analysis-2025'

    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER') or 'outputs'

    # Analysis settings
    DEFAULT_MIN_CELL_SIZE = 50
    DEFAULT_MAX_CELL_SIZE = 5000
    HIGH_RESOLUTION_THRESHOLD = 4096

    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'models'
    ENABLE_GPU = os.environ.get('ENABLE_GPU', 'False').lower() == 'true'

    # Performance settings
    MAX_WORKERS = os.environ.get('MAX_WORKERS', 4)
    TIMEOUT_SECONDS = int(os.environ.get('TIMEOUT_SECONDS', 300))  # 5 minutes

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'medical_analysis.log')

    # Security
    WTF_CSRF_ENABLED = False  # Disabled for API usage

    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Ensure upload and output directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}