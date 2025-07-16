#!/usr/bin/env python3
"""
Configuration Management for Handball Project
Centralized configuration for all scrapers and predictors
"""

import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration settings"""
    base_url: str = "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures"
    timeout: int = 30
    delay_between_requests: int = 1
    locale: str = "en-EN"
    
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'cross-site'
            }

@dataclass
class FileConfig:
    """File path configuration"""
    # Training data files
    training_files: List[str] = None
    
    # Prediction file
    prediction_file: str = "hbl_DAIKIN_HBL_2024_25_20250715_063647.json"
    
    # Output directories
    output_dir: str = "."
    predictions_dir: str = "predictions"
    scraped_data_dir: str = "scraped_data"
    
    def __post_init__(self):
        if self.training_files is None:
            self.training_files = [
                'hbl_LIQUI_MOLY_HBL_2020_21_20250715_063647.json',
                'hbl_LIQUI_MOLY_HBL_2021_22_20250715_063647.json',
                'hbl_LIQUI_MOLY_HBL_2022_23_20250715_063647.json',
                'hbl_LIQUI_MOLY_HBL_2023_24_20250715_063647.json'
            ]
    
    def get_training_file_path(self, filename: str) -> str:
        """Get full path for training file"""
        return os.path.join(self.output_dir, filename)
    
    def get_prediction_file_path(self) -> str:
        """Get full path for prediction file"""
        return os.path.join(self.output_dir, self.prediction_file)
    
    def get_output_file_path(self, filename: str, subdir: str = None) -> str:
        """Get full path for output file"""
        if subdir:
            return os.path.join(self.output_dir, subdir, filename)
        return os.path.join(self.output_dir, filename)

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    # Random state for reproducibility
    random_state: int = 42
    
    # Test split ratio
    test_size: float = 0.2
    
    # Cross-validation folds
    cv_folds: int = 5
    
    # XGBoost parameters
    xgb_params: Dict = None
    
    # Feature selection
    max_features: int = 50
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }

@dataclass
class SeasonConfig:
    """Season-specific configuration"""
    seasons: List[Dict] = None
    
    def __post_init__(self):
        if self.seasons is None:
            self.seasons = [
                {
                    "nameLocal": "LIQUI MOLY HBL 2020/21",
                    "seasonId": "c6af4477-3956-11ef-8a59-d7d20f00d915",
                    "year": 2020
                },
                {
                    "nameLocal": "LIQUI MOLY HBL 2021/22",
                    "seasonId": "c7a62b94-3956-11ef-b16c-c5be6d4b9d5e",
                    "year": 2021
                },
                {
                    "nameLocal": "LIQUI MOLY HBL 2022/23",
                    "seasonId": "c8ae2056-3956-11ef-98d2-2b14b522d449",
                    "year": 2022
                },
                {
                    "nameLocal": "LIQUI MOLY HBL 2023/24",
                    "seasonId": "c916976b-3956-11ef-abdc-2b14b522d449",
                    "year": 2023
                },
                {
                    "nameLocal": "DAIKIN HBL 2024/25",
                    "seasonId": "cabcf509-4373-11ef-a370-9d3c1e90234a",
                    "year": 2024
                }
            ]
    
    def get_season_by_year(self, year: int) -> Dict:
        """Get season configuration by year"""
        for season in self.seasons:
            if season["year"] == year:
                return season
        return None
    
    def get_current_season(self) -> Dict:
        """Get current season (2024/25)"""
        return self.get_season_by_year(2024)

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    file_logging: bool = False
    log_file: str = "handball.log"

class HandballConfig:
    """Main configuration class"""
    
    def __init__(self):
        self.api = APIConfig()
        self.files = FileConfig()
        self.ml = MLConfig()
        self.seasons = SeasonConfig()
        self.logging = LoggingConfig()
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        # API configuration
        if os.getenv('HANDBALL_API_URL'):
            self.api.base_url = os.getenv('HANDBALL_API_URL')
        
        if os.getenv('HANDBALL_API_TIMEOUT'):
            self.api.timeout = int(os.getenv('HANDBALL_API_TIMEOUT'))
        
        if os.getenv('HANDBALL_API_DELAY'):
            self.api.delay_between_requests = int(os.getenv('HANDBALL_API_DELAY'))
        
        # File configuration
        if os.getenv('HANDBALL_OUTPUT_DIR'):
            self.files.output_dir = os.getenv('HANDBALL_OUTPUT_DIR')
        
        if os.getenv('HANDBALL_PREDICTION_FILE'):
            self.files.prediction_file = os.getenv('HANDBALL_PREDICTION_FILE')
        
        # ML configuration
        if os.getenv('HANDBALL_ML_RANDOM_STATE'):
            self.ml.random_state = int(os.getenv('HANDBALL_ML_RANDOM_STATE'))
        
        if os.getenv('HANDBALL_ML_TEST_SIZE'):
            self.ml.test_size = float(os.getenv('HANDBALL_ML_TEST_SIZE'))
        
        # Logging configuration
        if os.getenv('HANDBALL_LOG_LEVEL'):
            self.logging.level = os.getenv('HANDBALL_LOG_LEVEL')
        
        if os.getenv('HANDBALL_LOG_FILE'):
            self.logging.file_logging = True
            self.logging.log_file = os.getenv('HANDBALL_LOG_FILE')
    
    def validate(self) -> bool:
        """Validate configuration"""
        # Check if training files exist
        missing_files = []
        for file in self.files.training_files:
            if not os.path.exists(self.files.get_training_file_path(file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"Warning: Missing training files: {missing_files}")
        
        # Check if prediction file exists
        if not os.path.exists(self.files.get_prediction_file_path()):
            print(f"Warning: Missing prediction file: {self.files.prediction_file}")
        
        # Validate ML parameters
        if not (0 < self.ml.test_size < 1):
            print(f"Error: test_size must be between 0 and 1, got {self.ml.test_size}")
            return False
        
        if self.ml.cv_folds < 2:
            print(f"Error: cv_folds must be at least 2, got {self.ml.cv_folds}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 HANDBALL PROJECT CONFIGURATION")
        print("=" * 40)
        print(f"API URL: {self.api.base_url}")
        print(f"API Timeout: {self.api.timeout}s")
        print(f"Request Delay: {self.api.delay_between_requests}s")
        print(f"Output Directory: {self.files.output_dir}")
        print(f"Training Files: {len(self.files.training_files)}")
        print(f"Prediction File: {self.files.prediction_file}")
        print(f"ML Random State: {self.ml.random_state}")
        print(f"ML Test Size: {self.ml.test_size}")
        print(f"CV Folds: {self.ml.cv_folds}")
        print(f"Logging Level: {self.logging.level}")
        print(f"Seasons: {len(self.seasons.seasons)}")

# Global configuration instance
config = HandballConfig()

# Update from environment variables
config.update_from_env()

# Validate configuration
if not config.validate():
    print("Configuration validation failed!")