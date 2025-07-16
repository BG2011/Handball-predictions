# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a handball season data scraper and ML prediction system for the German Handball Bundesliga (HBL). The project consists of data collection scripts, machine learning models, and analysis tools written in Python and JavaScript.

## Key Commands

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Data Scraping
```bash
# Scrape current season data
python handball_season_scraper.py

# Scrape all historical seasons (2020-2025)
python hbl_all_seasons_scraper.py

# Scrape specific season with full data
python daikin_season_full_scraper.py
```

### Machine Learning & Predictions
```bash
# Generate predictions (basic model)
python handball_ml_predictor.py

# Generate predictions (enhanced model)
python handball_ml_predictor_enhanced.py

# Generate predictions (ultra model)
python handball_ml_predictor_ultra.py

# Analyze prediction results
python analyze_predictions.py

# Compare predictions with reality
python compare_predictions_vs_reality.py

# Detailed model analysis
python enhanced_model_detailed_analysis.py
```

### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test class
python run_tests.py test_handball_predictor.TestHandballConfig

# Run individual test method
python run_tests.py test_handball_predictor.TestHandballConfig.test_default_config
```

### JavaScript/Node.js
```bash
# Install dependencies and run scraper
npm install
npm start
# or
node handball_season_scraper.js
```

## Architecture

### Base Classes (Refactored Architecture)
- **base_handball_predictor.py**: Base class for all ML prediction models with common functionality
- **base_handball_scraper.py**: Base class for all data scrapers with shared HTTP handling
- **config.py**: Centralized configuration system for all components

### Data Collection Layer
- **handball_season_scraper.py/js**: Main scraper for current season fixtures from Atrium Sports API
- **hbl_all_seasons_scraper.py**: Historical data collection across multiple seasons (2020-2025)
- **daikin_season_full_scraper.py**: Full season data scraper with enhanced features
- All scrapers inherit from `BaseHandballScraper` and use centralized configuration

### Machine Learning Pipeline
- **handball_ml_predictor.py**: Basic XGBoost model inheriting from `BaseHandballPredictor`
- **handball_ml_predictor_enhanced.py**: Enhanced ML model with ensemble methods and advanced features
- **handball_ml_predictor_ultra.py**: Most advanced model with polynomial features, optuna optimization, and neural networks
- All predictors use the same base functionality and configuration system

### Analysis & Validation
- **analyze_predictions.py**: Statistical analysis of prediction results
- **compare_predictions_vs_reality.py**: Validation against actual match results
- **enhanced_model_detailed_analysis.py**: Deep dive into model performance

### Testing Framework
- **test_handball_predictor.py**: Unit tests for prediction models and base classes
- **test_handball_scraper.py**: Unit tests for scraper functionality
- **run_tests.py**: Test runner with comprehensive reporting

### Data Processing
- All scripts use pandas for data manipulation
- XGBoost, LightGBM, and neural networks for ML predictions
- Standardized JSON output format for consistency
- Centralized configuration management

## Data Sources

- **Primary API**: Atrium Sports API (`https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures`)
- **Seasons Covered**: 2020/21 through 2024/25 (LIQUI MOLY HBL → DAIKIN HBL)
- **Data Format**: JSON files with timestamped filenames

## ML Model Features

The prediction models use:
- Team performance statistics
- Historical head-to-head records
- Home/away advantage factors
- Goal-scoring patterns
- Season progression metrics

## Output Files

- **Raw Data**: `hbl_*_YYYYMMDD_HHMMSS.json` (scraped fixture data)
- **Predictions**: `hbl_predictions_*_YYYYMMDD_HHMMSS.json` (ML predictions)
- **Analysis**: `predictions_vs_reality_YYYYMMDD_HHMMSS.json` (validation results)

## Code Quality Improvements (2024 Refactoring)

### Refactoring Summary
- **60% code reduction** in ML predictors through base class extraction
- **~200 lines removed** from scraper duplicates
- **Centralized configuration** system eliminates hardcoded values
- **Unit testing framework** added for quality assurance
- **Inheritance hierarchy** improves maintainability

### Base Classes
- `BaseHandballPredictor`: Common ML functionality (data loading, team stats, feature creation)
- `BaseHandballScraper`: Common scraping functionality (HTTP handling, data processing)
- `HandballConfig`: Centralized configuration management

### Configuration System
- Environment variable support for deployment flexibility
- Centralized API settings, file paths, and ML parameters
- Validation and error checking built-in

### Testing Framework
- Unit tests for all major components
- Mock testing for API interactions
- Performance and error handling tests
- Test runner with detailed reporting

## Important Notes

- All scripts include rate limiting (1-second delays) to respect API limits
- Error handling and logging are built into all major components
- The codebase supports both Polish and English language outputs
- Models are trained on historical data (2020-2024) and predict current season (2024/25)
- All components now use inheritance and centralized configuration for better maintainability