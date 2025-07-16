#!/usr/bin/env python3
"""
Base Handball ML Predictor
Common functionality for all handball prediction models
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseHandballPredictor:
    """Base class for handball prediction models"""
    
    def __init__(self):
        self.team_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.result_model = None
        self.goals_home_model = None
        self.goals_away_model = None
        
        # Default training files - can be overridden by subclasses
        self.training_files = [
            'hbl_LIQUI_MOLY_HBL_2020_21_20250715_063647.json',
            'hbl_LIQUI_MOLY_HBL_2021_22_20250715_063647.json',
            'hbl_LIQUI_MOLY_HBL_2022_23_20250715_063647.json',
            'hbl_LIQUI_MOLY_HBL_2023_24_20250715_063647.json'
        ]
        
        self.prediction_file = 'hbl_DAIKIN_HBL_2024_25_20250715_063647.json'
        self.team_stats = {}
        
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from all seasons - can be overridden by subclasses"""
        all_matches = []
        
        for file in self.training_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    matches = data['matches']
                    
                    for match in matches:
                        # Filter only played matches with results
                        if (match['bramki_domowe'] != '' and match['bramki_goscinne'] != '' and
                            match['bramki_domowe'] != '0' and match['bramki_goscinne'] != '0' and
                            match['status'] == 'Final'):
                            all_matches.append(match)
                    
                    logger.info(f"Loaded {len(matches)} matches from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        logger.info(f"Total loaded {len(all_matches)} training matches")
        return pd.DataFrame(all_matches)
    
    def load_prediction_data(self) -> pd.DataFrame:
        """Load data for prediction"""
        try:
            with open(self.prediction_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                matches = data['matches']
                logger.info(f"Loaded {len(matches)} matches for prediction")
                return pd.DataFrame(matches)
        except Exception as e:
            logger.error(f"Error loading prediction file: {e}")
            return pd.DataFrame()
    
    def calculate_basic_team_stats(self, df: pd.DataFrame):
        """Calculate basic team statistics - can be extended by subclasses"""
        self.team_stats = {}
        
        for _, match in df.iterrows():
            home_team = match['druzyna_domowa']
            away_team = match['druzyna_goscinna']
            home_goals = int(match['bramki_domowe'])
            away_goals = int(match['bramki_goscinne'])
            
            # Initialize statistics if they don't exist
            for team in [home_team, away_team]:
                if team not in self.team_stats:
                    self.team_stats[team] = {
                        'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                        'goals_for': 0, 'goals_against': 0,
                        'home_matches': 0, 'home_wins': 0, 'home_goals_for': 0, 'home_goals_against': 0,
                        'away_matches': 0, 'away_wins': 0, 'away_goals_for': 0, 'away_goals_against': 0
                    }
            
            # Update statistics
            self._update_team_stats(home_team, away_team, home_goals, away_goals, is_home=True)
            self._update_team_stats(away_team, home_team, away_goals, home_goals, is_home=False)
    
    def _update_team_stats(self, team: str, opponent: str, goals_for: int, goals_against: int, is_home: bool):
        """Update team statistics for a single match"""
        stats = self.team_stats[team]
        stats['matches'] += 1
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against
        
        # Determine result
        if goals_for > goals_against:
            stats['wins'] += 1
            result = 'win'
        elif goals_for < goals_against:
            stats['losses'] += 1
            result = 'loss'
        else:
            stats['draws'] += 1
            result = 'draw'
        
        # Home/away specific stats
        if is_home:
            stats['home_matches'] += 1
            stats['home_goals_for'] += goals_for
            stats['home_goals_against'] += goals_against
            if result == 'win':
                stats['home_wins'] += 1
        else:
            stats['away_matches'] += 1
            stats['away_goals_for'] += goals_for
            stats['away_goals_against'] += goals_against
            if result == 'win':
                stats['away_wins'] += 1
    
    def get_team_features(self, home_team: str, away_team: str) -> Dict:
        """Get basic features for a team matchup - can be extended by subclasses"""
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        def safe_divide(a, b):
            return a / b if b > 0 else 0
        
        # Home team features
        home_features = {
            'home_win_rate': safe_divide(home_stats.get('wins', 0), home_stats.get('matches', 1)),
            'home_goals_per_match': safe_divide(home_stats.get('goals_for', 0), home_stats.get('matches', 1)),
            'home_goals_against_per_match': safe_divide(home_stats.get('goals_against', 0), home_stats.get('matches', 1)),
            'home_home_win_rate': safe_divide(home_stats.get('home_wins', 0), home_stats.get('home_matches', 1)),
            'home_home_goals_per_match': safe_divide(home_stats.get('home_goals_for', 0), home_stats.get('home_matches', 1))
        }
        
        # Away team features
        away_features = {
            'away_win_rate': safe_divide(away_stats.get('wins', 0), away_stats.get('matches', 1)),
            'away_goals_per_match': safe_divide(away_stats.get('goals_for', 0), away_stats.get('matches', 1)),
            'away_goals_against_per_match': safe_divide(away_stats.get('goals_against', 0), away_stats.get('matches', 1)),
            'away_away_win_rate': safe_divide(away_stats.get('away_wins', 0), away_stats.get('away_matches', 1)),
            'away_away_goals_per_match': safe_divide(away_stats.get('away_goals_for', 0), away_stats.get('away_matches', 1))
        }
        
        return {**home_features, **away_features}
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic feature set - can be extended by subclasses"""
        features = []
        
        for _, match in df.iterrows():
            home_team = match['druzyna_domowa']
            away_team = match['druzyna_goscinna']
            
            # Get team features
            team_features = self.get_team_features(home_team, away_team)
            
            # Add team encoding
            team_features['home_team_encoded'] = self.team_encoder.transform([home_team])[0] if home_team in self.team_encoder.classes_ else 0
            team_features['away_team_encoded'] = self.team_encoder.transform([away_team])[0] if away_team in self.team_encoder.classes_ else 0
            
            features.append(team_features)
        
        return pd.DataFrame(features)
    
    def prepare_target_variables(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Prepare target variables for training"""
        results = []
        home_goals = []
        away_goals = []
        
        for _, match in df.iterrows():
            home_score = int(match['bramki_domowe'])
            away_score = int(match['bramki_goscinne'])
            
            home_goals.append(home_score)
            away_goals.append(away_score)
            
            # Determine result
            if home_score > away_score:
                results.append('Wygrana gospodarzy')
            elif home_score < away_score:
                results.append('Wygrana gości')
            else:
                results.append('Remis')
        
        return pd.Series(results), pd.Series(home_goals), pd.Series(away_goals)
    
    def save_predictions(self, predictions: List[Dict], filename: str = None):
        """Save predictions to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hbl_predictions_{timestamp}.json"
        
        data_to_save = {
            'metadata': {
                'model_type': self.__class__.__name__,
                'total_predictions': len(predictions),
                'created_at': datetime.now().isoformat(),
                'training_files': self.training_files,
                'prediction_file': self.prediction_file
            },
            'predictions': predictions
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Predictions saved to {filename}")
        return filename
    
    def fit_teams_encoder(self, df: pd.DataFrame):
        """Fit the team encoder with all teams"""
        all_teams = set(df['druzyna_domowa'].unique()) | set(df['druzyna_goscinna'].unique())
        self.team_encoder.fit(list(all_teams))
        logger.info(f"Team encoder fitted with {len(all_teams)} teams")
    
    def print_team_stats(self):
        """Print team statistics summary"""
        print(f"\n📊 STATYSTYKI DRUŻYN ({len(self.team_stats)} drużyn):")
        print("=" * 60)
        
        for team, stats in sorted(self.team_stats.items()):
            win_rate = stats['wins'] / stats['matches'] if stats['matches'] > 0 else 0
            avg_goals = stats['goals_for'] / stats['matches'] if stats['matches'] > 0 else 0
            
            print(f"{team:25} | M: {stats['matches']:2d} | W: {win_rate:.1%} | "
                  f"Goals: {avg_goals:.1f} | GD: {stats['goals_for']-stats['goals_against']:+3d}")
    
    # Abstract methods to be implemented by subclasses
    def train_models(self, X_train, y_result_train, y_home_goals_train, y_away_goals_train):
        """Train the prediction models - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train_models method")
    
    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Predict match outcome - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict_match method")
    
    def run_prediction(self):
        """Main prediction pipeline - can be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement run_prediction method")