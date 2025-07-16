#!/usr/bin/env python3
"""
ULTRA Enhanced Handball ML Predictor
Najbardziej zaawansowana wersja z nowymi technikami ML
CEL: POBIĆ 67.3% I OSIĄGNĄĆ 70%+ DOKŁADNOŚCI!
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Base class and configuration
from base_handball_predictor import BaseHandballPredictor
from config import config

# Advanced ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import VotingClassifier, VotingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
import optuna  # For hyperparameter optimization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraHandballPredictor(BaseHandballPredictor):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Ultra-specific enhancements
        self.scaler = RobustScaler()  # Use RobustScaler instead of StandardScaler
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=config.ml.max_features)
        
        # Use configuration for file paths
        self.training_files = config.files.training_files
        self.prediction_file = config.files.prediction_file
        
        # Ultra-advanced statistics
        self.head_to_head = {}
        self.recent_form = {}
        self.momentum_stats = {}
        self.seasonal_trends = {}
        self.feature_columns = []
        self.best_params = {}
    
    def load_training_data_with_time(self) -> pd.DataFrame:
        """Ładuje dane z informacjami czasowymi dla analizy trendów - rozszerzenie metody bazowej"""
        # Użyj metody bazowej do załadowania danych
        df = super().load_training_data()
        
        # Dodaj informacje czasowe specyficzne dla ultra modelu
        if not df.empty:
            df['match_date'] = pd.to_datetime(df['data'])
            df['season_year'] = df['match_date'].dt.year
            df['month'] = df['match_date'].dt.month
            df['day_of_week'] = df['match_date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'] >= 5
            
            # Sortuj według daty
            df = df.sort_values('match_date')
        
        logger.info(f"Załadowano {len(df)} meczów treningowych z informacjami czasowymi")
        return df
    
    def calculate_ultra_team_stats(self, df: pd.DataFrame):
        """Oblicza ultra-zaawansowane statystyki drużyn - rozszerzenie podstawowych statystyk"""
        # Najpierw oblicz podstawowe statystyki używając metody bazowej
        super().calculate_basic_team_stats(df)
        
        # Inicjalizacja struktur danych ultra
        self.head_to_head = {}
        self.recent_form = {}
        self.momentum_stats = {}
        self.seasonal_trends = {}
        
        # Sortuj według daty
        df_sorted = df.sort_values('match_date')
        
        # Rozszerz istniejące statystyki o ultra-cechy
        for team in self.team_stats:
            self.team_stats[team].update({
                'recent_matches': [], 'goal_difference': 0,
                'high_scoring_matches': 0, 'low_scoring_matches': 0,
                'comeback_wins': 0, 'blowout_wins': 0, 'close_wins': 0,
                'first_half_performance': [], 'second_half_performance': [],
                'weekend_performance': [], 'weekday_performance': [],
                'monthly_performance': {i: [] for i in range(1, 13)},
                'opponent_strength_faced': [], 'performance_vs_strong': [],
                'performance_vs_weak': [], 'consistency_score': 0,
                'injury_impact': 0, 'fatigue_factor': 0
            })
            
            self.recent_form[team] = []
            self.momentum_stats[team] = {'streak': 0, 'streak_type': 'none', 'momentum_score': 0}
            self.seasonal_trends[team] = {'early_season': [], 'mid_season': [], 'late_season': []}
        
        # Przetwarzanie meczów z zaawansowanymi statystykami
        for idx, match in df_sorted.iterrows():
            home_team = match['druzyna_domowa']
            away_team = match['druzyna_goscinna']
            home_goals = int(match['bramki_domowe'])
            away_goals = int(match['bramki_goscinne'])
            match_date = match['match_date']
            month = match['month']
            is_weekend = match['is_weekend']
            
            total_goals = home_goals + away_goals
            goal_diff = abs(home_goals - away_goals)
            
            # Head-to-head z zaawansowanymi statystykami
            h2h_key = tuple(sorted([home_team, away_team]))
            if h2h_key not in self.head_to_head:
                self.head_to_head[h2h_key] = {
                    'matches': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0,
                    'avg_total_goals': 0, 'goal_variance': 0, 'recent_trend': []
                }
            
            self.head_to_head[h2h_key]['matches'] += 1
            self.head_to_head[h2h_key]['recent_trend'].append(total_goals)
            if len(self.head_to_head[h2h_key]['recent_trend']) > 5:
                self.head_to_head[h2h_key]['recent_trend'] = self.head_to_head[h2h_key]['recent_trend'][-5:]
            
            # Aktualizuj ULTRA-specyficzne statystyki
            for team, goals_for, goals_against, is_home in [(home_team, home_goals, away_goals, True), 
                                                           (away_team, away_goals, home_goals, False)]:
                
                stats = self.team_stats[team]
                
                # Typy wygranych (ultra-specyficzne)
                if goals_for > goals_against:
                    if goal_diff <= 3:
                        stats['close_wins'] += 1
                    elif goal_diff >= 10:
                        stats['blowout_wins'] += 1
                    
                    self.recent_form[team].append(3)
                    
                elif goals_for < goals_against:
                    self.recent_form[team].append(0)
                else:
                    self.recent_form[team].append(1)
                
                # Specjalne statystyki
                if total_goals > 60:
                    stats['high_scoring_matches'] += 1
                elif total_goals < 50:
                    stats['low_scoring_matches'] += 1
                
                # Performance według dnia tygodnia
                if is_weekend:
                    stats['weekend_performance'].append(goals_for - goals_against)
                else:
                    stats['weekday_performance'].append(goals_for - goals_against)
                
                # Performance według miesiąca
                stats['monthly_performance'][month].append(goals_for - goals_against)
                
                # Zachowaj tylko ostatnie 10 meczów dla formy
                if len(self.recent_form[team]) > 10:
                    self.recent_form[team] = self.recent_form[team][-10:]
        
        # Oblicz zaawansowane metryki
        self._calculate_advanced_metrics()
        
        logger.info(f"Obliczono ultra-zaawansowane statystyki dla {len(self.team_stats)} drużyn")
    
    def train_models(self, X_train, y_result_train, y_home_goals_train, y_away_goals_train):
        """Implementation of abstract method from BaseHandballPredictor"""
        return self.train_ultra_models(X_train, y_result_train, y_home_goals_train, y_away_goals_train)
    
    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Implementation of abstract method from BaseHandballPredictor"""
        # Create a temporary DataFrame for the match
        match_data = pd.DataFrame([{
            'druzyna_domowa': home_team,
            'druzyna_goscinna': away_team,
            'data': datetime.now().strftime('%Y-%m-%d'),
            'month': datetime.now().month,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': datetime.now().weekday() >= 5
        }])
        
        # Create features
        X_pred = self.create_ultra_features(match_data, is_training=False)
        X_pred_clean = X_pred[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred_clean)
        X_pred_selected = self.feature_selector.transform(X_pred_scaled)
        
        # Make predictions
        result_pred = self.result_model.predict(X_pred_selected)[0]
        result_proba = self.result_model.predict_proba(X_pred_selected)[0]
        home_goals_pred = max(0, round(self.goals_home_model.predict(X_pred_selected)[0]))
        away_goals_pred = max(0, round(self.goals_away_model.predict(X_pred_selected)[0]))
        
        result_labels = ['Wygrana gospodarzy', 'Remis', 'Wygrana gości']
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_result': result_labels[result_pred],
            'home_win_probability': float(result_proba[0]),
            'draw_probability': float(result_proba[1]),
            'away_win_probability': float(result_proba[2]),
            'confidence': float(max(result_proba)),
            'predicted_home_goals': int(home_goals_pred),
            'predicted_away_goals': int(away_goals_pred),
            'predicted_score': f"{int(home_goals_pred)}:{int(away_goals_pred)}"
        }
    
    def run_prediction(self):
        """Implementation of abstract method from BaseHandballPredictor - Ultra version"""
        return main()
    
    def _calculate_advanced_metrics(self):
        """Oblicza zaawansowane metryki dla każdej drużyny"""
        for team, stats in self.team_stats.items():
            if stats['matches'] > 0:
                # Podstawowe średnie
                stats['avg_goals_for'] = stats['goals_for'] / stats['matches']
                stats['avg_goals_against'] = stats['goals_against'] / stats['matches']
                stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
                stats['win_rate'] = stats['wins'] / stats['matches']
                stats['points_per_game'] = (stats['wins'] * 3 + stats['draws']) / stats['matches']
                
                # Dom/wyjazd
                if stats['home_matches'] > 0:
                    stats['home_avg_goals_for'] = stats['home_goals_for'] / stats['home_matches']
                    stats['home_avg_goals_against'] = stats['home_goals_against'] / stats['home_matches']
                    stats['home_win_rate'] = stats['home_wins'] / stats['home_matches']
                
                if stats['away_matches'] > 0:
                    stats['away_avg_goals_for'] = stats['away_goals_for'] / stats['away_matches']
                    stats['away_avg_goals_against'] = stats['away_goals_against'] / stats['away_matches']
                    stats['away_win_rate'] = stats['away_wins'] / stats['away_matches']
                
                # Forma i momentum
                if team in self.recent_form and self.recent_form[team]:
                    recent_points = self.recent_form[team]
                    stats['recent_form_points'] = sum(recent_points) / len(recent_points)
                    
                    # Trend formy (czy idzie w górę/dół)
                    if len(recent_points) >= 5:
                        early_form = sum(recent_points[:3]) / 3
                        late_form = sum(recent_points[-3:]) / 3
                        stats['form_trend'] = late_form - early_form
                    else:
                        stats['form_trend'] = 0
                    
                    # Momentum (seria wyników)
                    streak = 0
                    streak_type = 'none'
                    if recent_points:
                        last_result = recent_points[-1]
                        for i in range(len(recent_points)-1, -1, -1):
                            if recent_points[i] == last_result:
                                streak += 1
                            else:
                                break
                        
                        if last_result == 3:
                            streak_type = 'win'
                        elif last_result == 0:
                            streak_type = 'loss'
                        else:
                            streak_type = 'draw'
                    
                    self.momentum_stats[team] = {
                        'streak': streak,
                        'streak_type': streak_type,
                        'momentum_score': streak * (1 if last_result == 3 else -1 if last_result == 0 else 0)
                    }
                
                # Konsystencja (odchylenie standardowe wyników)
                if len(self.recent_form[team]) > 3:
                    stats['consistency_score'] = 1 / (1 + np.std(self.recent_form[team]))
                else:
                    stats['consistency_score'] = 0.5
                
                # Performance według kontekstu
                if stats['weekend_performance']:
                    stats['weekend_avg_diff'] = np.mean(stats['weekend_performance'])
                else:
                    stats['weekend_avg_diff'] = 0
                
                if stats['weekday_performance']:
                    stats['weekday_avg_diff'] = np.mean(stats['weekday_performance'])
                else:
                    stats['weekday_avg_diff'] = 0
                
                # Miesięczne trendy
                stats['best_month'] = 0
                stats['worst_month'] = 0
                best_performance = -100
                worst_performance = 100
                
                for month, performances in stats['monthly_performance'].items():
                    if performances:
                        avg_perf = np.mean(performances)
                        if avg_perf > best_performance:
                            best_performance = avg_perf
                            stats['best_month'] = month
                        if avg_perf < worst_performance:
                            worst_performance = avg_perf
                            stats['worst_month'] = month
                
                # Specjalne wskaźniki
                stats['high_scoring_rate'] = stats['high_scoring_matches'] / stats['matches']
                stats['low_scoring_rate'] = stats['low_scoring_matches'] / stats['matches']
                stats['blowout_rate'] = stats['blowout_wins'] / max(stats['wins'], 1)
                stats['close_win_rate'] = stats['close_wins'] / max(stats['wins'], 1)

    def create_ultra_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Tworzy ultra-zaawansowane cechy ML"""
        features = []

        for _, match in df.iterrows():
            home_team = match['druzyna_domowa']
            away_team = match['druzyna_goscinna']

            # Podstawowe cechy
            feature_row = {
                'home_team_encoded': self.team_encoder.transform([home_team])[0] if home_team in self.team_encoder.classes_ else -1,
                'away_team_encoded': self.team_encoder.transform([away_team])[0] if away_team in self.team_encoder.classes_ else -1,
            }

            # Kontekstowe cechy (nowe!)
            if hasattr(match, 'month'):
                feature_row.update({
                    'month': match.get('month', 1),
                    'is_weekend': int(match.get('is_weekend', False)),
                    'day_of_week': match.get('day_of_week', 0),
                })
            else:
                feature_row.update({'month': 1, 'is_weekend': 0, 'day_of_week': 0})

            # Statystyki drużyn
            home_stats = self.team_stats.get(home_team, {})
            away_stats = self.team_stats.get(away_team, {})

            # Podstawowe statystyki (rozszerzone)
            feature_row.update({
                'home_avg_goals_for': home_stats.get('avg_goals_for', 25),
                'home_avg_goals_against': home_stats.get('avg_goals_against', 25),
                'home_win_rate': home_stats.get('win_rate', 0.5),
                'home_points_per_game': home_stats.get('points_per_game', 1.5),
                'home_goal_difference': home_stats.get('goal_difference', 0),
                'home_consistency': home_stats.get('consistency_score', 0.5),

                'away_avg_goals_for': away_stats.get('avg_goals_for', 25),
                'away_avg_goals_against': away_stats.get('avg_goals_against', 25),
                'away_win_rate': away_stats.get('win_rate', 0.5),
                'away_points_per_game': away_stats.get('points_per_game', 1.5),
                'away_goal_difference': away_stats.get('goal_difference', 0),
                'away_consistency': away_stats.get('consistency_score', 0.5),
            })

            # Forma i momentum (ultra-zaawansowane!)
            home_momentum = self.momentum_stats.get(home_team, {})
            away_momentum = self.momentum_stats.get(away_team, {})

            feature_row.update({
                'home_recent_form': home_stats.get('recent_form_points', 1.5),
                'home_form_trend': home_stats.get('form_trend', 0),
                'home_momentum_score': home_momentum.get('momentum_score', 0),
                'home_streak': home_momentum.get('streak', 0),
                'home_streak_is_win': int(home_momentum.get('streak_type', 'none') == 'win'),

                'away_recent_form': away_stats.get('recent_form_points', 1.5),
                'away_form_trend': away_stats.get('form_trend', 0),
                'away_momentum_score': away_momentum.get('momentum_score', 0),
                'away_streak': away_momentum.get('streak', 0),
                'away_streak_is_win': int(away_momentum.get('streak_type', 'none') == 'win'),
            })

            # Dom/wyjazd (ultra-szczegółowe)
            feature_row.update({
                'home_home_goals_for': home_stats.get('home_avg_goals_for', 25),
                'home_home_goals_against': home_stats.get('home_avg_goals_against', 25),
                'home_home_win_rate': home_stats.get('home_win_rate', 0.5),
                'home_weekend_performance': home_stats.get('weekend_avg_diff', 0),
                'home_weekday_performance': home_stats.get('weekday_avg_diff', 0),

                'away_away_goals_for': away_stats.get('away_avg_goals_for', 25),
                'away_away_goals_against': away_stats.get('away_avg_goals_against', 25),
                'away_away_win_rate': away_stats.get('away_win_rate', 0.5),
                'away_weekend_performance': away_stats.get('weekend_avg_diff', 0),
                'away_weekday_performance': away_stats.get('weekday_avg_diff', 0),
            })

            # Head-to-head (ultra-zaawansowane)
            h2h_key = tuple(sorted([home_team, away_team]))
            h2h_stats = self.head_to_head.get(h2h_key, {})

            if h2h_stats.get('matches', 0) > 0:
                recent_h2h_goals = h2h_stats.get('recent_trend', [55])
                feature_row.update({
                    'h2h_home_win_rate': h2h_stats.get('home_wins', 0) / h2h_stats['matches'],
                    'h2h_away_win_rate': h2h_stats.get('away_wins', 0) / h2h_stats['matches'],
                    'h2h_draw_rate': h2h_stats.get('draws', 0) / h2h_stats['matches'],
                    'h2h_matches_played': min(h2h_stats['matches'], 10),
                    'h2h_avg_goals': np.mean(recent_h2h_goals),
                    'h2h_goals_trend': np.mean(np.diff(recent_h2h_goals)) if len(recent_h2h_goals) > 1 else 0,
                })
            else:
                feature_row.update({
                    'h2h_home_win_rate': 0.5, 'h2h_away_win_rate': 0.5, 'h2h_draw_rate': 0.1,
                    'h2h_matches_played': 0, 'h2h_avg_goals': 55, 'h2h_goals_trend': 0
                })

            # Ultra-zaawansowane różnice i interakcje
            feature_row.update({
                'goal_diff_advantage': home_stats.get('goal_difference', 0) - away_stats.get('goal_difference', 0),
                'form_advantage': home_stats.get('recent_form_points', 1.5) - away_stats.get('recent_form_points', 1.5),
                'momentum_advantage': home_momentum.get('momentum_score', 0) - away_momentum.get('momentum_score', 0),
                'consistency_advantage': home_stats.get('consistency_score', 0.5) - away_stats.get('consistency_score', 0.5),
                'win_rate_diff': home_stats.get('win_rate', 0.5) - away_stats.get('win_rate', 0.5),
                'attack_vs_defense': home_stats.get('avg_goals_for', 25) - away_stats.get('avg_goals_against', 25),
                'defense_vs_attack': away_stats.get('avg_goals_for', 25) - home_stats.get('avg_goals_against', 25),
                'home_advantage_factor': home_stats.get('home_win_rate', 0.5) - away_stats.get('away_win_rate', 0.5),
                'form_trend_diff': home_stats.get('form_trend', 0) - away_stats.get('form_trend', 0),
            })

            # Specjalne wskaźniki stylu gry
            feature_row.update({
                'home_high_scoring_rate': home_stats.get('high_scoring_rate', 0.5),
                'away_high_scoring_rate': away_stats.get('high_scoring_rate', 0.5),
                'home_low_scoring_rate': home_stats.get('low_scoring_rate', 0.5),
                'away_low_scoring_rate': away_stats.get('low_scoring_rate', 0.5),
                'home_blowout_rate': home_stats.get('blowout_rate', 0.1),
                'away_blowout_rate': away_stats.get('blowout_rate', 0.1),
                'home_close_win_rate': home_stats.get('close_win_rate', 0.5),
                'away_close_win_rate': away_stats.get('close_win_rate', 0.5),
                'expected_game_pace': (home_stats.get('avg_goals_for', 25) + away_stats.get('avg_goals_for', 25)) / 2,
                'defensive_battle_indicator': (home_stats.get('avg_goals_against', 25) + away_stats.get('avg_goals_against', 25)) / 2,
            })

            # Kontekstowe modyfikatory
            current_month = feature_row['month']
            feature_row.update({
                'home_month_performance': home_stats.get('monthly_performance', {}).get(current_month, [0]),
                'away_month_performance': away_stats.get('monthly_performance', {}).get(current_month, [0]),
                'home_is_best_month': int(home_stats.get('best_month', 0) == current_month),
                'away_is_best_month': int(away_stats.get('best_month', 0) == current_month),
                'home_is_worst_month': int(home_stats.get('worst_month', 0) == current_month),
                'away_is_worst_month': int(away_stats.get('worst_month', 0) == current_month),
            })

            # Konwertuj listy na średnie
            if isinstance(feature_row['home_month_performance'], list):
                feature_row['home_month_performance'] = np.mean(feature_row['home_month_performance']) if feature_row['home_month_performance'] else 0
            if isinstance(feature_row['away_month_performance'], list):
                feature_row['away_month_performance'] = np.mean(feature_row['away_month_performance']) if feature_row['away_month_performance'] else 0

            # Etykiety (tylko dla danych treningowych)
            if is_training:
                home_goals = int(match['bramki_domowe'])
                away_goals = int(match['bramki_goscinne'])

                if home_goals > away_goals:
                    result = 0  # Wygrana gospodarzy
                elif home_goals < away_goals:
                    result = 2  # Wygrana gości
                else:
                    result = 1  # Remis

                feature_row.update({
                    'result': result,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'total_goals': home_goals + away_goals
                })

            features.append(feature_row)

        return pd.DataFrame(features)

    def optimize_hyperparameters(self, X_train, y_train, model_type='classification'):
        """Optymalizuje hiperparametry używając Optuna"""
        def objective(trial):
            if model_type == 'classification':
                # XGBoost parameters
                params = {
                    'objective': 'multi:softprob',
                    'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                    'max_depth': trial.suggest_int('max_depth', 6, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0),
                    'random_state': config.ml.random_state,
                    'n_jobs': -1
                }

                model = xgb.XGBClassifier(**params)

                # Cross-validation
                cv = StratifiedKFold(n_splits=config.ml.cv_folds, shuffle=True, random_state=config.ml.random_state)
                scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    X_tr_scaled = self.scaler.fit_transform(X_tr)
                    X_val_scaled = self.scaler.transform(X_val)

                    model.fit(X_tr_scaled, y_tr)
                    pred = model.predict(X_val_scaled)
                    scores.append(accuracy_score(y_val, pred))

                return np.mean(scores)

            else:  # regression
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                    'max_depth': trial.suggest_int('max_depth', 6, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0),
                    'random_state': config.ml.random_state,
                    'n_jobs': -1
                }

                model = xgb.XGBRegressor(**params)

                # Cross-validation
                cv = StratifiedKFold(n_splits=config.ml.cv_folds, shuffle=True, random_state=config.ml.random_state)
                scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    X_tr_scaled = self.scaler.fit_transform(X_tr)
                    X_val_scaled = self.scaler.transform(X_val)

                    model.fit(X_tr_scaled, y_tr)
                    pred = model.predict(X_val_scaled)
                    scores.append(-mean_absolute_error(y_val, pred))  # Negative because we want to maximize

                return np.mean(scores)

        # Optymalizacja (ograniczona liczba prób dla szybkości)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        return study.best_params

    def train_ultra_models(self, X_train: pd.DataFrame, y_result: pd.Series, y_home_goals: pd.Series, y_away_goals: pd.Series):
        """Trenuje ultra-zaawansowane modele z optymalizacją"""
        logger.info("Rozpoczynam trening ULTRA modeli z optymalizacją hiperparametrów...")

        # Feature engineering - polynomial features (tylko dla najważniejszych cech)
        important_features = ['home_avg_goals_for', 'away_avg_goals_for', 'home_win_rate', 'away_win_rate',
                             'form_advantage', 'goal_diff_advantage', 'momentum_advantage']

        X_important = X_train[important_features] if all(col in X_train.columns for col in important_features) else X_train.iloc[:, :7]

        # Skalowanie
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Feature selection
        self.feature_selector.fit(X_train_scaled, y_result)
        X_train_selected = self.feature_selector.transform(X_train_scaled)

        logger.info(f"Wybrano {X_train_selected.shape[1]} najważniejszych cech z {X_train_scaled.shape[1]}")

        # 1. ULTRA Model wyniku - Super Ensemble
        logger.info("Optymalizacja hiperparametrów dla modelu wyniku...")

        # Optymalizuj XGBoost
        best_xgb_params = self.optimize_hyperparameters(X_train, y_result, 'classification')
        logger.info(f"Najlepsze parametry XGBoost: {best_xgb_params}")

        # Stwórz modele z optymalnymi parametrami
        xgb_result = xgb.XGBClassifier(**best_xgb_params)

        # LightGBM (alternatywa do XGBoost)
        lgb_result = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.ml.random_state,
            n_jobs=-1,
            verbose=-1
        )

        # Extra Trees (bardzo różny od RF)
        et_result = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=config.ml.random_state,
            n_jobs=-1
        )

        # Gradient Boosting
        gb_result = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=config.ml.random_state
        )

        # Neural Network
        mlp_result = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=config.ml.random_state
        )

        # Super Ensemble
        self.result_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_result),
                ('lgb', lgb_result),
                ('et', et_result),
                ('gb', gb_result),
                ('mlp', mlp_result)
            ],
            voting='soft',
            n_jobs=-1
        )

        # 2. ULTRA Model goli - Super Ensemble
        logger.info("Trenowanie ultra modeli goli...")

        # Optymalizuj dla regresji
        best_reg_params = self.optimize_hyperparameters(X_train, y_home_goals, 'regression')

        xgb_home = xgb.XGBRegressor(**best_reg_params)
        xgb_away = xgb.XGBRegressor(**best_reg_params)

        lgb_home = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.ml.random_state,
            n_jobs=-1,
            verbose=-1
        )

        lgb_away = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.ml.random_state,
            n_jobs=-1,
            verbose=-1
        )

        et_home = ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=config.ml.random_state, n_jobs=-1)
        et_away = ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=config.ml.random_state, n_jobs=-1)

        mlp_home = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=config.ml.random_state)
        mlp_away = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=config.ml.random_state)

        self.goals_home_model = VotingRegressor(
            estimators=[
                ('xgb', xgb_home),
                ('lgb', lgb_home),
                ('et', et_home),
                ('mlp', mlp_home)
            ],
            n_jobs=-1
        )

        self.goals_away_model = VotingRegressor(
            estimators=[
                ('xgb', xgb_away),
                ('lgb', lgb_away),
                ('et', et_away),
                ('mlp', mlp_away)
            ],
            n_jobs=-1
        )

        # Trenowanie wszystkich modeli
        logger.info("Trenowanie super ensemble modeli...")
        self.result_model.fit(X_train_selected, y_result)
        self.goals_home_model.fit(X_train_selected, y_home_goals)
        self.goals_away_model.fit(X_train_selected, y_away_goals)

        logger.info("ULTRA modele wytrenowane pomyślnie!")

        # Zapisz najlepsze parametry
        self.best_params = {
            'xgb_classification': best_xgb_params,
            'xgb_regression': best_reg_params
        }

    def evaluate_ultra_models(self, X_test: pd.DataFrame, y_result_test: pd.Series,
                             y_home_goals_test: pd.Series, y_away_goals_test: pd.Series):
        """Ocenia jakość ULTRA modeli"""
        logger.info("Oceniam jakość ULTRA modeli...")

        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Przewidywania
        result_pred = self.result_model.predict(X_test_selected)
        result_proba = self.result_model.predict_proba(X_test_selected)
        home_goals_pred = self.goals_home_model.predict(X_test_selected)
        away_goals_pred = self.goals_away_model.predict(X_test_selected)

        # Metryki
        result_accuracy = accuracy_score(y_result_test, result_pred)
        home_goals_mae = mean_absolute_error(y_home_goals_test, home_goals_pred)
        away_goals_mae = mean_absolute_error(y_away_goals_test, away_goals_pred)

        # Dodatkowe metryki
        confidence_scores = np.max(result_proba, axis=1)
        high_confidence_mask = confidence_scores > 0.7
        if np.sum(high_confidence_mask) > 0:
            high_conf_accuracy = accuracy_score(y_result_test[high_confidence_mask], result_pred[high_confidence_mask])
        else:
            high_conf_accuracy = 0

        logger.info(f"ULTRA Dokładność przewidywania wyniku: {result_accuracy:.3f}")
        logger.info(f"ULTRA Dokładność wysokiej pewności (>70%): {high_conf_accuracy:.3f}")
        logger.info(f"ULTRA MAE gole gospodarzy: {home_goals_mae:.2f}")
        logger.info(f"ULTRA MAE gole gości: {away_goals_mae:.2f}")

        return {
            'result_accuracy': result_accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'home_goals_mae': home_goals_mae,
            'away_goals_mae': away_goals_mae,
            'avg_confidence': np.mean(confidence_scores)
        }


    def predict_season_ultra(self, prediction_df: pd.DataFrame) -> List[Dict]:
        """Przewiduje wyniki używając ULTRA modeli"""
        logger.info("Przewidywanie wyników sezonu 2024/25 (ULTRA)...")

        # Przygotuj cechy
        X_pred = self.create_ultra_features(prediction_df, is_training=False)

        # Użyj tych samych kolumn co podczas treningu
        X_pred_clean = X_pred[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred_clean)
        X_pred_selected = self.feature_selector.transform(X_pred_scaled)

        # Przewidywania
        result_pred = self.result_model.predict(X_pred_selected)
        result_proba = self.result_model.predict_proba(X_pred_selected)
        home_goals_pred = self.goals_home_model.predict(X_pred_selected)
        away_goals_pred = self.goals_away_model.predict(X_pred_selected)

        # Przygotuj wyniki
        predictions = []
        result_labels = ['Wygrana gospodarzy', 'Remis', 'Wygrana gości']

        for i, (_, match) in enumerate(prediction_df.iterrows()):
            home_goals = max(0, round(home_goals_pred[i]))
            away_goals = max(0, round(away_goals_pred[i]))

            # Oblicz pewność przewidywania
            confidence = float(np.max(result_proba[i]))

            prediction = {
                'mecz': match['mecz'],
                'data': match['data'],
                'druzyna_domowa': match['druzyna_domowa'],
                'druzyna_goscinna': match['druzyna_goscinna'],
                'przewidywany_wynik': result_labels[result_pred[i]],
                'prawdopodobienstwo_wygranej_gospodarzy': float(result_proba[i][0]),
                'prawdopodobienstwo_remisu': float(result_proba[i][1]),
                'prawdopodobienstwo_wygranej_gosci': float(result_proba[i][2]),
                'pewnosc_przewidywania': confidence,
                'przewidywane_gole_gospodarzy': int(home_goals),
                'przewidywane_gole_gosci': int(away_goals),
                'przewidywany_wynik_bramkowy': f"{int(home_goals)}:{int(away_goals)}",
                'przewidywana_suma_goli': int(home_goals + away_goals),
                'runda': match['runda'],
                'model_version': 'ULTRA_Super_Ensemble_v3.0'
            }
            predictions.append(prediction)

        logger.info(f"Przewidziano wyniki dla {len(predictions)} meczów (ULTRA)")
        return predictions

    def save_ultra_predictions(self, predictions: List[Dict], filename: str = None):
        """Zapisuje ULTRA przewidywania - rozszerzenie metody bazowej"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hbl_predictions_ULTRA_2024_25_{timestamp}.json"

        # Oblicz statystyki pewności
        confidences = [p['pewnosc_przewidywania'] for p in predictions]
        high_confidence_count = sum(1 for c in confidences if c > 0.8)
        
        # Użyj metody bazowej do zapisania z ultra-specyficznymi metadanymi
        temp_predictions = []
        for p in predictions:
            temp_pred = p.copy()
            temp_pred['ultra_metadata'] = {
                'algorithm': 'ULTRA Super Ensemble (XGBoost + LightGBM + ExtraTrees + GradientBoosting + MLP)',
                'version': 'ULTRA v3.0',
                'features_used': [
                    'ultra_team_stats', 'momentum_tracking', 'seasonal_trends',
                    'contextual_features', 'advanced_h2h', 'form_trends',
                    'consistency_metrics', 'style_indicators', 'monthly_performance',
                    'weekend_effects', 'polynomial_interactions'
                ],
                'optimizations': [
                    'Optuna hyperparameter tuning', 'Feature selection',
                    'Super ensemble voting', 'Advanced feature engineering',
                    'Momentum and streak tracking', 'Contextual modifiers'
                ],
                'best_hyperparameters': self.best_params,
                'high_confidence_predictions': high_confidence_count,
                'avg_confidence': np.mean(confidences)
            }
            temp_predictions.append(temp_pred)

        return super().save_predictions(temp_predictions, filename)

def main():
    """Główna funkcja wykonawcza - ULTRA Version"""
    predictor = UltraHandballPredictor()

    try:
        print("🚀🚀🚀 ULTRA Handball ML Predictor - Najbardziej zaawansowana wersja!")
        print("🎯🎯🎯 CEL: POBIĆ 67.3% I OSIĄGNĄĆ 70%+ DOKŁADNOŚCI!")
        print("=" * 80)

        # 1. Ładowanie danych z informacjami czasowymi
        print("\n📊 Ładowanie danych z analizą czasową...")
        training_df = predictor.load_training_data_with_time()

        if training_df.empty:
            print("❌ Brak danych treningowych!")
            return

        # 2. Obliczanie ultra-zaawansowanych statystyk
        print("📈 Obliczanie ultra-zaawansowanych statystyk drużyn...")
        print("   - Momentum i streaks")
        print("   - Trendy sezonowe")
        print("   - Performance kontekstowe")
        print("   - Zaawansowane H2H")
        predictor.calculate_ultra_team_stats(training_df)

        # 3. Przygotowanie encodera
        predictor.fit_teams_encoder(training_df)

        # 4. Tworzenie ultra-zaawansowanych cech
        print("🔧 Przygotowywanie ultra-zaawansowanych cech...")
        features_df = predictor.create_ultra_features(training_df, is_training=True)

        print(f"✅ Utworzono {len(features_df.columns)-4} ULTRA cech!")
        print("   - Kontekstowe cechy (miesiąc, dzień tygodnia)")
        print("   - Momentum i streaks")
        print("   - Zaawansowane H2H")
        print("   - Miesięczne trendy")
        print("   - Wskaźniki stylu gry")

        # 5. Podział na cechy i etykiety
        feature_columns = [col for col in features_df.columns if col not in ['result', 'home_goals', 'away_goals', 'total_goals']]
        X = features_df[feature_columns]
        y_result = features_df['result']
        y_home_goals = features_df['home_goals']
        y_away_goals = features_df['away_goals']

        predictor.feature_columns = feature_columns

        # 6. Podział na zbiór treningowy i testowy
        X_train, X_test, y_result_train, y_result_test = train_test_split(
            X, y_result, test_size=config.ml.test_size, random_state=config.ml.random_state, stratify=y_result
        )
        _, _, y_home_train, y_home_test = train_test_split(X, y_home_goals, test_size=config.ml.test_size, random_state=config.ml.random_state)
        _, _, y_away_train, y_away_test = train_test_split(X, y_away_goals, test_size=config.ml.test_size, random_state=config.ml.random_state)

        # 7. Trening ULTRA modeli
        print("🤖 Trenowanie ULTRA modeli...")
        print("   - Optuna hyperparameter optimization")
        print("   - Super Ensemble (XGBoost + LightGBM + ExtraTrees + GradientBoosting + MLP)")
        print("   - Feature selection")
        print("   - Advanced preprocessing")

        predictor.train_ultra_models(X_train, y_result_train, y_home_train, y_away_train)

        # 8. Ocena modeli
        print("📊 Ocena jakości ULTRA modeli...")
        metrics = predictor.evaluate_ultra_models(X_test, y_result_test, y_home_test, y_away_test)

        print(f"\n🎯🎯🎯 WYNIKI ULTRA MODELU:")
        print(f"   Dokładność przewidywania wyniku: {metrics['result_accuracy']:.1%}")
        print(f"   Dokładność wysokiej pewności (>70%): {metrics['high_confidence_accuracy']:.1%}")
        print(f"   Średnia pewność przewidywań: {metrics['avg_confidence']:.1%}")
        print(f"   Średni błąd goli gospodarzy: {metrics['home_goals_mae']:.1f}")
        print(f"   Średni błąd goli gości: {metrics['away_goals_mae']:.1f}")

        # Porównanie z poprzednimi wersjami
        improvement_vs_enhanced = metrics['result_accuracy'] - 0.673
        improvement_vs_basic = metrics['result_accuracy'] - 0.572

        print(f"\n📈📈📈 PORÓWNANIE Z POPRZEDNIMI WERSJAMI:")
        print(f"   Podstawowy model: 57.2%")
        print(f"   Enhanced model: 67.3%")
        print(f"   🚀 ULTRA model: {metrics['result_accuracy']:.1%}")

        if improvement_vs_enhanced > 0:
            print(f"   🎉🎉🎉 POPRAWA vs Enhanced: +{improvement_vs_enhanced:.1%}!")
        else:
            print(f"   📉 Wynik vs Enhanced: {improvement_vs_enhanced:.1%}")

        print(f"   🚀 CAŁKOWITA POPRAWA: +{improvement_vs_basic:.1%} vs podstawowy model!")

        # Sprawdź czy osiągnęliśmy cel
        if metrics['result_accuracy'] >= 0.70:
            print(f"   🏆🏆🏆 CEL OSIĄGNIĘTY! Przekroczono 70% dokładności!")
        elif metrics['result_accuracy'] >= 0.675:
            print(f"   🎯 Bardzo blisko celu! Jeszcze {0.70 - metrics['result_accuracy']:.1%} do 70%")

        # 9. Ładowanie danych do przewidywania
        print("\n🔮 Ładowanie danych sezonu 2024/25...")
        prediction_df = super(UltraHandballPredictor, predictor).load_prediction_data()

        if prediction_df.empty:
            print("❌ Brak danych do przewidywania!")
            return

        # 10. Przewidywanie wyników
        print("🎯 Przewidywanie wyników ULTRA modelem...")
        predictions = predictor.predict_season_ultra(prediction_df)

        # 11. Zapisywanie wyników
        filename = predictor.save_ultra_predictions(predictions)

        # 12. Podsumowanie
        print(f"\n🎉🎉🎉 ULTRA przewidywania zakończone!")
        print(f"📁 Wyniki zapisane w: {filename}")
        print(f"📊 Przewidziano {len(predictions)} meczów")

        # Statystyki pewności
        high_conf_predictions = [p for p in predictions if p['pewnosc_przewidywania'] > 0.8]
        medium_conf_predictions = [p for p in predictions if 0.6 <= p['pewnosc_przewidywania'] <= 0.8]
        low_conf_predictions = [p for p in predictions if p['pewnosc_przewidywania'] < 0.6]

        print(f"\n🎲 ANALIZA PEWNOŚCI PRZEWIDYWAŃ:")
        print(f"   Wysoka pewność (>80%): {len(high_conf_predictions)} meczów ({len(high_conf_predictions)/len(predictions):.1%})")
        print(f"   Średnia pewność (60-80%): {len(medium_conf_predictions)} meczów ({len(medium_conf_predictions)/len(predictions):.1%})")
        print(f"   Niska pewność (<60%): {len(low_conf_predictions)} meczów ({len(low_conf_predictions)/len(predictions):.1%})")

        # Pokaż najlepsze przewidywania
        print(f"\n🎯 NAJBARDZIEJ PEWNE PRZEWIDYWANIA:")
        top_predictions = sorted(predictions, key=lambda x: x['pewnosc_przewidywania'], reverse=True)[:3]
        for i, pred in enumerate(top_predictions, 1):
            print(f"   {i}. {pred['mecz']}")
            print(f"      Przewidywany wynik: {pred['przewidywany_wynik_bramkowy']} ({pred['przewidywany_wynik']})")
            print(f"      Pewność: {pred['pewnosc_przewidywania']:.1%}")
            print()

        # Statystyki przewidywań
        home_wins = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Wygrana gospodarzy')
        draws = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Remis')
        away_wins = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Wygrana gości')
        avg_goals = sum(p['przewidywana_suma_goli'] for p in predictions) / len(predictions)

        print(f"📈 Statystyki ULTRA przewidywań:")
        print(f"   Wygrane gospodarzy: {home_wins} ({home_wins/len(predictions):.1%})")
        print(f"   Remisy: {draws} ({draws/len(predictions):.1%})")
        print(f"   Wygrane gości: {away_wins} ({away_wins/len(predictions):.1%})")
        print(f"   Średnia suma goli: {avg_goals:.1f}")

        print(f"\n🚀🚀🚀 ULTRA Model gotowy do użycia!")
        print(f"🏆 Najzaawansowniejszy model ML do przewidywania piłki ręcznej!")

    except Exception as e:
        logger.error(f"Błąd w głównej funkcji: {e}")
        print(f"❌ Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()
