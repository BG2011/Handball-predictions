#!/usr/bin/env python3
"""
Handball ML Predictor
Przewiduje wyniki meczów piłki ręcznej HBL używając XGBoost
Trenuje na danych z sezonów 2020-2024, przewiduje sezon 2024/25
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Import base class and config
from base_handball_predictor import BaseHandballPredictor
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HandballPredictor(BaseHandballPredictor):
    def __init__(self):
        super().__init__()
        # Use configuration from config.py
        self.training_files = config.files.training_files
        self.prediction_file = config.files.prediction_file
    
    def calculate_team_stats(self, df: pd.DataFrame):
        """Oblicza statystyki drużyn na podstawie danych historycznych - rozszerza podstawowe statystyki"""
        # Użyj podstawowych statystyk z klasy bazowej
        self.calculate_basic_team_stats(df)
        
        # Oblicz dodatkowe średnie dla zgodności z oryginalnym kodem
        for team, stats in self.team_stats.items():
            if stats['matches'] > 0:
                stats['avg_goals_for'] = stats['goals_for'] / stats['matches']
                stats['avg_goals_against'] = stats['goals_against'] / stats['matches']
                stats['win_rate'] = stats['wins'] / stats['matches']
                
                if stats['home_matches'] > 0:
                    stats['home_avg_goals_for'] = stats['home_goals_for'] / stats['home_matches']
                    stats['home_avg_goals_against'] = stats['home_goals_against'] / stats['home_matches']
                    stats['home_win_rate'] = stats['home_wins'] / stats['home_matches']
                
                if stats['away_matches'] > 0:
                    stats['away_avg_goals_for'] = stats['away_goals_for'] / stats['away_matches']
                    stats['away_avg_goals_against'] = stats['away_goals_against'] / stats['away_matches']
                    stats['away_win_rate'] = stats['away_wins'] / stats['away_matches']
        
        logger.info(f"Obliczono statystyki dla {len(self.team_stats)} drużyn")
    
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Tworzy cechy do modelu ML - rozszerza podstawowe cechy"""
        # Użyj podstawowych cech z klasy bazowej
        features_df = self.create_basic_features(df)
        
        # Dodaj dodatkowe cechy specyficzne dla XGBoost
        for i, (_, match) in enumerate(df.iterrows()):
            home_team = match['druzyna_domowa']
            away_team = match['druzyna_goscinna']
            
            # Statystyki drużyn
            home_stats = self.team_stats.get(home_team, {})
            away_stats = self.team_stats.get(away_team, {})
            
            # Dodaj dodatkowe cechy dla XGBoost
            features_df.loc[i, 'home_avg_goals_for'] = home_stats.get('avg_goals_for', 25)
            features_df.loc[i, 'home_avg_goals_against'] = home_stats.get('avg_goals_against', 25)
            features_df.loc[i, 'home_home_avg_goals_for'] = home_stats.get('home_avg_goals_for', 25)
            features_df.loc[i, 'home_home_avg_goals_against'] = home_stats.get('home_avg_goals_against', 25)
            features_df.loc[i, 'home_home_win_rate'] = home_stats.get('home_win_rate', 0.5)
            
            features_df.loc[i, 'away_avg_goals_for'] = away_stats.get('avg_goals_for', 25)
            features_df.loc[i, 'away_avg_goals_against'] = away_stats.get('avg_goals_against', 25)
            features_df.loc[i, 'away_away_avg_goals_for'] = away_stats.get('away_avg_goals_for', 25)
            features_df.loc[i, 'away_away_avg_goals_against'] = away_stats.get('away_avg_goals_against', 25)
            features_df.loc[i, 'away_away_win_rate'] = away_stats.get('away_win_rate', 0.5)
            
            # Różnice między drużynami
            features_df.loc[i, 'goal_diff_home'] = features_df.loc[i, 'home_avg_goals_for'] - features_df.loc[i, 'home_avg_goals_against']
            features_df.loc[i, 'goal_diff_away'] = features_df.loc[i, 'away_avg_goals_for'] - features_df.loc[i, 'away_avg_goals_against']
            features_df.loc[i, 'win_rate_diff'] = features_df.loc[i, 'home_win_rate'] - features_df.loc[i, 'away_win_rate']
            features_df.loc[i, 'attack_vs_defense'] = features_df.loc[i, 'home_avg_goals_for'] - features_df.loc[i, 'away_avg_goals_against']
            features_df.loc[i, 'defense_vs_attack'] = features_df.loc[i, 'away_avg_goals_for'] - features_df.loc[i, 'home_avg_goals_against']
            
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
                
                features_df.loc[i, 'result'] = result
                features_df.loc[i, 'home_goals'] = home_goals
                features_df.loc[i, 'away_goals'] = away_goals
                features_df.loc[i, 'total_goals'] = home_goals + away_goals
        
        return features_df
    
    def train_models(self, X_train: pd.DataFrame, y_result: pd.Series, y_home_goals: pd.Series, y_away_goals: pd.Series):
        """Trenuje modele XGBoost"""
        logger.info("Rozpoczynam trening modeli...")
        
        # Użyj parametrów z konfiguracji
        xgb_params = config.ml.xgb_params.copy()
        
        # Model wyniku meczu (klasyfikacja)
        self.result_model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=200,  # Zwiększona wartość dla lepszej jakości
            **xgb_params
        )
        
        # Model goli gospodarzy (regresja)
        self.goals_home_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,  # Zwiększona wartość dla lepszej jakości
            **xgb_params
        )
        
        # Model goli gości (regresja)
        self.goals_away_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,  # Zwiększona wartość dla lepszej jakości
            **xgb_params
        )
        
        # Trenowanie
        self.result_model.fit(X_train, y_result)
        self.goals_home_model.fit(X_train, y_home_goals)
        self.goals_away_model.fit(X_train, y_away_goals)
        
        logger.info("Modele wytrenowane pomyślnie!")
    
    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Przewiduje wynik pojedynczego meczu"""
        # Tworzy DataFrame dla pojedynczego meczu
        match_data = pd.DataFrame([{
            'druzyna_domowa': home_team,
            'druzyna_goscinna': away_team,
            'mecz': f"{home_team} - {away_team}",
            'data': datetime.now().isoformat()
        }])
        
        # Przygotuj cechy
        X_pred = self.create_features(match_data, is_training=False)
        
        # Usuń kolumny z etykietami jeśli istnieją
        feature_columns = [col for col in X_pred.columns if col not in ['result', 'home_goals', 'away_goals', 'total_goals']]
        X_pred_clean = X_pred[feature_columns]
        
        # Przewidywania
        result_pred = self.result_model.predict(X_pred_clean)[0]
        result_proba = self.result_model.predict_proba(X_pred_clean)[0]
        home_goals_pred = max(0, round(self.goals_home_model.predict(X_pred_clean)[0]))
        away_goals_pred = max(0, round(self.goals_away_model.predict(X_pred_clean)[0]))
        
        # Mapowanie wyników
        result_labels = ['Wygrana gospodarzy', 'Remis', 'Wygrana gości']
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_result': result_labels[result_pred],
            'home_win_probability': float(result_proba[0]),
            'draw_probability': float(result_proba[1]),
            'away_win_probability': float(result_proba[2]),
            'predicted_home_goals': int(home_goals_pred),
            'predicted_away_goals': int(away_goals_pred),
            'predicted_score': f"{int(home_goals_pred)}:{int(away_goals_pred)}",
            'predicted_total_goals': int(home_goals_pred + away_goals_pred)
        }
    
    def evaluate_models(self, X_test: pd.DataFrame, y_result_test: pd.Series, 
                       y_home_goals_test: pd.Series, y_away_goals_test: pd.Series):
        """Ocenia jakość modeli"""
        logger.info("Oceniam jakość modeli...")
        
        # Przewidywania
        result_pred = self.result_model.predict(X_test)
        home_goals_pred = self.goals_home_model.predict(X_test)
        away_goals_pred = self.goals_away_model.predict(X_test)
        
        # Metryki dla wyniku
        result_accuracy = accuracy_score(y_result_test, result_pred)
        logger.info(f"Dokładność przewidywania wyniku: {result_accuracy:.3f}")
        
        # Metryki dla goli
        home_goals_mae = mean_absolute_error(y_home_goals_test, home_goals_pred)
        away_goals_mae = mean_absolute_error(y_away_goals_test, away_goals_pred)
        
        logger.info(f"MAE gole gospodarzy: {home_goals_mae:.2f}")
        logger.info(f"MAE gole gości: {away_goals_mae:.2f}")
        
        return {
            'result_accuracy': result_accuracy,
            'home_goals_mae': home_goals_mae,
            'away_goals_mae': away_goals_mae
        }

    def predict_season(self, prediction_df: pd.DataFrame) -> List[Dict]:
        """Przewiduje wyniki dla całego sezonu"""
        logger.info("Przewidywanie wyników sezonu 2024/25...")

        # Przygotuj cechy
        X_pred = self.create_features(prediction_df, is_training=False)

        # Usuń kolumny z etykietami jeśli istnieją
        feature_columns = [col for col in X_pred.columns if col not in ['result', 'home_goals', 'away_goals', 'total_goals']]
        X_pred_clean = X_pred[feature_columns]

        # Przewidywania
        result_pred = self.result_model.predict(X_pred_clean)
        result_proba = self.result_model.predict_proba(X_pred_clean)
        home_goals_pred = self.goals_home_model.predict(X_pred_clean)
        away_goals_pred = self.goals_away_model.predict(X_pred_clean)

        # Przygotuj wyniki
        predictions = []
        result_labels = ['Wygrana gospodarzy', 'Remis', 'Wygrana gości']

        for i, (_, match) in enumerate(prediction_df.iterrows()):
            home_goals = max(0, round(home_goals_pred[i]))
            away_goals = max(0, round(away_goals_pred[i]))

            prediction = {
                'mecz': match['mecz'],
                'data': match['data'],
                'druzyna_domowa': match['druzyna_domowa'],
                'druzyna_goscinna': match['druzyna_goscinna'],
                'przewidywany_wynik': result_labels[result_pred[i]],
                'prawdopodobienstwo_wygranej_gospodarzy': float(result_proba[i][0]),
                'prawdopodobienstwo_remisu': float(result_proba[i][1]),
                'prawdopodobienstwo_wygranej_gosci': float(result_proba[i][2]),
                'przewidywane_gole_gospodarzy': int(home_goals),
                'przewidywane_gole_gosci': int(away_goals),
                'przewidywany_wynik_bramkowy': f"{int(home_goals)}:{int(away_goals)}",
                'przewidywana_suma_goli': int(home_goals + away_goals),
                'runda': match['runda']
            }
            predictions.append(prediction)

        logger.info(f"Przewidziano wyniki dla {len(predictions)} meczów")
        return predictions

    def save_predictions(self, predictions: List[Dict], filename: str = None):
        """Zapisuje przewidywania do pliku JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hbl_predictions_2024_25_{timestamp}.json"

        data_to_save = {
            'metadata': {
                'season': 'DAIKIN HBL 2024/25',
                'total_predictions': len(predictions),
                'generated_at': datetime.now().isoformat(),
                'model_info': {
                    'algorithm': 'XGBoost',
                    'training_seasons': ['2020/21', '2021/22', '2022/23', '2023/24'],
                    'features_used': ['team_stats', 'home_advantage', 'historical_performance']
                }
            },
            'predictions': predictions
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f"Przewidywania zapisane do: {filename}")
        return filename

    def run_prediction(self):
        """Główna metoda wykonawcza - implementuje przepływ predykcji"""
        try:
            print("🏐 Handball ML Predictor - Przewidywanie wyników HBL 2024/25")
            print("=" * 70)

            # 1. Ładowanie danych treningowych
            print("\n📊 Ładowanie danych treningowych...")
            training_df = self.load_training_data()

            if training_df.empty:
                print("❌ Brak danych treningowych!")
                return

            # 2. Obliczanie statystyk drużyn
            print("📈 Obliczanie statystyk drużyn...")
            self.calculate_team_stats(training_df)

            # 3. Przygotowanie encodera dla drużyn
            print("🔧 Przygotowywanie encodera drużyn...")
            self.fit_teams_encoder(training_df)

            # 4. Tworzenie cech treningowych
            print("🔧 Przygotowywanie cech do treningu...")
            features_df = self.create_features(training_df, is_training=True)

            # 5. Podział na cechy i etykiety
            feature_columns = [col for col in features_df.columns if col not in ['result', 'home_goals', 'away_goals', 'total_goals']]
            X = features_df[feature_columns]
            y_result = features_df['result']
            y_home_goals = features_df['home_goals']
            y_away_goals = features_df['away_goals']

            # 6. Podział na zbiór treningowy i testowy
            X_train, X_test, y_result_train, y_result_test = train_test_split(
                X, y_result, test_size=config.ml.test_size, random_state=config.ml.random_state
            )
            _, _, y_home_train, y_home_test = train_test_split(
                X, y_home_goals, test_size=config.ml.test_size, random_state=config.ml.random_state
            )
            _, _, y_away_train, y_away_test = train_test_split(
                X, y_away_goals, test_size=config.ml.test_size, random_state=config.ml.random_state
            )

            # 7. Trening modeli
            print("🤖 Trenowanie modeli XGBoost...")
            self.train_models(X_train, y_result_train, y_home_train, y_away_train)

            # 8. Ocena modeli
            print("📊 Ocena jakości modeli...")
            metrics = self.evaluate_models(X_test, y_result_test, y_home_test, y_away_test)

            print(f"✅ Wyniki oceny:")
            print(f"   Dokładność przewidywania wyniku: {metrics['result_accuracy']:.1%}")
            print(f"   Średni błąd goli gospodarzy: {metrics['home_goals_mae']:.1f}")
            print(f"   Średni błąd goli gości: {metrics['away_goals_mae']:.1f}")

            # 9. Ładowanie danych do przewidywania
            print("\n🔮 Ładowanie danych sezonu 2024/25...")
            prediction_df = self.load_prediction_data()

            if prediction_df.empty:
                print("❌ Brak danych do przewidywania!")
                return

            # 10. Przewidywanie wyników
            print("🎯 Przewidywanie wyników...")
            predictions = self.predict_season(prediction_df)

            # 11. Zapisywanie wyników
            filename = self.save_predictions(predictions)

            # 12. Podsumowanie
            print(f"\n🎉 Przewidywania zakończone pomyślnie!")
            print(f"📁 Wyniki zapisane w: {filename}")
            print(f"📊 Przewidziano {len(predictions)} meczów")

            # Pokaż przykładowe przewidywania
            print(f"\n🎯 Przykładowe przewidywania:")
            for i, pred in enumerate(predictions[:5]):
                print(f"   {pred['mecz']}")
                print(f"      Przewidywany wynik: {pred['przewidywany_wynik_bramkowy']} ({pred['przewidywany_wynik']})")
                print(f"      Prawdopodobieństwa: W1: {pred['prawdopodobienstwo_wygranej_gospodarzy']:.1%}, "
                      f"X: {pred['prawdopodobienstwo_remisu']:.1%}, W2: {pred['prawdopodobienstwo_wygranej_gosci']:.1%}")
                print()

            # Statystyki przewidywań
            home_wins = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Wygrana gospodarzy')
            draws = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Remis')
            away_wins = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Wygrana gości')
            avg_goals = sum(p['przewidywana_suma_goli'] for p in predictions) / len(predictions)

            print(f"📈 Statystyki przewidywań:")
            print(f"   Wygrane gospodarzy: {home_wins} ({home_wins/len(predictions):.1%})")
            print(f"   Remisy: {draws} ({draws/len(predictions):.1%})")
            print(f"   Wygrane gości: {away_wins} ({away_wins/len(predictions):.1%})")
            print(f"   Średnia suma goli: {avg_goals:.1f}")

            return filename

        except Exception as e:
            logger.error(f"Błąd w głównej funkcji: {e}")
            print(f"❌ Wystąpił błąd: {e}")
            return None

def main():
    """Główna funkcja wykonawcza"""
    predictor = HandballPredictor()
    predictor.run_prediction()

if __name__ == "__main__":
    main()
