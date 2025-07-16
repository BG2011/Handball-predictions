#!/usr/bin/env python3
"""
Szczegółowa analiza dokładności modelu Enhanced
Analizuje osobno dokładność przewidywania zwycięzcy i goli
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedModelAnalyzer:
    def __init__(self):
        self.enhanced_predictions_file = 'hbl_predictions_ENHANCED_2024_25_20250715_070326.json'
        self.reality_file = 'daikin_hbl_2024_25_FULL_20250715_065530.json'
        
    def load_enhanced_predictions(self) -> List[Dict]:
        """Ładuje przewidywania Enhanced modelu"""
        try:
            with open(self.enhanced_predictions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['predictions']
        except Exception as e:
            logger.error(f"Błąd ładowania przewidywań Enhanced: {e}")
            return []
    
    def load_reality(self) -> List[Dict]:
        """Ładuje rzeczywiste wyniki"""
        try:
            with open(self.reality_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['matches']
        except Exception as e:
            logger.error(f"Błąd ładowania rzeczywistych wyników: {e}")
            return []
    
    def match_predictions_with_reality(self, predictions: List[Dict], reality: List[Dict]) -> List[Dict]:
        """Dopasowuje przewidywania Enhanced z rzeczywistymi wynikami"""
        matched_data = []
        
        # Stwórz słownik rzeczywistych wyników
        reality_dict = {}
        for match in reality:
            key = f"{match['druzyna_domowa']} vs {match['druzyna_goscinna']}"
            reality_dict[key] = match
        
        for pred in predictions:
            match_key = pred['mecz']
            if match_key in reality_dict:
                real_match = reality_dict[match_key]
                
                # Sprawdź czy mecz został rozegrany
                if (real_match['status'] == 'Final' and 
                    real_match['bramki_domowe'] and real_match['bramki_goscinne']):
                    
                    try:
                        real_home_goals = int(real_match['bramki_domowe'])
                        real_away_goals = int(real_match['bramki_goscinne'])
                        
                        # Określ rzeczywisty wynik
                        if real_home_goals > real_away_goals:
                            real_result = 'Wygrana gospodarzy'
                        elif real_home_goals < real_away_goals:
                            real_result = 'Wygrana gości'
                        else:
                            real_result = 'Remis'
                        
                        matched_data.append({
                            'mecz': match_key,
                            'data': pred['data'],
                            'druzyna_domowa': pred['druzyna_domowa'],
                            'druzyna_goscinna': pred['druzyna_goscinna'],
                            
                            # Przewidywania Enhanced
                            'pred_wynik': pred['przewidywany_wynik'],
                            'pred_home_goals': pred['przewidywane_gole_gospodarzy'],
                            'pred_away_goals': pred['przewidywane_gole_gosci'],
                            'pred_total_goals': pred['przewidywana_suma_goli'],
                            'pred_prob_home': pred['prawdopodobienstwo_wygranej_gospodarzy'],
                            'pred_prob_draw': pred['prawdopodobienstwo_remisu'],
                            'pred_prob_away': pred['prawdopodobienstwo_wygranej_gosci'],
                            
                            # Rzeczywistość
                            'real_wynik': real_result,
                            'real_home_goals': real_home_goals,
                            'real_away_goals': real_away_goals,
                            'real_total_goals': real_home_goals + real_away_goals,
                            'real_wynik_bramkowy': f"{real_home_goals}:{real_away_goals}",
                            
                            # Porównania
                            'result_correct': pred['przewidywany_wynik'] == real_result,
                            'home_goals_diff': abs(pred['przewidywane_gole_gospodarzy'] - real_home_goals),
                            'away_goals_diff': abs(pred['przewidywane_gole_gosci'] - real_away_goals),
                            'total_goals_diff': abs(pred['przewidywana_suma_goli'] - (real_home_goals + real_away_goals))
                        })
                    except ValueError:
                        continue
        
        logger.info(f"Dopasowano {len(matched_data)} meczów Enhanced")
        return matched_data
    
    def analyze_winner_prediction_accuracy(self, matched_data: List[Dict]):
        """Szczegółowa analiza dokładności przewidywania zwycięzcy"""
        if not matched_data:
            print("❌ Brak danych do analizy zwycięzcy")
            return
        
        df = pd.DataFrame(matched_data)
        
        print("🏆 SZCZEGÓŁOWA ANALIZA PRZEWIDYWANIA ZWYCIĘZCY (Enhanced Model)")
        print("=" * 70)
        
        # Ogólna dokładność
        total_matches = len(df)
        correct_results = df['result_correct'].sum()
        overall_accuracy = correct_results / total_matches
        
        print(f"\n📊 OGÓLNA DOKŁADNOŚĆ PRZEWIDYWANIA ZWYCIĘZCY:")
        print(f"   Poprawne przewidywania: {correct_results}/{total_matches}")
        print(f"   Ogólna dokładność: {overall_accuracy:.1%}")
        
        # Analiza według typu wyniku
        print(f"\n🎯 DOKŁADNOŚĆ WEDŁUG TYPU WYNIKU:")
        
        result_types = ['Wygrana gospodarzy', 'Remis', 'Wygrana gości']
        result_stats = {}
        
        for result_type in result_types:
            real_matches = df[df['real_wynik'] == result_type]
            if len(real_matches) > 0:
                correct_predictions = real_matches['result_correct'].sum()
                accuracy = correct_predictions / len(real_matches)
                result_stats[result_type] = {
                    'total': len(real_matches),
                    'correct': correct_predictions,
                    'accuracy': accuracy
                }
                print(f"   {result_type}: {correct_predictions}/{len(real_matches)} ({accuracy:.1%})")
            else:
                result_stats[result_type] = {'total': 0, 'correct': 0, 'accuracy': 0}
        
        # Macierz pomyłek
        print(f"\n🔍 MACIERZ POMYŁEK:")
        y_true = df['real_wynik'].tolist()
        y_pred = df['pred_wynik'].tolist()
        
        cm = confusion_matrix(y_true, y_pred, labels=result_types)
        
        print("        Przewidywane:")
        print("Rzeczywiste    W1    X    W2")
        for i, real_type in enumerate(['W1', 'X', 'W2']):
            row = f"{real_type:12s}"
            for j in range(3):
                row += f"{cm[i][j]:5d}"
            print(row)
        
        # Analiza prawdopodobieństw
        print(f"\n🎲 ANALIZA PRAWDOPODOBIEŃSTW:")
        
        # Sprawdź jak często wysokie prawdopodobieństwa były poprawne
        confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for conf_level in confidence_levels:
            high_confidence = df[df[['pred_prob_home', 'pred_prob_draw', 'pred_prob_away']].max(axis=1) > conf_level]
            if len(high_confidence) > 0:
                high_conf_accuracy = high_confidence['result_correct'].mean()
                print(f"   Przewidywania z pewnością >{conf_level:.0%}: {len(high_confidence)} meczów, dokładność: {high_conf_accuracy:.1%}")
        
        return result_stats
    
    def analyze_goals_prediction_accuracy(self, matched_data: List[Dict]):
        """Szczegółowa analiza dokładności przewidywania goli"""
        if not matched_data:
            print("❌ Brak danych do analizy goli")
            return
        
        df = pd.DataFrame(matched_data)
        
        print("\n⚽ SZCZEGÓŁOWA ANALIZA PRZEWIDYWANIA GOLI (Enhanced Model)")
        print("=" * 70)
        
        # Podstawowe metryki
        home_goals_mae = df['home_goals_diff'].mean()
        away_goals_mae = df['away_goals_diff'].mean()
        total_goals_mae = df['total_goals_diff'].mean()
        
        home_goals_rmse = np.sqrt(mean_squared_error(df['real_home_goals'], df['pred_home_goals']))
        away_goals_rmse = np.sqrt(mean_squared_error(df['real_away_goals'], df['pred_away_goals']))
        total_goals_rmse = np.sqrt(mean_squared_error(df['real_total_goals'], df['pred_total_goals']))
        
        print(f"\n📊 PODSTAWOWE METRYKI BŁĘDU:")
        print(f"   MAE gole gospodarzy: {home_goals_mae:.2f}")
        print(f"   MAE gole gości: {away_goals_mae:.2f}")
        print(f"   MAE suma goli: {total_goals_mae:.2f}")
        print(f"   RMSE gole gospodarzy: {home_goals_rmse:.2f}")
        print(f"   RMSE gole gości: {away_goals_rmse:.2f}")
        print(f"   RMSE suma goli: {total_goals_rmse:.2f}")
        
        # Dokładność w różnych przedziałach
        print(f"\n🎯 DOKŁADNOŚĆ W PRZEDZIAŁACH BŁĘDU:")
        
        tolerance_levels = [1, 2, 3, 5]
        
        for tolerance in tolerance_levels:
            home_within_tolerance = (df['home_goals_diff'] <= tolerance).sum()
            away_within_tolerance = (df['away_goals_diff'] <= tolerance).sum()
            total_within_tolerance = (df['total_goals_diff'] <= tolerance).sum()
            
            total_matches = len(df)
            
            print(f"   Błąd ≤{tolerance} goli:")
            print(f"      Gole gospodarzy: {home_within_tolerance}/{total_matches} ({home_within_tolerance/total_matches:.1%})")
            print(f"      Gole gości: {away_within_tolerance}/{total_matches} ({away_within_tolerance/total_matches:.1%})")
            print(f"      Suma goli: {total_within_tolerance}/{total_matches} ({total_within_tolerance/total_matches:.1%})")
        
        # Dokładne trafienia
        exact_home = (df['home_goals_diff'] == 0).sum()
        exact_away = (df['away_goals_diff'] == 0).sum()
        exact_total = (df['total_goals_diff'] == 0).sum()
        exact_both = ((df['home_goals_diff'] == 0) & (df['away_goals_diff'] == 0)).sum()
        
        print(f"\n🎯 DOKŁADNE TRAFIENIA:")
        print(f"   Dokładne gole gospodarzy: {exact_home}/{len(df)} ({exact_home/len(df):.1%})")
        print(f"   Dokładne gole gości: {exact_away}/{len(df)} ({exact_away/len(df):.1%})")
        print(f"   Dokładna suma goli: {exact_total}/{len(df)} ({exact_total/len(df):.1%})")
        print(f"   Dokładny wynik bramkowy: {exact_both}/{len(df)} ({exact_both/len(df):.1%})")
        
        # Porównanie średnich
        pred_avg_home = df['pred_home_goals'].mean()
        real_avg_home = df['real_home_goals'].mean()
        pred_avg_away = df['pred_away_goals'].mean()
        real_avg_away = df['real_away_goals'].mean()
        pred_avg_total = df['pred_total_goals'].mean()
        real_avg_total = df['real_total_goals'].mean()
        
        print(f"\n📈 PORÓWNANIE ŚREDNICH:")
        print(f"   Średnia goli gospodarzy - przewidywana: {pred_avg_home:.1f}, rzeczywista: {real_avg_home:.1f} (różnica: {abs(pred_avg_home - real_avg_home):.1f})")
        print(f"   Średnia goli gości - przewidywana: {pred_avg_away:.1f}, rzeczywista: {real_avg_away:.1f} (różnica: {abs(pred_avg_away - real_avg_away):.1f})")
        print(f"   Średnia suma goli - przewidywana: {pred_avg_total:.1f}, rzeczywista: {real_avg_total:.1f} (różnica: {abs(pred_avg_total - real_avg_total):.1f})")
        
        # Najlepsze i najgorsze przewidywania goli
        print(f"\n🏆 NAJLEPSZE PRZEWIDYWANIA GOLI (najmniejszy błąd sumy):")
        best_goals = df.nsmallest(5, 'total_goals_diff')
        for i, (_, match) in enumerate(best_goals.iterrows(), 1):
            print(f"   {i}. {match['mecz']}: przewidywane {match['pred_home_goals']}:{match['pred_away_goals']}, "
                  f"rzeczywiste {match['real_home_goals']}:{match['real_away_goals']} (błąd: {match['total_goals_diff']})")
        
        print(f"\n💥 NAJGORSZE PRZEWIDYWANIA GOLI (największy błąd sumy):")
        worst_goals = df.nlargest(5, 'total_goals_diff')
        for i, (_, match) in enumerate(worst_goals.iterrows(), 1):
            print(f"   {i}. {match['mecz']}: przewidywane {match['pred_home_goals']}:{match['pred_away_goals']}, "
                  f"rzeczywiste {match['real_home_goals']}:{match['real_away_goals']} (błąd: {match['total_goals_diff']})")
        
        return {
            'home_goals_mae': home_goals_mae,
            'away_goals_mae': away_goals_mae,
            'total_goals_mae': total_goals_mae,
            'exact_home_rate': exact_home/len(df),
            'exact_away_rate': exact_away/len(df),
            'exact_total_rate': exact_total/len(df),
            'exact_both_rate': exact_both/len(df)
        }

def main():
    """Główna funkcja analizy Enhanced modelu"""
    analyzer = EnhancedModelAnalyzer()
    
    try:
        print("🔍 SZCZEGÓŁOWA ANALIZA MODELU ENHANCED")
        print("🎯 Analiza dokładności przewidywania zwycięzcy vs goli")
        print("=" * 80)
        
        # Załaduj dane
        predictions = analyzer.load_enhanced_predictions()
        reality = analyzer.load_reality()
        
        if not predictions or not reality:
            print("❌ Nie udało się załadować danych")
            return
        
        print(f"📊 Załadowano {len(predictions)} przewidywań Enhanced i {len(reality)} rzeczywistych wyników")
        
        # Dopasuj dane
        matched_data = analyzer.match_predictions_with_reality(predictions, reality)
        
        if not matched_data:
            print("❌ Nie udało się dopasować danych")
            return
        
        print(f"✅ Dopasowano {len(matched_data)} meczów do analizy")
        
        # Analizuj przewidywanie zwycięzcy
        winner_stats = analyzer.analyze_winner_prediction_accuracy(matched_data)
        
        # Analizuj przewidywanie goli
        goals_stats = analyzer.analyze_goals_prediction_accuracy(matched_data)
        
        # Podsumowanie
        print(f"\n🎉 PODSUMOWANIE ANALIZY ENHANCED MODELU:")
        print(f"📊 Przeanalizowano {len(matched_data)} meczów")
        print(f"🏆 Dokładność przewidywania zwycięzcy: {sum(1 for m in matched_data if m['result_correct'])/len(matched_data):.1%}")
        print(f"⚽ Średni błąd goli: {goals_stats['total_goals_mae']:.1f}")
        print(f"🎯 Dokładne wyniki bramkowe: {goals_stats['exact_both_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Błąd w głównej funkcji: {e}")
        print(f"❌ Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()
