#!/usr/bin/env python3
"""
Porównanie przewidywań ML z rzeczywistymi wynikami
Analizuje dokładność modelu na podstawie prawdziwych danych sezonu DAIKIN HBL 2024/25
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionValidator:
    def __init__(self):
        self.predictions_file = 'hbl_predictions_2024_25_20250715_065054.json'
        self.reality_file = 'daikin_hbl_2024_25_FULL_20250715_065530.json'
        
    def load_predictions(self) -> List[Dict]:
        """Ładuje przewidywania ML"""
        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['predictions']
        except Exception as e:
            logger.error(f"Błąd ładowania przewidywań: {e}")
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
        """Dopasowuje przewidywania z rzeczywistymi wynikami"""
        matched_data = []
        
        # Stwórz słownik rzeczywistych wyników dla szybkiego wyszukiwania
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
                            
                            # Przewidywania
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
        
        logger.info(f"Dopasowano {len(matched_data)} meczów")
        return matched_data
    
    def analyze_accuracy(self, matched_data: List[Dict]):
        """Analizuje dokładność przewidywań"""
        if not matched_data:
            print("❌ Brak danych do analizy")
            return
        
        df = pd.DataFrame(matched_data)
        total_matches = len(df)
        
        print("🏐 ANALIZA DOKŁADNOŚCI PRZEWIDYWAŃ ML")
        print("=" * 50)
        
        # Dokładność wyniku
        correct_results = df['result_correct'].sum()
        result_accuracy = correct_results / total_matches
        
        print(f"\n🎯 DOKŁADNOŚĆ WYNIKU:")
        print(f"   Poprawne przewidywania: {correct_results}/{total_matches} ({result_accuracy:.1%})")
        
        # Analiza według typu wyniku
        result_breakdown = df.groupby('real_wynik')['result_correct'].agg(['count', 'sum', 'mean'])
        print(f"\n📊 DOKŁADNOŚĆ WEDŁUG TYPU WYNIKU:")
        for result_type, stats in result_breakdown.iterrows():
            print(f"   {result_type}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
        
        # Dokładność goli
        home_goals_mae = df['home_goals_diff'].mean()
        away_goals_mae = df['away_goals_diff'].mean()
        total_goals_mae = df['total_goals_diff'].mean()
        
        print(f"\n⚽ DOKŁADNOŚĆ GOLI (średni błąd absolutny):")
        print(f"   Gole gospodarzy: {home_goals_mae:.1f}")
        print(f"   Gole gości: {away_goals_mae:.1f}")
        print(f"   Suma goli: {total_goals_mae:.1f}")
        
        # Najlepsze i najgorsze przewidywania
        print(f"\n🏆 NAJLEPSZE PRZEWIDYWANIA (najmniejszy błąd goli):")
        best_predictions = df.nsmallest(5, 'total_goals_diff')
        for _, match in best_predictions.iterrows():
            print(f"   {match['mecz']}: przewidywane {match['pred_home_goals']}:{match['pred_away_goals']}, "
                  f"rzeczywiste {match['real_home_goals']}:{match['real_away_goals']} "
                  f"(błąd: {match['total_goals_diff']} goli)")
        
        print(f"\n💥 NAJGORSZE PRZEWIDYWANIA (największy błąd goli):")
        worst_predictions = df.nlargest(5, 'total_goals_diff')
        for _, match in worst_predictions.iterrows():
            print(f"   {match['mecz']}: przewidywane {match['pred_home_goals']}:{match['pred_away_goals']}, "
                  f"rzeczywiste {match['real_home_goals']}:{match['real_away_goals']} "
                  f"(błąd: {match['total_goals_diff']} goli)")
        
        # Analiza prawdopodobieństw
        print(f"\n🎲 ANALIZA PRAWDOPODOBIEŃSTW:")
        
        # Sprawdź jak często wysokie prawdopodobieństwa były poprawne
        high_confidence = df[df[['pred_prob_home', 'pred_prob_draw', 'pred_prob_away']].max(axis=1) > 0.7]
        if len(high_confidence) > 0:
            high_conf_accuracy = high_confidence['result_correct'].mean()
            print(f"   Przewidywania z wysoką pewnością (>70%): {high_conf_accuracy:.1%} dokładności")
        
        low_confidence = df[df[['pred_prob_home', 'pred_prob_draw', 'pred_prob_away']].max(axis=1) < 0.5]
        if len(low_confidence) > 0:
            low_conf_accuracy = low_confidence['result_correct'].mean()
            print(f"   Przewidywania z niską pewnością (<50%): {low_conf_accuracy:.1%} dokładności")
        
        # Porównanie średnich
        print(f"\n📈 PORÓWNANIE ŚREDNICH:")
        print(f"   Przewidywana średnia goli: {df['pred_total_goals'].mean():.1f}")
        print(f"   Rzeczywista średnia goli: {df['real_total_goals'].mean():.1f}")
        print(f"   Różnica: {abs(df['pred_total_goals'].mean() - df['real_total_goals'].mean()):.1f}")
        
        # Przewaga domowa
        pred_home_wins = (df['pred_wynik'] == 'Wygrana gospodarzy').sum()
        real_home_wins = (df['real_wynik'] == 'Wygrana gospodarzy').sum()
        
        print(f"\n🏠 PRZEWAGA DOMOWA:")
        print(f"   Przewidywane wygrane gospodarzy: {pred_home_wins} ({pred_home_wins/total_matches:.1%})")
        print(f"   Rzeczywiste wygrane gospodarzy: {real_home_wins} ({real_home_wins/total_matches:.1%})")
        
        return {
            'total_matches': total_matches,
            'result_accuracy': result_accuracy,
            'home_goals_mae': home_goals_mae,
            'away_goals_mae': away_goals_mae,
            'total_goals_mae': total_goals_mae
        }
    
    def save_comparison(self, matched_data: List[Dict], filename: str = None):
        """Zapisuje porównanie do pliku JSON"""
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_vs_reality_{timestamp}.json"
        
        data_to_save = {
            'metadata': {
                'total_comparisons': len(matched_data),
                'predictions_file': self.predictions_file,
                'reality_file': self.reality_file,
                'generated_at': datetime.now().isoformat()
            },
            'comparisons': matched_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Porównanie zapisane do: {filename}")
        return filename

def main():
    """Główna funkcja wykonawcza"""
    validator = PredictionValidator()
    
    try:
        print("🔍 Porównanie przewidywań ML z rzeczywistymi wynikami")
        print("=" * 60)
        
        # Załaduj dane
        predictions = validator.load_predictions()
        reality = validator.load_reality()
        
        if not predictions or not reality:
            print("❌ Nie udało się załadować danych")
            return
        
        print(f"📊 Załadowano {len(predictions)} przewidywań i {len(reality)} rzeczywistych wyników")
        
        # Dopasuj dane
        matched_data = validator.match_predictions_with_reality(predictions, reality)
        
        if not matched_data:
            print("❌ Nie udało się dopasować danych")
            return
        
        # Analizuj dokładność
        metrics = validator.analyze_accuracy(matched_data)
        
        # Zapisz porównanie
        filename = validator.save_comparison(matched_data)
        
        print(f"\n✅ Analiza zakończona!")
        print(f"📁 Szczegółowe porównanie zapisane w: {filename}")
        
    except Exception as e:
        logger.error(f"Błąd w głównej funkcji: {e}")
        print(f"❌ Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()
