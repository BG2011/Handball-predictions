#!/usr/bin/env python3
"""
Walidacja Ultra-Enhanced Modelu
Testuje rozwiązanie 5 głównych problemów:
1. Dokładność przewidywania remisów
2. Korekcja przewagi domowej
3. Dokładność przewidywania goli
4. Kalibracja pewności
5. Poprawa dla problematycznych drużyn
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraModelValidator:
    def __init__(self):
        self.ultra_predictions_file = 'hbl_predictions_ULTRA_2024_25_20250715_072459.json'
        self.reality_file = 'daikin_hbl_2024_25_FULL_20250715_065530.json'
        self.enhanced_predictions_file = 'hbl_predictions_ENHANCED_2024_25_20250715_070326.json'
        
    def load_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Ładuje przewidywania Ultra, Enhanced i rzeczywiste wyniki"""
        try:
            with open(self.ultra_predictions_file, 'r', encoding='utf-8') as f:
                ultra_predictions = json.load(f)['predictions']
            
            with open(self.enhanced_predictions_file, 'r', encoding='utf-8') as f:
                enhanced_predictions = json.load(f)['predictions']
            
            with open(self.reality_file, 'r', encoding='utf-8') as f:
                reality = json.load(f)['matches']
                
            return ultra_predictions, enhanced_predictions, reality
        except Exception as e:
            logger.error(f"Błąd ładowania danych: {e}")
            return [], [], []
    
    def match_predictions_with_reality(self, predictions: List[Dict], reality: List[Dict]) -> List[Dict]:
        """Dopasowuje przewidywania z rzeczywistymi wynikami"""
        matched_data = []
        
        reality_dict = {}
        for match in reality:
            key = f"{match['druzyna_domowa']} vs {match['druzyna_goscinna']}"
            reality_dict[key] = match
        
        for pred in predictions:
            match_key = pred['mecz']
            if match_key in reality_dict:
                real_match = reality_dict[match_key]
                
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
                            'model': pred.get('model_version', 'Unknown'),
                            'pred_wynik': pred['przewidywany_wynik'],
                            'pred_home_goals': pred['przewidywane_gole_gospodarzy'],
                            'pred_away_goals': pred['przewidywane_gole_gosci'],
                            'pred_total_goals': pred['przewidywana_suma_goli'],
                            'pred_prob_home': pred['prawdopodobienstwo_wygranej_gospodarzy'],
                            'pred_prob_draw': pred['prawdopodobienstwo_remisu'],
                            'pred_prob_away': pred['prawdopodobienstwo_wygranej_gosci'],
                            'real_wynik': real_result,
                            'real_home_goals': real_home_goals,
                            'real_away_goals': real_away_goals,
                            'real_total_goals': real_home_goals + real_away_goals,
                            'result_correct': pred['przewidywany_wynik'] == real_result,
                            'home_goals_diff': abs(pred['przewidywane_gole_gospodarzy'] - real_home_goals),
                            'away_goals_diff': abs(pred['przewidywane_gole_gosci'] - real_away_goals),
                            'total_goals_diff': abs(pred['przewidywana_suma_goli'] - (real_home_goals + real_away_goals)),
                            'max_confidence': max(pred['prawdopodobienstwo_wygranej_gospodarzy'], 
                                               pred['prawdopodobienstwo_remisu'], 
                                               pred['prawdopodobienstwo_wygranej_gosci'])
                        })
                    except ValueError:
                        continue
        
        return matched_data
    
    def compare_models(self, ultra_data: List[Dict], enhanced_data: List[Dict]) -> Dict:
        """Porównuje wyniki Ultra vs Enhanced modelu"""
        print("🔍 PORÓWNANIE ULTRA vs ENHANCED MODEL")
        print("=" * 60)
        
        # Create DataFrames
        ultra_df = pd.DataFrame(ultra_data)
        enhanced_df = pd.DataFrame(enhanced_data)
        
        # Overall accuracy
        ultra_accuracy = ultra_df['result_correct'].mean()
        enhanced_accuracy = enhanced_df['result_correct'].mean()
        
        print(f"\n📊 OGÓLNA DOKŁADNOŚĆ:")
        print(f"   Ultra: {ultra_accuracy:.1%}")
        print(f"   Enhanced: {enhanced_accuracy:.1%}")
        print(f"   Poprawa: {(ultra_accuracy - enhanced_accuracy)*100:.1f}pp")
        
        # Draw prediction accuracy
        real_draws = ultra_df[ultra_df['real_wynik'] == 'Remis']
        ultra_draw_correct = len(real_draws[real_draws['result_correct']])
        enhanced_draw_correct = len(enhanced_df[enhanced_df['real_wynik'] == 'Remis'])
        
        ultra_draw_accuracy = ultra_draw_correct / len(real_draws) if len(real_draws) > 0 else 0
        enhanced_draw_accuracy = enhanced_draw_correct / len(real_draws) if len(real_draws) > 0 else 0
        
        print(f"\n🎯 DOKŁADNOŚĆ REMISÓW:")
        print(f"   Ultra: {ultra_draw_accuracy:.1%} ({ultra_draw_correct}/{len(real_draws)})")
        print(f"   Enhanced: {enhanced_draw_accuracy:.1%} ({enhanced_draw_correct}/{len(real_draws)})")
        print(f"   Poprawa: {(ultra_draw_accuracy - enhanced_draw_accuracy)*100:.1f}pp")
        
        # Home/Away bias
        home_wins_real = ultra_df[ultra_df['real_wynik'] == 'Wygrana gospodarzy']
        away_wins_real = ultra_df[ultra_df['real_wynik'] == 'Wygrana gości']
        
        ultra_home_acc = len(home_wins_real[home_wins_real['result_correct']]) / len(home_wins_real)
        ultra_away_acc = len(away_wins_real[away_wins_real['result_correct']]) / len(away_wins_real)
        
        enhanced_home_acc = len(enhanced_df[enhanced_df['real_wynik'] == 'Wygrana gospodarzy']) / len(home_wins_real)
        enhanced_away_acc = len(enhanced_df[enhanced_df['real_wynik'] == 'Wygrana gości']) / len(away_wins_real)
        
        ultra_bias = abs(ultra_home_acc - ultra_away_acc)
        enhanced_bias = abs(enhanced_home_acc - enhanced_away_acc)
        
        print(f"\n🏠 KOREKCJA BIASU DOMOWEGO:")
        print(f"   Ultra - Home: {ultra_home_acc:.1%}, Away: {ultra_away_acc:.1%}, Bias: {ultra_bias:.1%}")
        print(f"   Enhanced - Home: {enhanced_home_acc:.1%}, Away: {enhanced_away_acc:.1%}, Bias: {enhanced_bias:.1%}")
        print(f"   Redukcja biasu: {(enhanced_bias - ultra_bias)*100:.1f}pp")
        
        # Goal prediction accuracy
        ultra_goal_mae = ultra_df['total_goals_diff'].mean()
        enhanced_goal_mae = enhanced_df['total_goals_diff'].mean()
        
        print(f"\n⚽ DOKŁADNOŚĆ GOLI:")
        print(f"   Ultra MAE: {ultra_goal_mae:.2f}")
        print(f"   Enhanced MAE: {enhanced_goal_mae:.2f}")
        print(f"   Poprawa: {(enhanced_goal_mae - ultra_goal_mae):.2f}")
        
        # Confidence calibration
        ultra_high_conf = ultra_df[ultra_df['max_confidence'] > 0.8]
        enhanced_high_conf = enhanced_df[enhanced_df['max_confidence'] > 0.8]
        
        ultra_false_conf = len(ultra_high_conf[~ultra_high_conf['result_correct']]) / len(ultra_high_conf) if len(ultra_high_conf) > 0 else 0
        enhanced_false_conf = len(enhanced_high_conf[~enhanced_high_conf['result_correct']]) / len(enhanced_high_conf) if len(enhanced_high_conf) > 0 else 0
        
        print(f"\n🎲 KALIBRACJA PEWNOŚCI:")
        print(f"   Ultra false confidence: {ultra_false_conf:.1%}")
        print(f"   Enhanced false confidence: {enhanced_false_conf:.1%}")
        print(f"   Poprawa: {(enhanced_false_conf - ultra_false_conf)*100:.1f}pp")
        
        return {
            'overall_improvement': (ultra_accuracy - enhanced_accuracy) * 100,
            'draw_improvement': (ultra_draw_accuracy - enhanced_draw_accuracy) * 100,
            'bias_reduction': (enhanced_bias - ultra_bias) * 100,
            'goal_improvement': enhanced_goal_mae - ultra_goal_mae,
            'calibration_improvement': (enhanced_false_conf - ultra_false_conf) * 100
        }
    
    def analyze_team_specific_improvements(self, ultra_data: List[Dict], enhanced_data: List[Dict]) -> Dict:
        """Analizuje poprawy dla konkretnych drużyn"""
        print("\n🏆 ANALIZA POPRAW DLA DRUŻYN")
        print("=" * 50)
        
        # Create DataFrames
        ultra_df = pd.DataFrame(ultra_data)
        enhanced_df = pd.DataFrame(enhanced_data)
        
        # Team-specific analysis
        team_stats = {}
        
        for team in ultra_df['mecz'].str.extract(r'(.+) vs')[0].unique():
            team_matches = ultra_df[ultra_df['mecz'].str.contains(team)]
            enhanced_team_matches = enhanced_df[enhanced_df['mecz'].str.contains(team)]
            
            if len(team_matches) > 0:
                ultra_acc = team_matches['result_correct'].mean()
                enhanced_acc = enhanced_team_matches['result_correct'].mean()
                
                team_stats[team] = {
                    'ultra_accuracy': ultra_acc,
                    'enhanced_accuracy': enhanced_acc,
                    'improvement': (ultra_acc - enhanced_acc) * 100,
                    'matches': len(team_matches)
                }
        
        # Sort by improvement
        best_improvements = sorted(team_stats.items(), key=lambda x: x[1]['improvement'], reverse=True)[:5]
        worst_improvements = sorted(team_stats.items(), key=lambda x: x[1]['improvement'])[:5]
        
        print("Największe poprawy:")
        for team, stats in best_improvements:
            print(f"   {team}: +{stats['improvement']:.1f}pp ({stats['matches']} meczów)")
        
        print("\nNajmniejsze poprawy:")
        for team, stats in worst_improvements:
            print(f"   {team}: {stats['improvement']:.1f}pp ({stats['matches']} meczów)")
        
        # TSV Hannover-Burgdorf specific
        tsv_matches = ultra_df[ultra_df['mecz'].str.contains('TSV Hannover-Burgdorf')]
        if len(tsv_matches) > 0:
            tsv_ultra_acc = tsv_matches['result_correct'].mean()
            tsv_enhanced_acc = enhanced_df[enhanced_df['mecz'].str.contains('TSV Hannover-Burgdorf')]['result_correct'].mean()
            print(f"\nTSV Hannover-Burgdorf:")
            print(f"   Ultra: {tsv_ultra_acc:.1%}, Enhanced: {tsv_enhanced_acc:.1%}")
            print(f"   Poprawa: {(tsv_ultra_acc - tsv_enhanced_acc)*100:.1f}pp")
        
        return team_stats
    
    def generate_improvement_report(self) -> Dict:
        """Generuje kompleksowy raport popraw"""
        print("📊 GENEROWANIE RAPORTU POPRAW ULTRA-MODELU")
        print("=" * 70)
        
        # Load data
        ultra_predictions, enhanced_predictions, reality = self.load_data()
        
        if not ultra_predictions or not enhanced_predictions or not reality:
            print("❌ Nie udało się załadować danych")
            return {}
        
        # Match predictions
        ultra_matched = self.match_predictions_with_reality(ultra_predictions, reality)
        enhanced_matched = self.match_predictions_with_reality(enhanced_predictions, reality)
        
        if not ultra_matched or not enhanced_matched:
            print("❌ Nie udało się dopasować danych")
            return {}
        
        print(f"✅ Przeanalizowano {len(ultra_matched)} meczów")
        
        # Compare models
        comparison = self.compare_models(ultra_matched, enhanced_matched)
        
        # Team analysis
        team_analysis = self.analyze_team_specific_improvements(ultra_matched, enhanced_matched)
        
        # Summary
        print("\n🎯 PODSUMOWANIE POPRAW ULTRA-MODELU")
        print("=" * 70)
        
        print("1. 📊 OGÓLNA POPRAWA:")
        print(f"   +{comparison['overall_improvement']:.1f}pp dokładności ogólnej")
        
        print("\n2. 🎯 POPRAWA REMISÓW:")
        print(f"   +{comparison['draw_improvement']:.1f}pp dokładności remisów")
        
        print("\n3. 🏠 KOREKCJA BIASU:")
        print(f"   -{comparison['bias_reduction']:.1f}pp różnicy dom/away")
        
        print("\n4. ⚽ POPRAWA GOLI:")
        print(f"   -{comparison['goal_improvement']:.2f} średni błąd goli")
        
        print("\n5. 🎲 KALIBRACJA PEWNOŚCI:")
        print(f"   -{comparison['calibration_improvement']:.1f}pp fałszywej pewności")
        
        # Target achievement
        print("\n🎯 CELE vs RZECZYWISTE:")
        print("   Cel: >65% dokładność ogólna")
        print("   Cel: >15% dokładność remisów")
        print("   Cel: <4.0 MAE goli")
        print("   Cel: <10% fałszywa pewność")
        
        return {
            'comparison': comparison,
            'team_analysis': team_analysis,
            'total_matches': len(ultra_matched),
            'summary': {
                'overall_improvement': comparison['overall_improvement'],
                'draw_improvement': comparison['draw_improvement'],
                'bias_reduction': comparison['bias_reduction'],
                'goal_improvement': comparison['goal_improvement'],
                'calibration_improvement': comparison['calibration_improvement']
            }
        }

def main():
    """Główna funkcja walidacji"""
