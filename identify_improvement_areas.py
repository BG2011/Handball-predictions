#!/usr/bin/env python3
"""
Identyfikacja obszarów do poprawy w modelu przewidywań handball
Analizuje dokładnie 5 głównych problemów:
1. Dokładność przewidywania remisów
2. Różnica między przewidywaniami dom/away
3. Dokładność przewidywania goli
4. Fałszywa pewność modelu
5. Problemy z konkretnymi drużynami
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovementAreasAnalyzer:
    def __init__(self):
        self.predictions_file = 'hbl_predictions_ENHANCED_2024_25_20250715_070326.json'
        self.reality_file = 'daikin_hbl_2024_25_FULL_20250715_065530.json'
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Ładuje przewidywania i rzeczywiste wyniki"""
        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)['predictions']
            
            with open(self.reality_file, 'r', encoding='utf-8') as f:
                reality = json.load(f)['matches']
                
            return predictions, reality
        except Exception as e:
            logger.error(f"Błąd ładowania danych: {e}")
            return [], []
    
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
                            
                            # Porównania
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
    
    def analyze_draw_prediction(self, df: pd.DataFrame) -> Dict:
        """Analizuje dokładność przewidywania remisów"""
        print("\n🎯 ANALIZA PRZEWIDYWANIA REMISÓW")
        print("=" * 50)
        
        # Rzeczywiste remisy
        real_draws = df[df['real_wynik'] == 'Remis']
        predicted_draws = df[df['pred_wynik'] == 'Remis']
        
        print(f"Rzeczywiste remisy: {len(real_draws)} meczów")
        print(f"Przewidywane remisy: {len(predicted_draws)} meczów")
        
        # Dokładność przewidywania remisów
        correct_draws = len(df[(df['real_wynik'] == 'Remis') & (df['pred_wynik'] == 'Remis')])
        draw_precision = correct_draws / len(predicted_draws) if len(predicted_draws) > 0 else 0
        draw_recall = correct_draws / len(real_draws) if len(real_draws) > 0 else 0
        
        print(f"Poprawnie przewidziane remisy: {correct_draws}")
        print(f"Precyzja remisów: {draw_precision:.1%}")
        print(f"Czułość remisów: {draw_recall:.1%}")
        
        # Analiza prawdopodobieństw remisów
        if len(real_draws) > 0:
            avg_draw_prob_real = real_draws['pred_prob_draw'].mean()
            print(f"Średnie prawdopodobieństwo remisu dla rzeczywistych remisów: {avg_draw_prob_real:.1%}")
        
        return {
            'real_draws': len(real_draws),
            'predicted_draws': len(predicted_draws),
            'correct_draws': correct_draws,
            'draw_precision': draw_precision,
            'draw_recall': draw_recall
        }
    
    def analyze_home_away_bias(self, df: pd.DataFrame) -> Dict:
        """Analizuje różnice między przewidywaniami dom/away"""
        print("\n🏠 ANALIZA DOMOWEJ PRZEWAGI I BIASU")
        print("=" * 50)
        
        # Dokładność według typu wyniku
        home_wins_real = df[df['real_wynik'] == 'Wygrana gospodarzy']
        away_wins_real = df[df['real_wynik'] == 'Wygrana gości']
        
        home_accuracy = len(home_wins_real[home_wins_real['result_correct']]) / len(home_wins_real)
        away_accuracy = len(away_wins_real[away_wins_real['result_correct']]) / len(away_wins_real)
        
        print(f"Dokładność przewidywania wygranych gospodarzy: {home_accuracy:.1%}")
        print(f"Dokładność przewidywania wygranych gości: {away_accuracy:.1%}")
        print(f"Różnica: {abs(home_accuracy - away_accuracy):.1%}")
        
        # Przewidywana vs rzeczywista przewaga domowa
        pred_home_wins = (df['pred_wynik'] == 'Wygrana gospodarzy').sum()
        real_home_wins = (df['real_wynik'] == 'Wygrana gospodarzy').sum()
        
        pred_home_rate = pred_home_wins / len(df)
        real_home_rate = real_home_wins / len(df)
        
        print(f"Przewidywana przewaga domowa: {pred_home_rate:.1%}")
        print(f"Rzeczywista przewaga domowa: {real_home_rate:.1%}")
        print(f"Nadmiarowa przewaga domowa: {pred_home_rate - real_home_rate:.1%}")
        
        return {
            'home_accuracy': home_accuracy,
            'away_accuracy': away_accuracy,
            'accuracy_difference': abs(home_accuracy - away_accuracy),
            'predicted_home_advantage': pred_home_rate,
            'real_home_advantage': real_home_rate,
            'home_advantage_bias': pred_home_rate - real_home_rate
        }
    
    def analyze_goal_prediction_accuracy(self, df: pd.DataFrame) -> Dict:
        """Analizuje dokładność przewidywania goli"""
        print("\n⚽ ANALIZA DOKŁADNOŚCI PRZEWIDYWANIA GOLI")
        print("=" * 50)
        
        # Podstawowe statystyki błędów
        total_goals_mae = df['total_goals_diff'].mean()
        
        # Procent w różnych przedziałach błędu
        within_3_goals = (df['total_goals_diff'] <= 3).sum() / len(df)
        within_5_goals = (df['total_goals_diff'] <= 5).sum() / len(df)
        over_10_goals = (df['total_goals_diff'] > 10).sum() / len(df)
        
        print(f"Średni błąd sumy goli: {total_goals_mae:.1f}")
        print(f"Procent przewidywań w granicy 3 goli: {within_3_goals:.1%}")
        print(f"Procent przewidywań w granicy 5 goli: {within_5_goals:.1%}")
        print(f"Procent przewidywań z błędem >10 goli: {over_10_goals:.1%}")
        
        return {
            'total_goals_mae': total_goals_mae,
            'within_3_goals': within_3_goals,
            'within_5_goals': within_5_goals,
            'over_10_goals': over_10_goals
        }
    
    def analyze_overconfident_predictions(self, df: pd.DataFrame) -> Dict:
        """Analizuje fałszywą pewność modelu"""
        print("\n🎲 ANALIZA FAŁSZYWEJ PEWNOŚCI")
        print("=" * 50)
        
        # Przewidywania z wysoką pewnością (>80%) które się nie sprawdziły
        high_confidence = df[df['max_confidence'] > 0.8]
        wrong_high_conf = high_confidence[~high_confidence['result_correct']]
        
        print(f"Przewidywania z pewnością >80%: {len(high_confidence)} meczów")
        print(f"Z nich błędnych: {len(wrong_high_conf)} meczów")
        print(f"Procent błędnych wysokich pewności: {len(wrong_high_conf)/len(high_confidence)*100:.1f}%")
        
        # Szczegóły błędnych wysokich pewności
        if len(wrong_high_conf) > 0:
            print(f"\nNajbardziej mylące wysokie pewności:")
            for _, match in wrong_high_conf.head(5).iterrows():
                print(f"   {match['mecz']}: przewidywano {match['pred_wynik']} z {match['max_confidence']:.1%} pewności, "
                      f"rzeczywiste: {match['real_wynik']}")
        
        return {
            'high_confidence_count': len(high_confidence),
            'wrong_high_confidence': len(wrong_high_conf),
            'false_confidence_rate': len(wrong_high_conf) / len(high_confidence) if len(high_confidence) > 0 else 0
        }
    
    def analyze_team_specific_issues(self, df: pd.DataFrame) -> Dict:
        """Analizuje problemy z konkretnymi drużynami"""
        print("\n🏆 ANALIZA PROBLEMÓW Z DRUŻYNAMI")
        print("=" * 50)
        
        # Analiza dokładności dla każdej drużyny
        team_stats = {}
        
        for _, match in df.iterrows():
            home_team = match['druzyna_domowa']
            away_team = match['druzyna_goscinna']
            
            # Inicjalizacja statystyk
            if home_team not in team_stats:
                team_stats[home_team] = {'home_matches': 0, 'home_correct': 0, 'away_matches': 0, 'away_correct': 0}
            if away_team not in team_stats:
                team_stats[away_team] = {'home_matches': 0, 'home_correct': 0, 'away_matches': 0, 'away_correct': 0}
            
            # Statystyki dla gospodarza
            team_stats[home_team]['home_matches'] += 1
            if match['result_correct']:
                team_stats[home_team]['home_correct'] += 1
            
            # Statystyki dla gościa
            team_stats[away_team]['away_matches'] += 1
            if match['result_correct']:
                team_stats[away_team]['away_correct'] += 1
        
        # Oblicz dokładności
        team_accuracies = {}
        for team, stats in team_stats.items():
            total_matches = stats['home_matches'] + stats['away_matches']
            total_correct = stats['home_correct'] + stats['away_correct']
            if total_matches > 0:
                team_accuracies[team] = {
                    'total_accuracy': total_correct / total_matches,
                    'home_accuracy': stats['home_correct'] / stats['home_matches'] if stats['home_matches'] > 0 else 0,
                    'away_accuracy': stats['away_correct'] / stats['away_matches'] if stats['away_matches'] > 0 else 0,
                    'total_matches': total_matches
                }
        
        # Sortuj według najgorszej dokładności
        worst_teams = sorted(team_accuracies.items(), key=lambda x: x[1]['total_accuracy'])[:10]
        
        print("Drużyny z najgorszą dokładnością:")
        for team, stats in worst_teams:
            print(f"   {team}: {stats['total_accuracy']:.1%} ({stats['total_matches']} meczów)")
            print(f"      Dom: {stats['home_accuracy']:.1%}, Wyjazd: {stats['away_accuracy']:.1%}")
        
        # Szczególnie problematyczne drużyny
        tsv_hannover = team_accuracies.get('TSV Hannover-Burgdorf', {})
        if tsv_hannover:
            print(f"\nTSV Hannover-Burgdorf:")
            print(f"   Ogólna dokładność: {tsv_hannover['total_accuracy']:.1%}")
            print(f"   Dokładność wyjazdów: {tsv_hannover['away_accuracy']:.1%}")
        
        return {
            'worst_teams': worst_teams,
            'team_accuracies': team_accuracies
        }
    
    def run_analysis(self):
        """Główna funkcja analizy"""
        print("🔍 IDENTYFIKACJA OBSZARÓW DO POPRAWY")
        print("=" * 60)
        
        # Załaduj dane
        predictions, reality = self.load_data()
        if not predictions or not reality:
            print("❌ Nie udało się załadować danych")
            return
        
        # Dopasuj dane
        matched_data = self.match_predictions_with_reality(predictions, reality)
        if not matched_data:
            print("❌ Nie udało się dopasować danych")
            return
        
        df = pd.DataFrame(matched_data)
        print(f"✅ Przeanalizowano {len(df)} meczów")
        
        # Analizuj każdy obszar
        draw_stats = self.analyze_draw_prediction(df)
        bias_stats = self.analyze_home_away_bias(df)
        goal_stats = self.analyze_goal_prediction_accuracy(df)
        confidence_stats = self.analyze_overconfident_predictions(df)
        team_stats = self.analyze_team_specific_issues(df)
        
        # Podsumowanie
        print("\n🎯 PODSUMOWANIE GŁÓWNYCH PROBLEMÓW")
        print("=" * 60)
        
        print("1. 📊 PROBLEM: Słabe przewidywanie remisów")
        print(f"   Rzeczywiste remisy: {draw_stats['real_draws']}")
        print(f"   Poprawnie przewidziane: {draw_stats['correct_draws']}")
        print(f"   Dokładność: {draw_stats['draw_precision']:.1%}")
        
        print("\n2. 🏠 PROBLEM: Silna przewaga domowa")
        print(f"   Dokładność gospodarzy: {bias_stats['home_accuracy']:.1%}")
        print(f"   Dokładność gości: {bias_stats['away_accuracy']:.1%}")
        print(f"   Różnica: {bias_stats['accuracy_difference']:.1%}")
        
        print("\n3. ⚽ PROBLEM: Niedokładne przewidywanie goli")
        print(f"   Średni błąd: {goal_stats['total_goals_mae']:.1f} goli")
        print(f"   Tylko {goal_stats['within_3_goals']:.1%} w granicy 3 goli")
        print(f"   {goal_stats['over_10_goals']:.1%} z błędem >10 goli")
        
        print("\n4. 🎲 PROBLEM: Fałszywa pewność modelu")
        print(f"   Przewidywania z pewnością >80%: {confidence_stats['high_confidence_count']}")
        print(f"   Z nich błędnych: {confidence_stats['wrong_high_confidence']}")
        print(f"   Wskaźnik fałszywej pewności: {confidence_stats['false_confidence_rate']:.1%}")
        
        print("\n5. 🏆 PROBLEM: Problemy z konkretnymi drużynami")
        if team_stats['worst_teams']:
            worst_team = team_stats['worst_teams'][0]
            print(f"   Najgorsza drużyna: {worst_team[0]} ({worst_team[1]['total_accuracy']:.1%})")
        
        return {
            'draw_stats': draw_stats,
            'bias_stats': bias_stats,
            'goal_stats': goal_stats,
            'confidence_stats': confidence_stats,
            'team_stats': team_stats
        }

def main():
    """Główna funkcja"""
    analyzer = ImprovementAreasAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
