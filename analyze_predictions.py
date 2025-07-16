#!/usr/bin/env python3
"""
Analiza przewidywań HBL 2024/25
Pokazuje szczegółowe statystyki i najciekawsze mecze
"""

import json
import pandas as pd
from collections import Counter

def load_predictions(filename):
    """Ładuje przewidywania z pliku JSON"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['predictions']

def analyze_predictions(predictions):
    """Analizuje przewidywania"""
    df = pd.DataFrame(predictions)
    
    print("🏐 ANALIZA PRZEWIDYWAŃ HBL 2024/25")
    print("=" * 50)
    
    # Podstawowe statystyki
    total_matches = len(predictions)
    home_wins = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Wygrana gospodarzy')
    draws = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Remis')
    away_wins = sum(1 for p in predictions if p['przewidywany_wynik'] == 'Wygrana gości')
    
    print(f"\n📊 PODSTAWOWE STATYSTYKI:")
    print(f"Łączna liczba meczów: {total_matches}")
    print(f"Wygrane gospodarzy: {home_wins} ({home_wins/total_matches:.1%})")
    print(f"Remisy: {draws} ({draws/total_matches:.1%})")
    print(f"Wygrane gości: {away_wins} ({away_wins/total_matches:.1%})")
    
    # Statystyki goli
    home_goals = [p['przewidywane_gole_gospodarzy'] for p in predictions]
    away_goals = [p['przewidywane_gole_gosci'] for p in predictions]
    total_goals = [p['przewidywana_suma_goli'] for p in predictions]
    
    print(f"\n⚽ STATYSTYKI GOLI:")
    print(f"Średnia goli gospodarzy: {sum(home_goals)/len(home_goals):.1f}")
    print(f"Średnia goli gości: {sum(away_goals)/len(away_goals):.1f}")
    print(f"Średnia suma goli: {sum(total_goals)/len(total_goals):.1f}")
    print(f"Najwyższy wynik: {max(total_goals)} goli")
    print(f"Najniższy wynik: {min(total_goals)} goli")
    
    # Najciekawsze mecze
    print(f"\n🔥 NAJCIEKAWSZE MECZE:")
    
    # Mecze z najwyższą sumą goli
    high_scoring = sorted(predictions, key=lambda x: x['przewidywana_suma_goli'], reverse=True)[:5]
    print(f"\n🎯 Mecze z najwyższą sumą goli:")
    for i, match in enumerate(high_scoring, 1):
        print(f"{i}. {match['mecz']} - {match['przewidywany_wynik_bramkowy']} ({match['przewidywana_suma_goli']} goli)")
    
    # Mecze z najniższą sumą goli
    low_scoring = sorted(predictions, key=lambda x: x['przewidywana_suma_goli'])[:5]
    print(f"\n🛡️ Mecze z najniższą sumą goli:")
    for i, match in enumerate(low_scoring, 1):
        print(f"{i}. {match['mecz']} - {match['przewidywany_wynik_bramkowy']} ({match['przewidywana_suma_goli']} goli)")
    
    # Najbardziej pewne przewidywania
    certain_predictions = []
    for p in predictions:
        max_prob = max(p['prawdopodobienstwo_wygranej_gospodarzy'], 
                      p['prawdopodobienstwo_remisu'], 
                      p['prawdopodobienstwo_wygranej_gosci'])
        certain_predictions.append((p, max_prob))
    
    certain_predictions.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n💯 Najbardziej pewne przewidywania:")
    for i, (match, prob) in enumerate(certain_predictions[:5], 1):
        print(f"{i}. {match['mecz']} - {match['przewidywany_wynik']} ({prob:.1%})")
    
    # Najbardziej niepewne przewidywania
    uncertain_predictions = sorted(certain_predictions, key=lambda x: x[1])[:5]
    print(f"\n❓ Najbardziej niepewne przewidywania:")
    for i, (match, prob) in enumerate(uncertain_predictions, 1):
        print(f"{i}. {match['mecz']} - {match['przewidywany_wynik']} ({prob:.1%})")
    
    # Statystyki drużyn
    print(f"\n🏆 STATYSTYKI DRUŻYN:")
    
    # Przewidywane wygrane w domu
    home_team_wins = Counter()
    away_team_wins = Counter()
    
    for p in predictions:
        if p['przewidywany_wynik'] == 'Wygrana gospodarzy':
            home_team_wins[p['druzyna_domowa']] += 1
        elif p['przewidywany_wynik'] == 'Wygrana gości':
            away_team_wins[p['druzyna_goscinna']] += 1
    
    # Łączne wygrane
    total_wins = Counter()
    for team, wins in home_team_wins.items():
        total_wins[team] += wins
    for team, wins in away_team_wins.items():
        total_wins[team] += wins
    
    print(f"\n🥇 Drużyny z największą liczbą przewidywanych wygranych:")
    for i, (team, wins) in enumerate(total_wins.most_common(10), 1):
        print(f"{i:2d}. {team}: {wins} wygranych")
    
    # Średnia goli drużyn
    team_goals_for = Counter()
    team_goals_against = Counter()
    team_matches = Counter()
    
    for p in predictions:
        home_team = p['druzyna_domowa']
        away_team = p['druzyna_goscinna']
        home_goals = p['przewidywane_gole_gospodarzy']
        away_goals = p['przewidywane_gole_gosci']
        
        team_goals_for[home_team] += home_goals
        team_goals_against[home_team] += away_goals
        team_goals_for[away_team] += away_goals
        team_goals_against[away_team] += home_goals
        team_matches[home_team] += 1
        team_matches[away_team] += 1
    
    # Oblicz średnie
    team_avg_goals_for = {team: goals/team_matches[team] for team, goals in team_goals_for.items()}
    team_avg_goals_against = {team: goals/team_matches[team] for team, goals in team_goals_against.items()}
    
    print(f"\n⚽ Najskuteczniejsze drużyny w ataku (średnia goli na mecz):")
    sorted_attack = sorted(team_avg_goals_for.items(), key=lambda x: x[1], reverse=True)
    for i, (team, avg) in enumerate(sorted_attack[:10], 1):
        print(f"{i:2d}. {team}: {avg:.1f} goli/mecz")
    
    print(f"\n🛡️ Najlepsze drużyny w obronie (średnia straconych goli na mecz):")
    sorted_defense = sorted(team_avg_goals_against.items(), key=lambda x: x[1])
    for i, (team, avg) in enumerate(sorted_defense[:10], 1):
        print(f"{i:2d}. {team}: {avg:.1f} goli/mecz")

def main():
    filename = 'hbl_predictions_2024_25_20250715_065054.json'
    
    try:
        predictions = load_predictions(filename)
        analyze_predictions(predictions)
        
        print(f"\n✅ Analiza zakończona!")
        print(f"📁 Analizowano plik: {filename}")
        
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku: {filename}")
    except Exception as e:
        print(f"❌ Błąd podczas analizy: {e}")

if __name__ == "__main__":
    main()
