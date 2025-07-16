#!/usr/bin/env python3
"""
DAIKIN HBL 2024/25 Full Season Scraper
Pobiera pełne dane z prawdziwymi wynikami dla sezonu DAIKIN HBL 2024/25
(bez zerowania wyników - prawdziwe dane do analizy i walidacji predykcji)
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from base_handball_scraper import BaseHandballScraper
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DaikinSeasonFullScraper(BaseHandballScraper):
    def __init__(self):
        # Initialize base class with configuration
        super().__init__(config.api.base_url)
        
        # Get current season from configuration
        current_season = config.seasons.get_current_season()
        self.season_id = current_season['seasonId']
        self.season_name = current_season['nameLocal']
    
    def fetch_season_data(self) -> Optional[List[Dict]]:
        """Pobiera pełne dane dla sezonu DAIKIN HBL 2024/25"""
        params = {'seasonId': self.season_id}
        
        logger.info(f"Pobieranie pełnych danych dla sezonu: {self.season_name}")
        data = self.make_request(params)
        
        if data:
            fixtures = self.extract_fixtures(data)
            logger.info(f"Znaleziono {len(fixtures)} meczów w sezonie {self.season_name}")
            return fixtures
        else:
            logger.warning(f"Brak danych dla sezonu {self.season_name}")
            return None
    
    def extract_match_info(self, fixture: Dict) -> Optional[Dict]:
        """Wyciąga informacje o meczu - BEZ ZEROWANIA WYNIKÓW"""
        try:
            # Use base class method to get basic match info
            match_data = self.process_fixture_data(fixture)
            
            # Formatuj mecz
            match = f"{match_data['druzyna_domowa']} vs {match_data['druzyna_goscinna']}"
            
            # TUTAJ NIE ZERUJEMY WYNIKÓW - pobieramy prawdziwe dane
            if match_data['bramki_domowe'] and match_data['bramki_goscinne'] and match_data['bramki_domowe'] != '' and match_data['bramki_goscinne'] != '':
                result = f"{match_data['bramki_domowe']}:{match_data['bramki_goscinne']}"
                status = match_data['status']
                home_score = match_data['bramki_domowe']
                away_score = match_data['bramki_goscinne']
            else:
                result = "Nie rozegrany"
                status = match_data['status']
                home_score = ""
                away_score = ""
            
            return {
                'sezon': self.season_name,
                'mecz': match,
                'data': match_data['data'],
                'wynik': result,
                'status': status,
                'runda': match_data['round'],
                'druzyna_domowa': match_data['druzyna_domowa'],
                'druzyna_goscinna': match_data['druzyna_goscinna'],
                'bramki_domowe': home_score,
                'bramki_goscinne': away_score,
                'fixture_id': match_data['fixture_id'],
                'venue_id': fixture.get('venueId', ''),
                'attendance': match_data['attendance'],
                'timezone': fixture.get('timezone', ''),
                'start_time_utc': fixture.get('startTimeUTC', ''),
                'round_code': fixture.get('roundCode', ''),
                'stage_code': fixture.get('stageCode', '')
            }
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania meczu: {e}")
            return None
    
    def scrape_data(self) -> List[Dict]:
        """Implementation of abstract method - returns all matches from current season"""
        fixtures = self.fetch_season_data()
        
        if not fixtures:
            logger.error("Nie udało się pobrać danych sezonu")
            return []
        
        processed_matches = []
        
        for fixture in fixtures:
            match_info = self.extract_match_info(fixture)
            if match_info:
                processed_matches.append(match_info)
        
        logger.info(f"Przetworzono {len(processed_matches)} meczów")
        return processed_matches
    
    def process_season_data(self) -> List[Dict]:
        """Przetwarza dane sezonu"""
        return self.scrape_data()
    
    def save_full_season_data(self, matches: List[Dict], filename: str = None):
        """Zapisuje pełne dane sezonu do pliku JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"daikin_hbl_2024_25_FULL_{timestamp}.json"
        
        # Statystyki
        played_matches = [m for m in matches if m['status'] == 'Final' and m['wynik'] != 'Nie rozegrany']
        scheduled_matches = [m for m in matches if m['status'] != 'Final' or m['wynik'] == 'Nie rozegrany']
        
        # Use base class method with custom metadata
        metadata = {
            'season_name': self.season_name,
            'season_id': self.season_id,
            'played_matches': len(played_matches),
            'scheduled_matches': len(scheduled_matches),
            'data_type': 'FULL_REAL_RESULTS',
            'note': 'Pełne dane z prawdziwymi wynikami - NIE wyzerowane'
        }
        
        saved_filename = self.save_data(matches, filename, metadata)
        logger.info(f"Pełne dane sezonu zapisane do: {saved_filename}")
        return saved_filename
    
    def analyze_season_data(self, matches: List[Dict]):
        """Analizuje dane sezonu"""
        played_matches = [m for m in matches if m['status'] == 'Final' and m['wynik'] != 'Nie rozegrany']
        scheduled_matches = [m for m in matches if m['status'] != 'Final' or m['wynik'] == 'Nie rozegrany']
        
        print(f"\n📊 ANALIZA SEZONU {self.season_name}:")
        print(f"   Łączna liczba meczów: {len(matches)}")
        print(f"   Mecze rozegrane: {len(played_matches)}")
        print(f"   Mecze zaplanowane: {len(scheduled_matches)}")
        
        if played_matches:
            # Statystyki goli
            total_goals = []
            home_goals = []
            away_goals = []
            home_wins = 0
            away_wins = 0
            draws = 0
            
            for match in played_matches:
                if match['bramki_domowe'] and match['bramki_goscinne']:
                    try:
                        h_goals = int(match['bramki_domowe'])
                        a_goals = int(match['bramki_goscinne'])
                        
                        home_goals.append(h_goals)
                        away_goals.append(a_goals)
                        total_goals.append(h_goals + a_goals)
                        
                        if h_goals > a_goals:
                            home_wins += 1
                        elif h_goals < a_goals:
                            away_wins += 1
                        else:
                            draws += 1
                    except ValueError:
                        continue
            
            if total_goals:
                print(f"\n⚽ STATYSTYKI GOLI (z {len(total_goals)} rozegranych meczów):")
                print(f"   Średnia goli na mecz: {sum(total_goals)/len(total_goals):.1f}")
                print(f"   Średnia goli gospodarzy: {sum(home_goals)/len(home_goals):.1f}")
                print(f"   Średnia goli gości: {sum(away_goals)/len(away_goals):.1f}")
                print(f"   Najwyższy wynik: {max(total_goals)} goli")
                print(f"   Najniższy wynik: {min(total_goals)} goli")
                
                print(f"\n🏆 WYNIKI:")
                print(f"   Wygrane gospodarzy: {home_wins} ({home_wins/len(total_goals):.1%})")
                print(f"   Remisy: {draws} ({draws/len(total_goals):.1%})")
                print(f"   Wygrane gości: {away_wins} ({away_wins/len(total_goals):.1%})")
        
        # Pokaż przykładowe mecze
        if played_matches:
            print(f"\n🎯 Przykładowe rozegrane mecze:")
            for i, match in enumerate(played_matches[:5]):
                print(f"   {match['mecz']} - {match['wynik']} ({match['data']})")
        
        if scheduled_matches:
            print(f"\n📅 Przykładowe zaplanowane mecze:")
            for i, match in enumerate(scheduled_matches[:5]):
                print(f"   {match['mecz']} - {match['status']} ({match['data']})")

def main():
    """Główna funkcja wykonawcza"""
    scraper = DaikinSeasonFullScraper()
    
    try:
        print("🏐 DAIKIN HBL 2024/25 - Pobieranie pełnych danych sezonu")
        print("=" * 60)
        print("⚠️  UWAGA: Ten skrypt pobiera PRAWDZIWE wyniki (nie wyzerowane)")
        print()
        
        # Pobierz i przetwórz dane
        matches = scraper.process_season_data()
        
        if matches:
            # Zapisz dane
            filename = scraper.save_full_season_data(matches)
            
            # Analizuj dane
            scraper.analyze_season_data(matches)
            
            print(f"\n✅ Pomyślnie pobrano pełne dane sezonu!")
            print(f"📁 Dane zapisane w: {filename}")
            
        else:
            print("❌ Nie udało się pobrać danych sezonu")
            
    except KeyboardInterrupt:
        scraper.handle_keyboard_interrupt()
    except Exception as e:
        scraper.handle_error(e)

if __name__ == "__main__":
    main()
