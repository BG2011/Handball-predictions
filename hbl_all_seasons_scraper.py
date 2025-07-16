#!/usr/bin/env python3
"""
HBL All Seasons Scraper
Pobiera dane meczów dla wszystkich sezonów HBL (2020-2025)
Wyciąga tylko: mecz, datę i wynik
"""

import json
from datetime import datetime
from typing import List, Dict, Any
import logging

from base_handball_scraper import BaseHandballScraper
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HBLAllSeasonsScraper(BaseHandballScraper):
    def __init__(self):
        # Initialize base class with configuration
        super().__init__(config.api.base_url)
        
        # Use configuration for seasons
        self.seasons = config.seasons.seasons
    
    def fetch_season_data(self, season_id: str, season_name: str) -> List[Dict]:
        """Pobiera dane dla konkretnego sezonu"""
        params = {'seasonId': season_id}
        
        logger.info(f"Pobieranie danych dla sezonu: {season_name}")
        data = self.make_request(params)
        
        if data:
            fixtures = self.extract_fixtures(data)
            logger.info(f"Znaleziono {len(fixtures)} meczów w sezonie {season_name}")
            return fixtures
        else:
            logger.warning(f"Brak danych dla sezonu {season_name}")
            return []
    
    def extract_match_info(self, fixture: Dict, season_name: str) -> Dict:
        """Wyciąga podstawowe informacje o meczu"""
        try:
            # Use base class method to get basic match info
            match_data = self.process_fixture_data(fixture)
            
            # Formatuj mecz
            match = f"{match_data['druzyna_domowa']} vs {match_data['druzyna_goscinna']}"

            # Dla sezonu DAIKIN HBL 2024/25 wyzeruj wyniki (do predykcji)
            if season_name == "DAIKIN HBL 2024/25":
                result = "0:0"
                home_score = "0"
                away_score = "0"
                status = "To Predict"
            else:
                # Formatuj wynik normalnie dla innych sezonów
                if match_data['bramki_domowe'] and match_data['bramki_goscinne']:
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
                'sezon': season_name,
                'mecz': match,
                'data': match_data['data'],
                'wynik': result,
                'status': status,
                'runda': match_data['round'],
                'druzyna_domowa': match_data['druzyna_domowa'],
                'druzyna_goscinna': match_data['druzyna_goscinna'],
                'bramki_domowe': home_score,
                'bramki_goscinne': away_score
            }

        except Exception as e:
            logger.error(f"Błąd przetwarzania meczu: {e}")
            return None
    
    def scrape_data(self) -> List[Dict]:
        """Implementation of abstract method - returns all matches from all seasons"""
        all_matches = []
        
        for season in self.seasons:
            season_name = season['nameLocal']
            season_id = season['seasonId']

            fixtures = self.fetch_season_data(season_id, season_name)
            
            for fixture in fixtures:
                match_info = self.extract_match_info(fixture, season_name)
                if match_info:
                    all_matches.append(match_info)
            
            logger.info(f"Sezon {season_name}: {len([m for m in all_matches if m['sezon'] == season_name])} meczów")
            self.add_delay()

        logger.info(f"Łącznie pobrano {len(all_matches)} meczów ze wszystkich sezonów")
        return all_matches
    
    def scrape_all_seasons(self) -> Dict[str, List[Dict]]:
        """Pobiera dane dla wszystkich sezonów i zwraca je pogrupowane"""
        seasons_data = {}

        for season in self.seasons:
            season_name = season['nameLocal']
            season_id = season['seasonId']

            fixtures = self.fetch_season_data(season_id, season_name)
            season_matches = []

            for fixture in fixtures:
                match_info = self.extract_match_info(fixture, season_name)
                if match_info:
                    season_matches.append(match_info)

            seasons_data[season_name] = season_matches
            logger.info(f"Sezon {season_name}: {len(season_matches)} meczów")
            self.add_delay()

        total_matches = sum(len(matches) for matches in seasons_data.values())
        logger.info(f"Łącznie pobrano {total_matches} meczów ze wszystkich sezonów")
        return seasons_data
    
    def save_seasons_separately(self, seasons_data: Dict[str, List[Dict]]):
        """Zapisuje każdy sezon do oddzielnego pliku JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        for season_name, matches in seasons_data.items():
            if not matches:
                continue

            # Stwórz bezpieczną nazwę pliku
            safe_season_name = season_name.replace('/', '_').replace(' ', '_').replace(':', '')
            filename = f"hbl_{safe_season_name}_{timestamp}.json"

            # Use base class method with custom metadata
            metadata = {
                'season_name': season_name,
                'is_prediction_season': season_name == "DAIKIN HBL 2024/25"
            }
            
            saved_filename = self.save_data(matches, filename, metadata)
            saved_files.append(saved_filename)
            logger.info(f"Sezon {season_name} zapisany do: {saved_filename}")

        return saved_files

    def save_all_seasons_combined(self, seasons_data: Dict[str, List[Dict]]):
        """Zapisuje wszystkie sezony do jednego pliku"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hbl_all_seasons_combined_{timestamp}.json"

        all_matches = []
        for matches in seasons_data.values():
            all_matches.extend(matches)

        # Use base class save_data but with custom structure for seasons
        metadata = {
            'seasons_count': len(seasons_data),
            'seasons': list(seasons_data.keys())
        }
        
        # Save with custom structure including seasons breakdown
        data_to_save = {
            'metadata': {
                'total_matches': len(all_matches),
                'scraped_at': datetime.now().isoformat(),
                'source_url': self.base_url,
                'seasons_count': len(seasons_data),
                'seasons': list(seasons_data.keys())
            },
            'seasons': seasons_data
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f"Wszystkie sezony zapisane do: {filename}")
        return filename

def main():
    """Główna funkcja wykonawcza"""
    scraper = HBLAllSeasonsScraper()

    try:
        print("🏐 Rozpoczynam pobieranie danych HBL dla wszystkich sezonów...")
        print("=" * 60)

        seasons_data = scraper.scrape_all_seasons()

        if seasons_data:
            # Zapisz każdy sezon oddzielnie
            separate_files = scraper.save_seasons_separately(seasons_data)

            # Zapisz wszystkie sezony razem
            combined_file = scraper.save_all_seasons_combined(seasons_data)

            total_matches = sum(len(matches) for matches in seasons_data.values())

            print(f"\n✅ Pomyślnie pobrano dane:")
            print(f"📊 Łączna liczba meczów: {total_matches}")
            print(f"📁 Plik zbiorczy: {combined_file}")
            print(f"📁 Pliki oddzielne:")
            for file in separate_files:
                print(f"   - {file}")

            print(f"\n📈 Statystyki sezonów:")
            for season_name, matches in seasons_data.items():
                prediction_note = " (wyniki wyzerowane do predykcji)" if season_name == "DAIKIN HBL 2024/25" else ""
                print(f"   {season_name}: {len(matches)} meczów{prediction_note}")

            # Pokaż przykładowe mecze z każdego sezonu
            print(f"\n🎯 Przykładowe mecze z każdego sezonu:")
            for season_name, matches in seasons_data.items():
                if matches:
                    match = matches[0]
                    print(f"   {season_name}: {match['mecz']} - {match['wynik']} ({match['data']})")

        else:
            print("❌ Nie udało się pobrać żadnych danych")

    except KeyboardInterrupt:
        scraper.handle_keyboard_interrupt()
    except Exception as e:
        scraper.handle_error(e)

if __name__ == "__main__":
    main()
