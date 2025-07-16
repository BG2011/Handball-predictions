#!/usr/bin/env python3
"""
Base Handball Scraper
Common functionality for all handball scraping tools
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseHandballScraper:
    """Base class for handball scraping tools"""
    
    def __init__(self, base_url: str = "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures"):
        self.base_url = base_url
        self.session = requests.Session()
        self.delay = 1  # Delay between requests in seconds
        self._setup_session()
    
    def _setup_session(self):
        """Setup session with common headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site'
        })
    
    def make_request(self, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            default_params = {'locale': 'en-EN'}
            if params:
                default_params.update(params)

            logger.info(f"Making request with params: {default_params}")
            response = self.session.get(self.base_url, params=default_params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Request successful, response size: {len(str(data))}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed: {e}")
            return None
    
    def extract_fixtures(self, data: Dict) -> List[Dict]:
        """Extract fixtures from API response"""
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict):
            fixtures = data['data'].get('fixtures', [])
            if fixtures:
                logger.info(f"Found {len(fixtures)} fixtures in response")
                return fixtures
            else:
                logger.warning("No fixtures found in data.fixtures")
                return []
        else:
            logger.warning("Unexpected response structure")
            return []
    
    def extract_team_info(self, competitors: List[Dict]) -> Tuple[Dict, Dict]:
        """Extract home and away team information"""
        if len(competitors) < 2:
            logger.warning("Not enough competitors in fixture")
            return {}, {}
        
        home_team = next((c for c in competitors if c.get('isHome', False)), competitors[0])
        away_team = next((c for c in competitors if not c.get('isHome', True)), competitors[1])
        
        return home_team, away_team
    
    def extract_team_names_and_scores(self, home_team: Dict, away_team: Dict) -> Tuple[str, str, str, str]:
        """Extract team names and scores"""
        home_name = home_team.get('name', 'Unknown')
        away_name = away_team.get('name', 'Unknown')
        home_score = home_team.get('score', '')
        away_score = away_team.get('score', '')
        
        return home_name, away_name, home_score, away_score
    
    def format_match_date(self, start_time: str) -> str:
        """Format match date from ISO string"""
        if start_time:
            try:
                # Parse date and format it
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M')
            except:
                return start_time
        else:
            return "Brak daty"
    
    def process_fixture_data(self, fixture: Dict) -> Dict:
        """Process a single fixture into standardized format - can be overridden by subclasses"""
        competitors = fixture.get('competitors', [])
        home_team, away_team = self.extract_team_info(competitors)
        home_name, away_name, home_score, away_score = self.extract_team_names_and_scores(home_team, away_team)
        
        return {
            'fixture_id': fixture.get('fixtureId', ''),
            'druzyna_domowa': home_name,
            'druzyna_goscinna': away_name,
            'bramki_domowe': home_score,
            'bramki_goscinne': away_score,
            'data': self.format_match_date(fixture.get('startTimeLocal', '')),
            'status': fixture.get('status', {}).get('label', 'Unknown'),
            'round': fixture.get('round', ''),
            'venue': fixture.get('venue', ''),
            'attendance': fixture.get('attendance', 0)
        }
    
    def save_data(self, data: List[Dict], filename: str = None, metadata: Dict = None) -> str:
        """Save data to JSON file with metadata"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"handball_season_{timestamp}.json"
        
        # Default metadata
        default_metadata = {
            'total_matches': len(data),
            'scraped_at': datetime.now().isoformat(),
            'source_url': self.base_url
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        data_to_save = {
            'metadata': default_metadata,
            'matches': data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        return filename
    
    def add_delay(self):
        """Add delay between requests to be respectful to the API"""
        time.sleep(self.delay)
    
    def print_statistics(self, matches: List[Dict]):
        """Print scraping statistics"""
        if not matches:
            print("❌ No matches found")
            return
        
        print(f"\n✅ Successfully scraped {len(matches)} matches")
        
        # Date range
        dates = [match.get('data', '') for match in matches if match.get('data')]
        if dates:
            dates.sort()
            print(f"📅 Date range: {dates[0]} to {dates[-1]}")
        
        # Teams
        teams = set()
        for match in matches:
            teams.add(match.get('druzyna_domowa', ''))
            teams.add(match.get('druzyna_goscinna', ''))
        teams.discard('')  # Remove empty strings
        teams.discard('Unknown')  # Remove unknown teams
        
        if teams:
            print(f"🏆 Teams found: {len(teams)}")
    
    def handle_keyboard_interrupt(self):
        """Handle keyboard interrupt gracefully"""
        print("\n⏹️  Scraping interrupted by user")
        logger.info("Scraping interrupted by user")
    
    def handle_error(self, error: Exception):
        """Handle unexpected errors"""
        logger.error(f"Unexpected error: {error}")
        print(f"❌ Error occurred: {error}")
    
    # Abstract methods to be implemented by subclasses
    def scrape_data(self) -> List[Dict]:
        """Main scraping method - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement scrape_data method")
    
    def run(self):
        """Main execution method - can be overridden by subclasses"""
        try:
            logger.info("Starting scraping process")
            matches = self.scrape_data()
            
            if matches:
                filename = self.save_data(matches)
                self.print_statistics(matches)
                return filename
            else:
                print("❌ No matches found to save")
                return None
                
        except KeyboardInterrupt:
            self.handle_keyboard_interrupt()
            return None
        except Exception as e:
            self.handle_error(e)
            return None