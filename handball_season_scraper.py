#!/usr/bin/env python3
"""
Handball Season Data Scraper
Fetches complete season data from Atrium Sports API
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from base_handball_scraper import BaseHandballScraper
from config import config

logger = logging.getLogger(__name__)

class HandballSeasonScraper(BaseHandballScraper):
    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = config.api.base_url
        super().__init__(base_url)
        self.all_fixtures = []
    
    def make_request_and_extract_fixtures(self, params: Dict[str, Any] = None) -> Optional[List[Dict]]:
        """Make API request with error handling and extract fixtures"""
        data = self.make_request(params)
        if data:
            return self.extract_fixtures(data)
        return None
    
    def explore_pagination(self) -> Dict[str, Any]:
        """Explore different pagination parameters"""
        pagination_params = [
            {'page': 1},
            {'offset': 0, 'limit': 50},
            {'round': 1},
            {'matchday': 1},
        ]

        results = {}
        for params in pagination_params:
            logger.info(f"Testing pagination with: {params}")
            fixtures = self.make_request_and_extract_fixtures(params)
            if fixtures:
                results[str(params)] = len(fixtures)
                self.add_delay()

        return results
    
    def fetch_by_rounds(self, max_rounds: int = 50) -> List[Dict]:
        """Try to fetch data by round numbers"""
        all_fixtures = []

        for round_num in range(1, max_rounds + 1):
            logger.info(f"Fetching round {round_num}")

            # Try different round parameter names
            for round_param in ['round', 'matchday', 'gameweek']:
                fixtures = self.make_request_and_extract_fixtures({round_param: round_num})
                if fixtures:
                    logger.info(f"Found {len(fixtures)} fixtures in round {round_num}")
                    all_fixtures.extend(fixtures)
                    break
                self.add_delay()
            else:
                # No data found for this round
                logger.info(f"No data found for round {round_num}")
                if round_num > 5:  # Stop if we haven't found data for several rounds
                    break

            self.add_delay()

        return all_fixtures
    
    def fetch_by_pagination(self, max_pages: int = 100) -> List[Dict]:
        """Try to fetch data using pagination"""
        all_fixtures = []

        for page in range(1, max_pages + 1):
            logger.info(f"Fetching page {page}")

            # Try different pagination parameter names
            for page_param in ['page', 'offset']:
                if page_param == 'offset':
                    params = {'offset': (page - 1) * 50, 'limit': 50}
                else:
                    params = {page_param: page}

                fixtures = self.make_request_and_extract_fixtures(params)
                if fixtures:
                    logger.info(f"Found {len(fixtures)} fixtures on page {page}")
                    all_fixtures.extend(fixtures)
                    break
                self.add_delay()
            else:
                # No data found for this page
                logger.info(f"No data found for page {page}")
                if page > 3:  # Stop if we haven't found data for several pages
                    break

            self.add_delay()

        return all_fixtures
    
    def remove_duplicates(self, fixtures: List[Dict]) -> List[Dict]:
        """Remove duplicate fixtures based on fixtureId"""
        unique_fixtures = {}
        for fixture in fixtures:
            fixture_id = fixture.get('fixtureId')
            if fixture_id and fixture_id not in unique_fixtures:
                unique_fixtures[fixture_id] = fixture
        return list(unique_fixtures.values())
    
    def scrape_data(self) -> List[Dict]:
        """Main method to fetch all season data - implements abstract method"""
        logger.info("Starting to fetch handball season data...")

        # First, get the basic data
        base_fixtures = self.make_request_and_extract_fixtures()
        if not base_fixtures:
            logger.error("Failed to fetch base data")
            return []

        logger.info(f"Base request returned {len(base_fixtures)} fixtures")
        all_fixtures = base_fixtures.copy()

        # Since we got all fixtures in the base request, we might not need pagination
        # But let's still try to explore if there are more
        logger.info("Exploring pagination options...")
        pagination_results = self.explore_pagination()
        logger.info(f"Pagination exploration results: {pagination_results}")

        # Try fetching by rounds (might give us different data)
        logger.info("Attempting to fetch by rounds...")
        round_fixtures = self.fetch_by_rounds()
        if round_fixtures:
            all_fixtures.extend(round_fixtures)

        # Try fetching by pagination (might give us different data)
        logger.info("Attempting to fetch by pagination...")
        page_fixtures = self.fetch_by_pagination()
        if page_fixtures:
            all_fixtures.extend(page_fixtures)

        # Remove duplicates based on fixtureId
        final_fixtures = self.remove_duplicates(all_fixtures)
        logger.info(f"Total unique fixtures found: {len(final_fixtures)}")

        # Process fixtures into standardized format
        processed_matches = []
        for fixture in final_fixtures:
            processed_match = self.process_fixture_data(fixture)
            processed_matches.append(processed_match)

        return processed_matches
    

def main():
    """Main execution function"""
    scraper = HandballSeasonScraper()
    
    # Use the base class run method for consistent execution
    filename = scraper.run()
    
    if filename:
        print(f"📁 Data saved to: {filename}")
    else:
        print("❌ No data was saved")

if __name__ == "__main__":
    main()
