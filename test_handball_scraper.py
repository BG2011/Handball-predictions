#!/usr/bin/env python3
"""
Unit Tests for Handball Scraper System
Tests for scraper functionality and data processing
"""

import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
import requests

# Import our modules
from base_handball_scraper import BaseHandballScraper
from handball_season_scraper import HandballSeasonScraper
from config import config

class TestHandballSeasonScraper(unittest.TestCase):
    """Test the season scraper implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scraper = HandballSeasonScraper()
        
        # Sample API response for testing
        self.sample_response = {
            'data': {
                'fixtures': [
                    {
                        'fixtureId': 'test-fixture-1',
                        'startTimeLocal': '2024-01-01T15:00:00',
                        'status': {'label': 'Final'},
                        'competitors': [
                            {'name': 'Team A', 'score': '25', 'isHome': True},
                            {'name': 'Team B', 'score': '23', 'isHome': False}
                        ],
                        'round': 'Round: 1',
                        'venue': 'Arena A',
                        'attendance': 5000
                    },
                    {
                        'fixtureId': 'test-fixture-2',
                        'startTimeLocal': '2024-01-02T16:00:00',
                        'status': {'label': 'Final'},
                        'competitors': [
                            {'name': 'Team C', 'score': '28', 'isHome': True},
                            {'name': 'Team D', 'score': '30', 'isHome': False}
                        ],
                        'round': 'Round: 1',
                        'venue': 'Arena B',
                        'attendance': 4500
                    }
                ]
            }
        }
    
    def test_initialization(self):
        """Test scraper initialization"""
        self.assertIsInstance(self.scraper, BaseHandballScraper)
        self.assertEqual(self.scraper.base_url, config.api.base_url)
        self.assertEqual(self.scraper.delay, config.api.delay_between_requests)
        self.assertIsNotNone(self.scraper.session)
        self.assertEqual(self.scraper.all_fixtures, [])
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_make_request_and_extract_fixtures(self, mock_get):
        """Test making API request and extracting fixtures"""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        fixtures = self.scraper.make_request_and_extract_fixtures({'page': 1})
        
        self.assertEqual(len(fixtures), 2)
        self.assertEqual(fixtures[0]['fixtureId'], 'test-fixture-1')
        self.assertEqual(fixtures[1]['fixtureId'], 'test-fixture-2')
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_make_request_failure(self, mock_get):
        """Test API request failure handling"""
        # Mock failed response
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        fixtures = self.scraper.make_request_and_extract_fixtures({'page': 1})
        
        self.assertIsNone(fixtures)
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_explore_pagination(self, mock_get):
        """Test pagination exploration"""
        # Mock response for pagination
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test pagination exploration
        pagination_fixtures = self.scraper.explore_pagination()
        
        # Should have tried multiple pagination methods
        self.assertGreaterEqual(len(pagination_fixtures), 0)
        
        # Check that it attempted different pagination parameters
        call_args_list = mock_get.call_args_list
        self.assertGreater(len(call_args_list), 1)
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_fetch_by_rounds(self, mock_get):
        """Test fetching data by rounds"""
        # Mock response for rounds
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test round-based fetching
        round_fixtures = self.scraper.fetch_by_rounds()
        
        # Should have tried multiple rounds
        self.assertGreaterEqual(len(round_fixtures), 0)
        
        # Check that it attempted different round parameters
        call_args_list = mock_get.call_args_list
        self.assertGreater(len(call_args_list), 1)
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_fetch_by_pagination(self, mock_get):
        """Test fetching data by pagination"""
        # Mock response for pagination
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test page-based fetching
        page_fixtures = self.scraper.fetch_by_pagination()
        
        # Should have tried multiple pages
        self.assertGreaterEqual(len(page_fixtures), 0)
        
        # Check that it attempted different page parameters
        call_args_list = mock_get.call_args_list
        self.assertGreater(len(call_args_list), 1)
    
    def test_remove_duplicates(self):
        """Test duplicate fixture removal"""
        # Create test fixtures with duplicates
        fixtures = [
            {'fixtureId': 'fixture-1', 'name': 'Match 1'},
            {'fixtureId': 'fixture-2', 'name': 'Match 2'},
            {'fixtureId': 'fixture-1', 'name': 'Match 1 Duplicate'},
            {'fixtureId': 'fixture-3', 'name': 'Match 3'}
        ]
        
        unique_fixtures = self.scraper.remove_duplicates(fixtures)
        
        self.assertEqual(len(unique_fixtures), 3)
        fixture_ids = [f['fixtureId'] for f in unique_fixtures]
        self.assertEqual(set(fixture_ids), {'fixture-1', 'fixture-2', 'fixture-3'})
    
    @patch('handball_season_scraper.HandballSeasonScraper.make_request_and_extract_fixtures')
    def test_scrape_data(self, mock_make_request):
        """Test the main scrape_data method"""
        # Mock the request method to return sample fixtures
        mock_make_request.return_value = [
            {
                'fixtureId': 'test-fixture-1',
                'startTimeLocal': '2024-01-01T15:00:00',
                'status': {'label': 'Final'},
                'competitors': [
                    {'name': 'Team A', 'score': '25', 'isHome': True},
                    {'name': 'Team B', 'score': '23', 'isHome': False}
                ],
                'round': 'Round: 1',
                'venue': 'Arena A',
                'attendance': 5000
            }
        ]
        
        matches = self.scraper.scrape_data()
        
        # Should have processed the fixture into match format
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['druzyna_domowa'], 'Team A')
        self.assertEqual(matches[0]['druzyna_goscinna'], 'Team B')
        self.assertEqual(matches[0]['bramki_domowe'], '25')
        self.assertEqual(matches[0]['bramki_goscinne'], '23')
    
    def test_save_data_integration(self):
        """Test data saving integration"""
        # Create sample match data
        sample_matches = [
            {
                'druzyna_domowa': 'Team A',
                'druzyna_goscinna': 'Team B',
                'bramki_domowe': '25',
                'bramki_goscinne': '23',
                'status': 'Final',
                'data': '2024-01-01 15:00'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            result_file = self.scraper.save_data(sample_matches, filename)
            
            # Check that file was created
            self.assertTrue(os.path.exists(result_file))
            
            # Check file contents
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('metadata', data)
            self.assertIn('matches', data)
            self.assertEqual(len(data['matches']), 1)
            self.assertEqual(data['metadata']['total_matches'], 1)
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

class TestScraperErrorHandling(unittest.TestCase):
    """Test error handling in scrapers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scraper = HandballSeasonScraper()
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_request_timeout_handling(self, mock_get):
        """Test handling of request timeouts"""
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        fixtures = self.scraper.make_request_and_extract_fixtures({'page': 1})
        
        self.assertIsNone(fixtures)
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_connection_error_handling(self, mock_get):
        """Test handling of connection errors"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        fixtures = self.scraper.make_request_and_extract_fixtures({'page': 1})
        
        self.assertIsNone(fixtures)
    
    @patch('base_handball_scraper.requests.Session.get')
    def test_json_decode_error_handling(self, mock_get):
        """Test handling of JSON decode errors"""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        fixtures = self.scraper.make_request_and_extract_fixtures({'page': 1})
        
        self.assertIsNone(fixtures)
    
    def test_malformed_competitor_data(self):
        """Test handling of malformed competitor data"""
        # Test with no competitors
        fixture_no_competitors = {
            'fixtureId': 'test-fixture-1',
            'competitors': []
        }
        
        processed = self.scraper.process_fixture_data(fixture_no_competitors)
        
        self.assertEqual(processed['druzyna_domowa'], 'Unknown')
        self.assertEqual(processed['druzyna_goscinna'], 'Unknown')
        
        # Test with missing competitor data
        fixture_missing_data = {
            'fixtureId': 'test-fixture-2',
            'competitors': [
                {'name': 'Team A'},  # Missing score and isHome
                {'name': 'Team B'}   # Missing score and isHome
            ]
        }
        
        processed = self.scraper.process_fixture_data(fixture_missing_data)
        
        self.assertEqual(processed['bramki_domowe'], '')
        self.assertEqual(processed['bramki_goscinne'], '')

class TestScraperPerformance(unittest.TestCase):
    """Test performance aspects of scrapers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scraper = HandballSeasonScraper()
    
    def test_delay_functionality(self):
        """Test that delay functionality works"""
        import time
        
        start_time = time.time()
        self.scraper.add_delay()
        end_time = time.time()
        
        # Should have waited at least the configured delay
        self.assertGreaterEqual(end_time - start_time, self.scraper.delay - 0.1)
    
    def test_duplicate_removal_performance(self):
        """Test performance of duplicate removal with large dataset"""
        # Create a large number of fixtures with some duplicates
        fixtures = []
        for i in range(1000):
            fixtures.append({
                'fixtureId': f'fixture-{i % 100}',  # This creates duplicates
                'name': f'Match {i}'
            })
        
        start_time = time.time()
        unique_fixtures = self.scraper.remove_duplicates(fixtures)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        self.assertLess(end_time - start_time, 1.0)
        
        # Should have correct number of unique fixtures
        self.assertEqual(len(unique_fixtures), 100)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)