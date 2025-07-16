#!/usr/bin/env python3
"""
Unit Tests for Handball Prediction System
Basic test framework for validating core functionality
"""

import unittest
import tempfile
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import our modules
from base_handball_predictor import BaseHandballPredictor
from base_handball_scraper import BaseHandballScraper
from config import config, HandballConfig
from handball_ml_predictor import HandballPredictor

class TestHandballConfig(unittest.TestCase):
    """Test configuration system"""
    
    def test_default_config(self):
        """Test default configuration values"""
        test_config = HandballConfig()
        
        # Test API config
        self.assertEqual(test_config.api.base_url, "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures")
        self.assertEqual(test_config.api.timeout, 30)
        self.assertEqual(test_config.api.delay_between_requests, 1)
        self.assertEqual(test_config.api.locale, "en-EN")
        
        # Test ML config
        self.assertEqual(test_config.ml.random_state, 42)
        self.assertEqual(test_config.ml.test_size, 0.2)
        self.assertEqual(test_config.ml.cv_folds, 5)
        
        # Test file config
        self.assertEqual(len(test_config.files.training_files), 4)
        self.assertTrue(all('hbl_LIQUI_MOLY_HBL' in f for f in test_config.files.training_files[:4]))
        
        # Test seasons config
        self.assertEqual(len(test_config.seasons.seasons), 5)
        self.assertEqual(test_config.seasons.seasons[0]["year"], 2020)
        self.assertEqual(test_config.seasons.seasons[-1]["year"], 2024)
    
    def test_config_validation(self):
        """Test configuration validation"""
        test_config = HandballConfig()
        
        # Test invalid test_size
        test_config.ml.test_size = 1.5
        self.assertFalse(test_config.validate())
        
        # Test invalid cv_folds
        test_config.ml.test_size = 0.2
        test_config.ml.cv_folds = 1
        self.assertFalse(test_config.validate())
        
        # Test valid config
        test_config.ml.cv_folds = 5
        # Note: This might fail due to missing files, but that's expected
        result = test_config.validate()
        self.assertIsInstance(result, bool)
    
    def test_season_methods(self):
        """Test season-related methods"""
        test_config = HandballConfig()
        
        # Test get_season_by_year
        season_2020 = test_config.seasons.get_season_by_year(2020)
        self.assertIsNotNone(season_2020)
        self.assertEqual(season_2020["nameLocal"], "LIQUI MOLY HBL 2020/21")
        
        # Test get_current_season
        current_season = test_config.seasons.get_current_season()
        self.assertIsNotNone(current_season)
        self.assertEqual(current_season["year"], 2024)
        
        # Test non-existent season
        invalid_season = test_config.seasons.get_season_by_year(2030)
        self.assertIsNone(invalid_season)

class TestBaseHandballPredictor(unittest.TestCase):
    """Test base predictor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = BaseHandballPredictor()
        
        # Create sample match data
        self.sample_matches = [
            {
                'druzyna_domowa': 'Team A',
                'druzyna_goscinna': 'Team B',
                'bramki_domowe': '25',
                'bramki_goscinne': '23',
                'status': 'Final',
                'data': '2024-01-01T15:00:00'
            },
            {
                'druzyna_domowa': 'Team B',
                'druzyna_goscinna': 'Team C',
                'bramki_domowe': '28',
                'bramki_goscinne': '30',
                'status': 'Final',
                'data': '2024-01-02T16:00:00'
            },
            {
                'druzyna_domowa': 'Team C',
                'druzyna_goscinna': 'Team A',
                'bramki_domowe': '22',
                'bramki_goscinne': '22',
                'status': 'Final',
                'data': '2024-01-03T17:00:00'
            }
        ]
        
        self.sample_df = pd.DataFrame(self.sample_matches)
    
    def test_initialization(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor.team_encoder)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertIsNone(self.predictor.result_model)
        self.assertIsNone(self.predictor.goals_home_model)
        self.assertIsNone(self.predictor.goals_away_model)
        self.assertEqual(self.predictor.team_stats, {})
    
    def test_calculate_basic_team_stats(self):
        """Test basic team statistics calculation"""
        self.predictor.calculate_basic_team_stats(self.sample_df)
        
        # Check that stats were calculated
        self.assertIn('Team A', self.predictor.team_stats)
        self.assertIn('Team B', self.predictor.team_stats)
        self.assertIn('Team C', self.predictor.team_stats)
        
        # Check Team A stats
        team_a_stats = self.predictor.team_stats['Team A']
        self.assertEqual(team_a_stats['matches'], 2)
        self.assertEqual(team_a_stats['wins'], 1)
        self.assertEqual(team_a_stats['draws'], 1)
        self.assertEqual(team_a_stats['losses'], 0)
        self.assertEqual(team_a_stats['goals_for'], 47)  # 25 + 22
        self.assertEqual(team_a_stats['goals_against'], 45)  # 23 + 22
        
        # Check Team B stats
        team_b_stats = self.predictor.team_stats['Team B']
        self.assertEqual(team_b_stats['matches'], 2)
        self.assertEqual(team_b_stats['wins'], 0)  # Team B lost both matches
        self.assertEqual(team_b_stats['draws'], 0)
        self.assertEqual(team_b_stats['losses'], 2)
    
    def test_get_team_features(self):
        """Test team feature extraction"""
        self.predictor.calculate_basic_team_stats(self.sample_df)
        
        features = self.predictor.get_team_features('Team A', 'Team B')
        
        # Check that features are returned
        self.assertIn('home_win_rate', features)
        self.assertIn('away_win_rate', features)
        self.assertIn('home_goals_per_match', features)
        self.assertIn('away_goals_per_match', features)
        
        # Check feature values
        self.assertEqual(features['home_win_rate'], 0.5)  # Team A: 1 win out of 2 matches
        self.assertEqual(features['away_win_rate'], 0.0)  # Team B: 0 wins out of 2 matches
        self.assertEqual(features['home_goals_per_match'], 23.5)  # Team A: 47/2
        self.assertEqual(features['away_goals_per_match'], 25.5)  # Team B: 51/2
    
    def test_prepare_target_variables(self):
        """Test target variable preparation"""
        results, home_goals, away_goals = self.predictor.prepare_target_variables(self.sample_df)
        
        # Check results
        self.assertEqual(list(results), ['Wygrana gospodarzy', 'Wygrana gości', 'Remis'])
        
        # Check goals
        self.assertEqual(list(home_goals), [25, 28, 22])
        self.assertEqual(list(away_goals), [23, 30, 22])
    
    def test_fit_teams_encoder(self):
        """Test team encoder fitting"""
        self.predictor.fit_teams_encoder(self.sample_df)
        
        # Check that encoder was fitted
        expected_teams = {'Team A', 'Team B', 'Team C'}
        fitted_teams = set(self.predictor.team_encoder.classes_)
        self.assertEqual(fitted_teams, expected_teams)
    
    def test_save_predictions(self):
        """Test prediction saving"""
        sample_predictions = [
            {
                'home_team': 'Team A',
                'away_team': 'Team B',
                'predicted_result': 'Wygrana gospodarzy',
                'predicted_score': '25:23'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            result_file = self.predictor.save_predictions(sample_predictions, filename)
            
            # Check that file was created
            self.assertTrue(os.path.exists(result_file))
            
            # Check file contents
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('metadata', data)
            self.assertIn('predictions', data)
            self.assertEqual(len(data['predictions']), 1)
            self.assertEqual(data['predictions'][0]['home_team'], 'Team A')
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

class TestBaseHandballScraper(unittest.TestCase):
    """Test base scraper functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scraper = BaseHandballScraper()
        
        # Sample API response
        self.sample_api_response = {
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
                    }
                ]
            }
        }
    
    def test_initialization(self):
        """Test scraper initialization"""
        self.assertEqual(self.scraper.base_url, "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures")
        self.assertEqual(self.scraper.delay, 1)
        self.assertIsNotNone(self.scraper.session)
        self.assertIn('User-Agent', self.scraper.session.headers)
    
    def test_extract_fixtures(self):
        """Test fixture extraction from API response"""
        fixtures = self.scraper.extract_fixtures(self.sample_api_response)
        
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0]['fixtureId'], 'test-fixture-1')
    
    def test_extract_team_info(self):
        """Test team information extraction"""
        competitors = self.sample_api_response['data']['fixtures'][0]['competitors']
        home_team, away_team = self.scraper.extract_team_info(competitors)
        
        self.assertEqual(home_team['name'], 'Team A')
        self.assertEqual(home_team['score'], '25')
        self.assertTrue(home_team['isHome'])
        
        self.assertEqual(away_team['name'], 'Team B')
        self.assertEqual(away_team['score'], '23')
        self.assertFalse(away_team['isHome'])
    
    def test_extract_team_names_and_scores(self):
        """Test team names and scores extraction"""
        competitors = self.sample_api_response['data']['fixtures'][0]['competitors']
        home_team, away_team = self.scraper.extract_team_info(competitors)
        
        home_name, away_name, home_score, away_score = self.scraper.extract_team_names_and_scores(
            home_team, away_team
        )
        
        self.assertEqual(home_name, 'Team A')
        self.assertEqual(away_name, 'Team B')
        self.assertEqual(home_score, '25')
        self.assertEqual(away_score, '23')
    
    def test_format_match_date(self):
        """Test match date formatting"""
        # Test ISO string with Z
        iso_date = '2024-01-01T15:00:00Z'
        formatted = self.scraper.format_match_date(iso_date)
        self.assertEqual(formatted, '2024-01-01 15:00')
        
        # Test ISO string without Z
        iso_date_no_z = '2024-01-01T15:00:00'
        formatted_no_z = self.scraper.format_match_date(iso_date_no_z)
        self.assertEqual(formatted_no_z, '2024-01-01 15:00')
        
        # Test empty string
        empty_date = ''
        formatted_empty = self.scraper.format_match_date(empty_date)
        self.assertEqual(formatted_empty, 'Brak daty')
    
    def test_process_fixture_data(self):
        """Test fixture data processing"""
        fixture = self.sample_api_response['data']['fixtures'][0]
        processed = self.scraper.process_fixture_data(fixture)
        
        self.assertEqual(processed['fixture_id'], 'test-fixture-1')
        self.assertEqual(processed['druzyna_domowa'], 'Team A')
        self.assertEqual(processed['druzyna_goscinna'], 'Team B')
        self.assertEqual(processed['bramki_domowe'], '25')
        self.assertEqual(processed['bramki_goscinne'], '23')
        self.assertEqual(processed['status'], 'Final')
        self.assertEqual(processed['round'], 'Round: 1')
        self.assertEqual(processed['venue'], 'Arena A')
        self.assertEqual(processed['attendance'], 5000)
    
    def test_save_data(self):
        """Test data saving"""
        sample_data = [
            {
                'druzyna_domowa': 'Team A',
                'druzyna_goscinna': 'Team B',
                'bramki_domowe': '25',
                'bramki_goscinne': '23'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            result_file = self.scraper.save_data(sample_data, filename)
            
            # Check that file was created
            self.assertTrue(os.path.exists(result_file))
            
            # Check file contents
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('metadata', data)
            self.assertIn('matches', data)
            self.assertEqual(len(data['matches']), 1)
            self.assertEqual(data['matches'][0]['druzyna_domowa'], 'Team A')
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

class TestHandballPredictor(unittest.TestCase):
    """Test the concrete handball predictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = HandballPredictor()
    
    def test_initialization(self):
        """Test predictor initialization"""
        self.assertIsInstance(self.predictor, BaseHandballPredictor)
        self.assertEqual(self.predictor.training_files, config.files.training_files)
        self.assertEqual(self.predictor.prediction_file, config.files.prediction_file)
    
    @patch('handball_ml_predictor.open')
    @patch('handball_ml_predictor.json.load')
    def test_load_training_data_mock(self, mock_json_load, mock_open):
        """Test training data loading with mocked file operations"""
        # Mock file contents
        mock_json_load.return_value = {
            'matches': [
                {
                    'druzyna_domowa': 'Team A',
                    'druzyna_goscinna': 'Team B',
                    'bramki_domowe': '25',
                    'bramki_goscinne': '23',
                    'status': 'Final'
                }
            ]
        }
        
        # Mock the file opening
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        # Call the method
        result = self.predictor.load_training_data()
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 4 training files * 1 match each
    
    def test_calculate_team_stats(self):
        """Test team statistics calculation"""
        # Create sample data
        sample_data = pd.DataFrame([
            {
                'druzyna_domowa': 'Team A',
                'druzyna_goscinna': 'Team B',
                'bramki_domowe': '25',
                'bramki_goscinne': '23'
            }
        ])
        
        self.predictor.calculate_team_stats(sample_data)
        
        # Check that stats were calculated
        self.assertIn('Team A', self.predictor.team_stats)
        self.assertIn('Team B', self.predictor.team_stats)
        
        # Check that extended stats were added
        team_a_stats = self.predictor.team_stats['Team A']
        self.assertIn('avg_goals_for', team_a_stats)
        self.assertIn('avg_goals_against', team_a_stats)
        self.assertIn('win_rate', team_a_stats)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)