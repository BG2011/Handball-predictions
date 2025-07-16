#!/usr/bin/env python3
"""
Fast Test Runner for Handball Project
Runs only unit tests that don't require network access
"""

import unittest
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_fast_tests():
    """Run fast tests only (no network dependencies)"""
    
    print("🏐 HANDBALL PROJECT - FAST TEST SUITE")
    print("=" * 50)
    
    # Import test classes
    from test_handball_predictor import TestHandballConfig, TestBaseHandballPredictor, TestHandballPredictor
    from test_handball_scraper import TestHandballSeasonScraper
    
    # Create test suite with only fast tests
    suite = unittest.TestSuite()
    
    # Configuration tests
    suite.addTest(TestHandballConfig('test_default_config'))
    suite.addTest(TestHandballConfig('test_config_validation'))
    suite.addTest(TestHandballConfig('test_season_methods'))
    
    # Base predictor tests
    suite.addTest(TestBaseHandballPredictor('test_initialization'))
    suite.addTest(TestBaseHandballPredictor('test_calculate_basic_team_stats'))
    suite.addTest(TestBaseHandballPredictor('test_get_team_features'))
    suite.addTest(TestBaseHandballPredictor('test_prepare_target_variables'))
    suite.addTest(TestBaseHandballPredictor('test_fit_teams_encoder'))
    suite.addTest(TestBaseHandballPredictor('test_save_predictions'))
    
    # Handball predictor tests
    suite.addTest(TestHandballPredictor('test_initialization'))
    suite.addTest(TestHandballPredictor('test_calculate_team_stats'))
    
    # Scraper tests (fast ones only)
    suite.addTest(TestHandballSeasonScraper('test_initialization'))
    suite.addTest(TestHandballSeasonScraper('test_remove_duplicates'))
    suite.addTest(TestHandballSeasonScraper('test_make_request_failure'))
    
    print(f"🚀 Running {suite.countTestCases()} fast tests...\n")
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    end_time = time.time()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 FAST TEST SUMMARY")
    print("=" * 50)
    
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"🧪 Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\\n')
            error_msg = "Unknown error"
            for line in reversed(error_lines):
                if line.strip() and not line.startswith('  '):
                    error_msg = line.strip()
                    break
            print(f"   - {test}: {error_msg}")
    
    # Success rate
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success rate: {success_rate:.1f}%")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_fast_tests()
    sys.exit(exit_code)