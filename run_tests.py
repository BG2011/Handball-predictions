#!/usr/bin/env python3
"""
Test Runner for Handball Project
Runs all tests and provides summary report
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Run all tests and provide summary report"""
    
    print("🏐 HANDBALL PROJECT TEST SUITE")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    test_modules = [
        'test_handball_predictor',
        'test_handball_scraper'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✅ Loaded tests from {module_name}")
        except ImportError as e:
            print(f"❌ Failed to load {module_name}: {e}")
    
    # Run tests
    print(f"\n🚀 Running {suite.countTestCases()} tests...\n")
    
    # Capture output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        descriptions=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print results
    print(stream.getvalue())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"🧪 Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = "Unknown error"
            for line in reversed(error_lines):
                if line.strip() and not line.startswith('  '):
                    error_msg = line.strip()
                    break
            print(f"   - {test}: {error_msg}")
    
    # Success rate
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success rate: {success_rate:.1f}%")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_name):
    """Run a specific test class or method"""
    
    print(f"🏐 Running specific test: {test_name}")
    print("=" * 50)
    
    # Create test suite for specific test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    # Run the test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        return run_specific_test(test_name)
    else:
        # Run all tests
        return run_tests()

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)