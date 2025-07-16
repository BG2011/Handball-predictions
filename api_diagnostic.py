#!/usr/bin/env python3
"""
API Diagnostic Script
Tests the handball API with different parameters and headers
"""

import requests
import json
from datetime import datetime

def test_api_endpoint():
    base_url = "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures"
    
    # Different header combinations to test
    header_sets = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site'
        },
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-EN,en;q=0.9'
        },
        {
            'User-Agent': 'Python-requests/2.25.1'
        }
    ]
    
    # Different parameter combinations to test
    param_sets = [
        {'locale': 'en-EN'},
        {'locale': 'en-US'},
        {'locale': 'en'},
        {},
        {'locale': 'en-EN', 'season': '2024'},
        {'locale': 'en-EN', 'season': '2025'},
        {'locale': 'en-EN', 'limit': 100},
        {'locale': 'en-EN', 'page': 1},
        {'locale': 'en-EN', 'offset': 0},
    ]
    
    print("🔍 Testing API endpoint with different configurations...\n")
    
    for i, headers in enumerate(header_sets):
        print(f"📋 Header Set {i+1}:")
        for key, value in headers.items():
            print(f"   {key}: {value}")
        print()
        
        for j, params in enumerate(param_sets):
            print(f"   🧪 Test {j+1} - Params: {params}")
            
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                
                print(f"      Status Code: {response.status_code}")
                print(f"      Response Headers: {dict(response.headers)}")
                print(f"      Content Length: {len(response.content)}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"      JSON Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        
                        if isinstance(data, dict) and 'fixtures' in data:
                            fixtures_count = len(data['fixtures']) if data['fixtures'] else 0
                            print(f"      Fixtures Count: {fixtures_count}")
                            
                            if fixtures_count > 0:
                                print(f"      ✅ SUCCESS! Found {fixtures_count} fixtures")
                                print(f"      Sample fixture keys: {list(data['fixtures'][0].keys())}")
                                
                                # Save successful response for analysis
                                with open(f'successful_response_{i}_{j}.json', 'w') as f:
                                    json.dump(data, f, indent=2)
                                print(f"      💾 Saved to successful_response_{i}_{j}.json")
                                return True
                        else:
                            print(f"      ⚠️  No fixtures in response")
                            
                    except json.JSONDecodeError as e:
                        print(f"      ❌ JSON decode error: {e}")
                        print(f"      Raw content (first 200 chars): {response.text[:200]}")
                else:
                    print(f"      ❌ HTTP Error: {response.status_code}")
                    print(f"      Response: {response.text[:200]}")
                    
            except requests.exceptions.RequestException as e:
                print(f"      ❌ Request failed: {e}")
            
            print()
        
        print("-" * 80)
        print()
    
    return False

def test_alternative_endpoints():
    """Test some alternative endpoint patterns"""
    print("🔍 Testing alternative endpoint patterns...\n")
    
    base_patterns = [
        "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures",
        "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/matches",
        "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/games",
        "https://eapi.web.prod.cloud.atriumsports.com/v1/fixtures/248",
        "https://eapi.web.prod.cloud.atriumsports.com/v2/embed/248/fixtures",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-EN,en;q=0.9'
    }
    
    for url in base_patterns:
        print(f"Testing: {url}")
        try:
            response = requests.get(url, params={'locale': 'en-EN'}, headers=headers, timeout=10)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and 'fixtures' in data and data['fixtures']:
                        print(f"  ✅ SUCCESS! Found {len(data['fixtures'])} fixtures")
                        return url
                except:
                    pass
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()
    
    return None

if __name__ == "__main__":
    print("🏐 Handball API Diagnostic Tool")
    print("=" * 50)
    print()
    
    # Test main endpoint
    success = test_api_endpoint()
    
    if not success:
        print("🔄 Main endpoint tests failed. Trying alternatives...")
        alternative = test_alternative_endpoints()
        
        if alternative:
            print(f"✅ Found working alternative: {alternative}")
        else:
            print("❌ No working endpoints found")
    
    print("\n🏁 Diagnostic complete!")
