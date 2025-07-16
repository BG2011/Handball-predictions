#!/usr/bin/env python3
"""
Examine the actual API response structure
"""

import requests
import json

def examine_api_response():
    url = "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site'
    }
    
    params = {'locale': 'en-EN'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Save full response for examination
            with open('full_response.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Response saved to full_response.json")
            print(f"Response size: {len(response.content)} bytes")
            print(f"Top-level keys: {list(data.keys())}")
            
            # Examine each top-level key
            for key, value in data.items():
                print(f"\n=== {key.upper()} ===")
                if isinstance(value, dict):
                    print(f"Type: dict with {len(value)} keys")
                    print(f"Keys: {list(value.keys())}")
                    
                    # Look for fixtures-like data
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list) and len(subvalue) > 0:
                            print(f"  {subkey}: list with {len(subvalue)} items")
                            if isinstance(subvalue[0], dict):
                                print(f"    First item keys: {list(subvalue[0].keys())}")
                                # Check if this looks like fixture data
                                first_item = subvalue[0]
                                if any(keyword in str(first_item).lower() for keyword in ['fixture', 'match', 'game', 'team', 'score']):
                                    print(f"    *** This might be fixture data! ***")
                        elif isinstance(subvalue, list):
                            print(f"  {subkey}: empty list")
                        else:
                            print(f"  {subkey}: {type(subvalue).__name__}")
                            
                elif isinstance(value, list):
                    print(f"Type: list with {len(value)} items")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"First item keys: {list(value[0].keys())}")
                else:
                    print(f"Type: {type(value).__name__}")
                    print(f"Value: {str(value)[:200]}...")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    examine_api_response()
