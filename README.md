# Handball Season Scraper

Scripts to fetch complete handball season data from the Atrium Sports API.

## Features

- 🏐 Fetches all handball fixtures for the season
- 🔄 Handles pagination and multiple rounds automatically
- 📊 Removes duplicate fixtures
- 💾 Saves data in structured JSON format
- 🛡️ Includes error handling and rate limiting
- 📈 Provides progress feedback and statistics

## Available Scripts

### Python Version

**Requirements:**
- Python 3.6+
- requests library

**Installation:**
```bash
pip install -r requirements.txt
```

**Usage:**
```bash
python handball_season_scraper.py
```

### JavaScript/Node.js Version

**Requirements:**
- Node.js 14+

**Usage:**
```bash
node handball_season_scraper.js
```

Or using npm:
```bash
npm start
```

## Output

Both scripts will:

1. **Fetch data** from the API using multiple strategies:
   - Base request
   - Pagination exploration
   - Round-by-round fetching
   - Page-by-page fetching

2. **Save results** to a JSON file named `handball_season_YYYYMMDD_HHMMSS.json`

3. **Display statistics**:
   - Total fixtures found
   - Date range of matches
   - Number of teams

## Output Format

The generated JSON file contains:

```json
{
  "metadata": {
    "total_fixtures": 306,
    "scraped_at": "2025-01-15T10:30:00.000Z",
    "source_url": "https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures"
  },
  "fixtures": [
    {
      "fixtureId": "a3cdd193-4374-11ef-97a0-07b53bf1eaa1",
      "name": "SG BBM Bietigheim vs. SC Magdeburg",
      "startTimeLocal": "2025-06-08T15:00:00",
      "status": {
        "label": "Final",
        "value": "CONFIRMED"
      },
      "competitors": [
        {
          "name": "SG BBM Bietigheim",
          "score": "25",
          "isHome": true
        },
        {
          "name": "SC Magdeburg", 
          "score": "35",
          "isHome": false
        }
      ],
      "round": "Round: 34",
      "venue": null,
      "attendance": 3750
    }
  ]
}
```

## How It Works

1. **Base Request**: Starts with the standard API endpoint
2. **Pagination Discovery**: Tests common pagination parameters
3. **Round Fetching**: Attempts to fetch data round by round
4. **Page Fetching**: Attempts to fetch data page by page
5. **Deduplication**: Removes duplicate fixtures based on `fixtureId`
6. **Data Saving**: Saves all unique fixtures to JSON file

## Rate Limiting

Both scripts include built-in delays between requests (1 second) to be respectful to the API server.

## Error Handling

- Network timeouts and connection errors
- JSON parsing errors
- Missing or malformed data
- Graceful handling of pagination limits

## Customization

You can modify the scripts to:
- Change the delay between requests
- Adjust maximum rounds/pages to fetch
- Modify output format
- Add additional data processing

## Example Output

```
Making request with params: {'locale': 'en-EN'}
Base request returned 1 fixtures
Exploring pagination options...
Testing pagination with: {'page': 1}
Testing pagination with: {'offset': 0, 'limit': 50}
Attempting to fetch by rounds...
Fetching round 1
Found 10 fixtures in round 1
Fetching round 2
Found 10 fixtures in round 2
...
Total unique fixtures found: 306

✅ Successfully scraped 306 fixtures
📁 Data saved to: handball_season_20250115_103000.json
📅 Date range: 2024-08-01 to 2025-06-08
🏆 Teams found: 18
```
