#!/usr/bin/env node
/**
 * Handball Season Data Scraper (Node.js)
 * Fetches complete season data from Atrium Sports API
 */

const https = require('https');
const fs = require('fs').promises;
const { URL } = require('url');

class HandballSeasonScraper {
    constructor(baseUrl = 'https://eapi.web.prod.cloud.atriumsports.com/v1/embed/248/fixtures') {
        this.baseUrl = baseUrl;
        this.allFixtures = [];
        this.delay = 1000; // Delay between requests in milliseconds
    }

    /**
     * Make HTTP request with error handling and extract fixtures
     */
    async makeRequest(params = {}) {
        const defaultParams = { locale: 'en-EN', ...params };
        const url = new URL(this.baseUrl);

        Object.keys(defaultParams).forEach(key => {
            url.searchParams.append(key, defaultParams[key]);
        });

        console.log(`Making request: ${url.toString()}`);

        return new Promise((resolve, reject) => {
            const options = {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-EN,en;q=0.9'
                }
            };

            const req = https.get(url.toString(), options, (res) => {
                let data = '';

                res.on('data', (chunk) => {
                    data += chunk;
                });

                res.on('end', () => {
                    try {
                        const jsonData = JSON.parse(data);

                        // Extract fixtures from the correct location in the response
                        if (jsonData && jsonData.data && jsonData.data.fixtures) {
                            const fixtures = jsonData.data.fixtures;
                            console.log(`Found ${fixtures.length} fixtures in response`);
                            resolve(fixtures);
                        } else {
                            console.warn('No fixtures found in response');
                            resolve([]);
                        }
                    } catch (error) {
                        console.error('JSON parse error:', error);
                        resolve(null);
                    }
                });
            });

            req.on('error', (error) => {
                console.error('Request error:', error);
                resolve(null);
            });

            req.setTimeout(30000, () => {
                req.destroy();
                console.error('Request timeout');
                resolve(null);
            });
        });
    }

    /**
     * Sleep function for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Explore different pagination parameters
     */
    async explorePagination() {
        const paginationParams = [
            { page: 1 },
            { offset: 0, limit: 50 },
            { round: 1 },
            { matchday: 1 }
        ];

        const results = {};
        
        for (const params of paginationParams) {
            console.log(`Testing pagination with:`, params);
            const fixtures = await this.makeRequest(params);

            if (fixtures) {
                results[JSON.stringify(params)] = fixtures.length;
            }

            await this.sleep(this.delay);
        }

        return results;
    }

    /**
     * Try to fetch data by round numbers
     */
    async fetchByRounds(maxRounds = 50) {
        const allFixtures = [];
        
        for (let roundNum = 1; roundNum <= maxRounds; roundNum++) {
            console.log(`Fetching round ${roundNum}`);
            
            // Try different round parameter names
            const roundParams = ['round', 'matchday', 'gameweek'];
            let foundData = false;
            
            for (const roundParam of roundParams) {
                const fixtures = await this.makeRequest({ [roundParam]: roundNum });

                if (fixtures && fixtures.length > 0) {
                    console.log(`Found ${fixtures.length} fixtures in round ${roundNum}`);
                    allFixtures.push(...fixtures);
                    foundData = true;
                    break;
                }

                await this.sleep(this.delay);
            }
            
            if (!foundData) {
                console.log(`No data found for round ${roundNum}`);
                if (roundNum > 5) { // Stop if we haven't found data for several rounds
                    break;
                }
            }
            
            await this.sleep(this.delay);
        }
        
        return allFixtures;
    }

    /**
     * Try to fetch data using pagination
     */
    async fetchByPagination(maxPages = 100) {
        const allFixtures = [];
        
        for (let page = 1; page <= maxPages; page++) {
            console.log(`Fetching page ${page}`);
            
            // Try different pagination parameter names
            const pageParams = ['page', 'offset'];
            let foundData = false;
            
            for (const pageParam of pageParams) {
                let params;
                if (pageParam === 'offset') {
                    params = { offset: (page - 1) * 50, limit: 50 };
                } else {
                    params = { [pageParam]: page };
                }
                
                const fixtures = await this.makeRequest(params);

                if (fixtures && fixtures.length > 0) {
                    console.log(`Found ${fixtures.length} fixtures on page ${page}`);
                    allFixtures.push(...fixtures);
                    foundData = true;
                    break;
                }
                
                await this.sleep(this.delay);
            }
            
            if (!foundData) {
                console.log(`No data found for page ${page}`);
                if (page > 3) { // Stop if we haven't found data for several pages
                    break;
                }
            }
            
            await this.sleep(this.delay);
        }
        
        return allFixtures;
    }

    /**
     * Main method to fetch all season data
     */
    async fetchAllData() {
        console.log('Starting to fetch handball season data...');
        
        // First, get the basic data
        const baseFixtures = await this.makeRequest();
        if (!baseFixtures) {
            console.error('Failed to fetch base data');
            return [];
        }

        console.log(`Base request returned ${baseFixtures.length} fixtures`);
        const allFixtures = [...baseFixtures];
        
        // Try to get more data using different methods
        console.log('Exploring pagination options...');
        const paginationResults = await this.explorePagination();
        console.log('Pagination exploration results:', paginationResults);
        
        // Try fetching by rounds
        console.log('Attempting to fetch by rounds...');
        const roundFixtures = await this.fetchByRounds();
        if (roundFixtures.length > 0) {
            allFixtures.push(...roundFixtures);
        }
        
        // Try fetching by pagination
        console.log('Attempting to fetch by pagination...');
        const pageFixtures = await this.fetchByPagination();
        if (pageFixtures.length > 0) {
            allFixtures.push(...pageFixtures);
        }
        
        // Remove duplicates based on fixtureId
        const uniqueFixtures = {};
        allFixtures.forEach(fixture => {
            const fixtureId = fixture.fixtureId;
            if (fixtureId && !uniqueFixtures[fixtureId]) {
                uniqueFixtures[fixtureId] = fixture;
            }
        });
        
        const finalFixtures = Object.values(uniqueFixtures);
        console.log(`Total unique fixtures found: ${finalFixtures.length}`);
        
        return finalFixtures;
    }

    /**
     * Save fixtures data to JSON file
     */
    async saveData(fixtures, filename = null) {
        if (!filename) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            filename = `handball_season_${timestamp}.json`;
        }
        
        const dataToSave = {
            metadata: {
                total_fixtures: fixtures.length,
                scraped_at: new Date().toISOString(),
                source_url: this.baseUrl
            },
            fixtures: fixtures
        };
        
        await fs.writeFile(filename, JSON.stringify(dataToSave, null, 2), 'utf8');
        console.log(`Data saved to ${filename}`);
        return filename;
    }
}

/**
 * Main execution function
 */
async function main() {
    const scraper = new HandballSeasonScraper();
    
    try {
        const fixtures = await scraper.fetchAllData();
        
        if (fixtures.length > 0) {
            const filename = await scraper.saveData(fixtures);
            console.log(`\n✅ Successfully scraped ${fixtures.length} fixtures`);
            console.log(`📁 Data saved to: ${filename}`);
            
            // Print some statistics
            if (fixtures.length > 0) {
                const dates = fixtures
                    .map(f => f.startTimeLocal)
                    .filter(date => date)
                    .sort();
                
                if (dates.length > 0) {
                    console.log(`📅 Date range: ${dates[0].slice(0, 10)} to ${dates[dates.length - 1].slice(0, 10)}`);
                }
                
                const teams = new Set();
                fixtures.forEach(fixture => {
                    if (fixture.competitors) {
                        fixture.competitors.forEach(competitor => {
                            if (competitor.name) {
                                teams.add(competitor.name);
                            }
                        });
                    }
                });
                console.log(`🏆 Teams found: ${teams.size}`);
            }
        } else {
            console.log('❌ No fixtures found');
        }
        
    } catch (error) {
        console.error('❌ Error occurred:', error);
    }
}

// Run if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = HandballSeasonScraper;
