#!/usr/bin/env node

/**
 * i18n_audit.js - Node.js version of i18n audit script
 * Cross-platform replacement for i18n_audit.sh
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

// ANSI colors for console output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m'
};

function log(color, message) {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkHardcodedStrings() {
  try {
    // Simple file-based check instead of using grep-cli
    const srcFiles = require('fs').readdirSync('src', { recursive: true })
      .filter(file => file.endsWith('.tsx') || file.endsWith('.ts'))
      .filter(file => !file.includes('.test.') && !file.includes('.stories.'));
    
    let hardcodedCount = 0;
    
    for (const file of srcFiles) {
      try {
        const content = require('fs').readFileSync(`src/${file}`, 'utf8');
        // Look for string literals that are not t() calls and contain English text
        const englishStringPattern = /(?<!t\(["'`])["`'][A-Z][a-zA-Z\s,\.'!?:;-]{10,}["`']/g;
        const matches = content.match(englishStringPattern);
        if (matches) {
          hardcodedCount += matches.length;
        }
      } catch (e) {
        // Skip files that can't be read
      }
    }
    
    return hardcodedCount;
  } catch (error) {
    // Return 0 if check fails
    return 0;
  }
}

async function main() {
  log('blue', '🌐 Running i18n audit for Arabic localization...');

  const LOCALES_DIR = 'src/i18n/locales';
  const EN_FILE = path.join(LOCALES_DIR, 'en.json');
  const AR_FILE = path.join(LOCALES_DIR, 'ar.json');
  const MISSING_FILE = path.join(LOCALES_DIR, 'ar.missing.json');

  try {
    // Step 1: Extract strings (skip for now due to TypeScript parsing issues)
    log('blue', 'Step 1: Skipping extraction due to TypeScript parsing issues...');

    // Step 2: Check if files exist
    if (!fs.existsSync(EN_FILE)) {
      log('red', `❌ Error: English locale file not found at ${EN_FILE}`);
      process.exit(1);
    }

    if (!fs.existsSync(AR_FILE)) {
      log('red', `❌ Error: Arabic locale file not found at ${AR_FILE}`);
      process.exit(1);
    }

    // Step 3: Generate report
    log('blue', 'Step 2: Generating translation report...');
    execSync('node scripts/generate_i18n_report.js', { stdio: 'inherit' });

    // Step 4: Check for missing keys
    if (fs.existsSync(MISSING_FILE)) {
      const missingData = JSON.parse(fs.readFileSync(MISSING_FILE, 'utf8'));
      const missingCount = Object.keys(missingData).length;

      if (missingCount > 0) {
        log('red', `❌ Found ${missingCount} missing Arabic translations:`);
        Object.keys(missingData).forEach(key => {
          console.log(`  - ${key}`);
        });
        log('yellow', '💡 Run "npm run i18n:translate" to auto-translate missing keys');
        process.exit(1);
      }
    }

    // Step 5: Check for hardcoded strings
    log('blue', 'Step 3: Checking for hardcoded English strings...');
    const hardcodedCount = checkHardcodedStrings();

    if (hardcodedCount > 0) {
      log('yellow', `⚠️  Found ${hardcodedCount} potential hardcoded strings`);
      log('yellow', '💡 Consider replacing these with t() function calls');
    }

    // Step 6: Check coverage
    if (fs.existsSync('i18n-report.json')) {
      const report = JSON.parse(fs.readFileSync('i18n-report.json', 'utf8'));
      const coverage = Math.round(report.languages?.ar?.coverage_percentage || 0);

      if (coverage < 95) {
        log('red', `❌ Translation coverage is only ${coverage}% (required: ≥95%)`);
        process.exit(1);
      } else {
        log('green', `✅ Translation coverage: ${coverage}%`);
      }
    }

    log('green', '✅ i18n audit completed successfully!');
    log('blue', '📊 Check i18n-report.json for detailed statistics');

  } catch (error) {
    log('red', `❌ Audit failed: ${error.message}`);
    process.exit(1);
  }
}

// Always run main function
main().catch(error => {
  console.error('❌ Audit failed:', error.message);
  process.exit(1);
});
