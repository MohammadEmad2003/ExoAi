#!/usr/bin/env node

/**
 * generate_i18n_report.js - Generate comprehensive i18n translation report
 */

import fs from 'fs';
import path from 'path';

const LOCALES_DIR = 'src/i18n/locales';
const OUTPUT_FILE = 'i18n-report.json';

function deepKeys(obj, prefix = '') {
  let keys = [];
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      keys = keys.concat(deepKeys(value, fullKey));
    } else {
      keys.push(fullKey);
    }
  }
  return keys;
}

function countAutoTranslated(obj, prefix = '') {
  let count = 0;
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      if (value._autoTranslated === true) {
        count++;
      } else {
        count += countAutoTranslated(value, fullKey);
      }
    }
  }
  return count;
}

function getValueByPath(obj, path) {
  return path.split('.').reduce((current, key) => current?.[key], obj);
}

function checkPagesWithUntranslated() {
  const pages = [
    'Home', 'Demo', 'Onboarding', 'ResearchHub', 'Quantum', 
    'Adversarial', 'Visualizations', 'Team', 'Resources', 'Challenge', 'Approach'
  ];
  
  // This would need a more sophisticated analysis in a real implementation
  // For now, we'll return an empty array as placeholder
  return [];
}

function main() {
  try {
    console.log('📊 Generating i18n translation report...');

    // Read locale files
    const enPath = path.join(LOCALES_DIR, 'en.json');
    const arPath = path.join(LOCALES_DIR, 'ar.json');
    const esPath = path.join(LOCALES_DIR, 'es.json');

    if (!fs.existsSync(enPath)) {
      throw new Error(`English locale file not found: ${enPath}`);
    }

    const enData = JSON.parse(fs.readFileSync(enPath, 'utf8'));
    const arData = fs.existsSync(arPath) ? JSON.parse(fs.readFileSync(arPath, 'utf8')) : {};
    const esData = fs.existsSync(esPath) ? JSON.parse(fs.readFileSync(esPath, 'utf8')) : {};

    // Extract all keys
    const enKeys = deepKeys(enData);
    const arKeys = deepKeys(arData);
    const esKeys = deepKeys(esData);

    // Find missing keys
    const missingInAr = enKeys.filter(key => !getValueByPath(arData, key));
    const missingInEs = enKeys.filter(key => !getValueByPath(esData, key));

    // Count auto-translated
    const autoTranslatedAr = countAutoTranslated(arData);
    const autoTranslatedEs = countAutoTranslated(esData);

    // Calculate coverage
    const arCoverage = arKeys.length / enKeys.length * 100;
    const esCoverage = esKeys.length / enKeys.length * 100;

    // Check for pages with untranslated strings
    const pagesWithUntranslated = checkPagesWithUntranslated();

    // Generate report
    const report = {
      generated_at: new Date().toISOString(),
      total_keys: enKeys.length,
      languages: {
        en: {
          keys_count: enKeys.length,
          keys_translated: enKeys.length,
          keys_missing: 0,
          keys_auto_translated: 0,
          coverage_percentage: 100
        },
        ar: {
          keys_count: arKeys.length,
          keys_translated: arKeys.length - missingInAr.length,
          keys_missing: missingInAr.length,
          keys_auto_translated: autoTranslatedAr,
          coverage_percentage: Math.round(arCoverage * 100) / 100,
          missing_keys: missingInAr
        },
        es: {
          keys_count: esKeys.length,
          keys_translated: esKeys.length - missingInEs.length,
          keys_missing: missingInEs.length,
          keys_auto_translated: autoTranslatedEs,
          coverage_percentage: Math.round(esCoverage * 100) / 100,
          missing_keys: missingInEs
        }
      },
      overall_coverage_percentage: Math.round(((arKeys.length + esKeys.length) / (enKeys.length * 2)) * 100),
      pages_with_untranslated: pagesWithUntranslated,
      summary: {
        total_keys: enKeys.length,
        arabic_coverage: Math.round(arCoverage * 100) / 100,
        spanish_coverage: Math.round(esCoverage * 100) / 100,
        needs_review: autoTranslatedAr + autoTranslatedEs
      }
    };

    // Write report
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(report, null, 2));
    
    console.log(`✅ Report generated: ${OUTPUT_FILE}`);
    console.log(`📈 Arabic coverage: ${report.languages.ar.coverage_percentage}%`);
    console.log(`📈 Spanish coverage: ${report.languages.es.coverage_percentage}%`);
    console.log(`🔍 Keys needing review: ${report.summary.needs_review}`);
    
    if (missingInAr.length > 0) {
      console.log(`❌ Missing Arabic keys: ${missingInAr.length}`);
    }

  } catch (error) {
    console.error('❌ Error generating i18n report:', error.message);
    process.exit(1);
  }
}

// Always run when script is executed directly
main();
