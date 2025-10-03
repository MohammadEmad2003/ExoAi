#!/bin/bash

# i18n_audit.sh - Check translation completeness for Arabic localization
# This script ensures no keys are missing in Arabic translations

set -e

echo "🌐 Running i18n audit for Arabic localization..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
LOCALES_DIR="src/i18n/locales"
EN_FILE="$LOCALES_DIR/en.json"
AR_FILE="$LOCALES_DIR/ar.json"
MISSING_FILE="$LOCALES_DIR/ar.missing.json"

# Step 1: Run i18next-scanner to extract strings
echo -e "${BLUE}Step 1: Extracting translatable strings...${NC}"
npx i18next-scanner --config i18next-scanner.config.js

# Step 2: Check if files exist
if [ ! -f "$EN_FILE" ]; then
    echo -e "${RED}❌ Error: English locale file not found at $EN_FILE${NC}"
    exit 1
fi

if [ ! -f "$AR_FILE" ]; then
    echo -e "${RED}❌ Error: Arabic locale file not found at $AR_FILE${NC}"
    exit 1
fi

# Step 3: Generate i18n report
echo -e "${BLUE}Step 2: Generating translation report...${NC}"
node scripts/generate_i18n_report.js

# Step 4: Check for missing keys
if [ -f "$MISSING_FILE" ]; then
    MISSING_COUNT=$(node -e "
        const fs = require('fs');
        try {
            const missing = JSON.parse(fs.readFileSync('$MISSING_FILE', 'utf8'));
            const count = Object.keys(missing).length;
            console.log(count);
        } catch (e) {
            console.log(0);
        }
    ")
    
    if [ "$MISSING_COUNT" -gt 0 ]; then
        echo -e "${RED}❌ Found $MISSING_COUNT missing Arabic translations:${NC}"
        node -e "
            const fs = require('fs');
            const missing = JSON.parse(fs.readFileSync('$MISSING_FILE', 'utf8'));
            Object.keys(missing).forEach(key => console.log('  - ' + key));
        "
        echo -e "${YELLOW}💡 Run 'npm run i18n:translate' to auto-translate missing keys${NC}"
        exit 1
    fi
fi

# Step 5: Check for hardcoded strings
echo -e "${BLUE}Step 3: Checking for hardcoded English strings...${NC}"
HARDCODED_COUNT=$(grep -r -E '"[A-Z][a-zA-Z\s,.'\'!'?:;-]{10,}"' src/ --include="*.tsx" --include="*.ts" | grep -v "\.test\." | grep -v "\.stories\." | wc -l || echo "0")

if [ "$HARDCODED_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $HARDCODED_COUNT potential hardcoded strings:${NC}"
    grep -r -E '"[A-Z][a-zA-Z\s,.'\'!'?:;-]{10,}"' src/ --include="*.tsx" --include="*.ts" | grep -v "\.test\." | grep -v "\.stories\." | head -5
    echo -e "${YELLOW}💡 Consider replacing these with t() function calls${NC}"
fi

# Step 6: Check report file
if [ -f "i18n-report.json" ]; then
    COVERAGE=$(node -e "
        const fs = require('fs');
        const report = JSON.parse(fs.readFileSync('i18n-report.json', 'utf8'));
        console.log(Math.round(report.coverage_percentage));
    ")
    
    if [ "$COVERAGE" -lt 95 ]; then
        echo -e "${RED}❌ Translation coverage is only $COVERAGE% (required: ≥95%)${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Translation coverage: $COVERAGE%${NC}"
    fi
fi

echo -e "${GREEN}✅ i18n audit completed successfully!${NC}"
echo -e "${BLUE}📊 Check i18n-report.json for detailed statistics${NC}"

exit 0
