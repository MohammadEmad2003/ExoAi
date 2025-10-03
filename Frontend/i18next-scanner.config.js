import fs from 'fs';
import path from 'path';

const config = {
  input: [
    'src/**/*.{ts,tsx}',
    // HTML files if any
    'index.html',
    // Include component files
    '!src/**/*.test.{ts,tsx}',
    '!src/**/*.stories.{ts,tsx}',
    '!**/node_modules/**'
  ],
  output: './src/i18n/locales',
  options: {
    debug: false,
    removeUnusedKeys: false,
    sort: true,
    func: {
      // The function name(s) to extract strings for
      list: ['t', 'i18next.t', 'i18n.t'],
      extensions: ['.ts', '.tsx']
    },
    trans: {
      // The Trans component to extract from
      component: 'Trans',
      i18nKey: 'i18nKey',
      defaultsKey: 'defaults',
      extensions: ['.ts', '.tsx'],
      fallbackKey: function(ns, value) {
        return value;
      }
    },
    lngs: ['en', 'ar', 'es'],
    ns: ['translation'],
    defaultLng: 'en',
    defaultNs: 'translation',
    defaultValue: function(lng, ns, key) {
      if (lng === 'en') {
        // For English, return the key as placeholder until we replace with actual values
        return key;
      }
      return '';
    },
    resource: {
      loadPath: '{{lng}}.json',
      savePath: '{{lng}}.missing.json',
      jsonIndent: 2,
      lineEnding: '\n'
    },
    nsSeparator: false,
    keySeparator: '.',
    interpolation: {
      prefix: '{{',
      suffix: '}}'
    }
  }
};

export default config;
