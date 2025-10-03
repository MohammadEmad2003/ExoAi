/**
 * i18n utilities for number formatting, date formatting, and locale-specific operations
 */

// Map of locale codes to their Intl locale identifiers
const LOCALE_MAP: Record<string, string> = {
  'en': 'en-US',
  'ar': 'ar-SA',
  'es': 'es-ES'
};

/**
 * Format numbers according to the current locale
 */
export const formatNumber = (
  value: number, 
  locale: string = 'en',
  options: Intl.NumberFormatOptions = {}
): string => {
  const intlLocale = LOCALE_MAP[locale] || locale;
  return new Intl.NumberFormat(intlLocale, options).format(value);
};

/**
 * Format percentages
 */
export const formatPercentage = (
  value: number,
  locale: string = 'en',
  decimals: number = 1
): string => {
  return formatNumber(value, locale, {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
};

/**
 * Format currency (useful for cost calculations)
 */
export const formatCurrency = (
  value: number,
  locale: string = 'en',
  currency: string = 'USD'
): string => {
  return formatNumber(value, locale, {
    style: 'currency',
    currency
  });
};

/**
 * Format dates according to locale
 */
export const formatDate = (
  date: Date | string | number,
  locale: string = 'en',
  options: Intl.DateTimeFormatOptions = {}
): string => {
  const intlLocale = LOCALE_MAP[locale] || locale;
  const dateObj = new Date(date);
  
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    ...options
  };
  
  return new Intl.DateTimeFormat(intlLocale, defaultOptions).format(dateObj);
};

/**
 * Format relative time (e.g., "2 days ago")
 */
export const formatRelativeTime = (
  date: Date | string | number,
  locale: string = 'en'
): string => {
  const intlLocale = LOCALE_MAP[locale] || locale;
  const dateObj = new Date(date);
  const now = new Date();
  const diffInSeconds = (now.getTime() - dateObj.getTime()) / 1000;
  
  const rtf = new Intl.RelativeTimeFormat(intlLocale, { numeric: 'auto' });
  
  const units: Array<[string, number]> = [
    ['year', 365 * 24 * 60 * 60],
    ['month', 30 * 24 * 60 * 60],
    ['week', 7 * 24 * 60 * 60],
    ['day', 24 * 60 * 60],
    ['hour', 60 * 60],
    ['minute', 60],
    ['second', 1]
  ];
  
  for (const [unit, secondsInUnit] of units) {
    if (Math.abs(diffInSeconds) >= secondsInUnit) {
      const value = Math.round(diffInSeconds / secondsInUnit);
      return rtf.format(-value, unit as Intl.RelativeTimeFormatUnit);
    }
  }
  
  return rtf.format(0, 'second');
};

/**
 * Format file sizes
 */
export const formatFileSize = (
  bytes: number,
  locale: string = 'en',
  decimals: number = 2
): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  const value = bytes / Math.pow(k, i);
  const formattedValue = formatNumber(value, locale, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
  
  return `${formattedValue} ${sizes[i]}`;
};

/**
 * Format duration in seconds to human readable format
 */
export const formatDuration = (
  seconds: number,
  locale: string = 'en'
): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  const parts: string[] = [];
  
  if (hours > 0) {
    parts.push(formatNumber(hours, locale) + 'h');
  }
  if (minutes > 0) {
    parts.push(formatNumber(minutes, locale) + 'm');
  }
  if (secs > 0 || parts.length === 0) {
    parts.push(formatNumber(secs, locale) + 's');
  }
  
  return parts.join(' ');
};

/**
 * Check if a locale uses RTL writing direction
 */
export const isRTL = (locale: string): boolean => {
  const rtlLocales = ['ar', 'he', 'fa', 'ur'];
  return rtlLocales.includes(locale);
};

/**
 * Get the text direction for a locale
 */
export const getTextDirection = (locale: string): 'ltr' | 'rtl' => {
  return isRTL(locale) ? 'rtl' : 'ltr';
};

/**
 * Format scientific notation numbers
 */
export const formatScientific = (
  value: number,
  locale: string = 'en',
  significantDigits: number = 3
): string => {
  return formatNumber(value, locale, {
    notation: 'scientific',
    maximumSignificantDigits: significantDigits
  });
};

/**
 * Format coordinates (for astronomical data)
 */
export const formatCoordinates = (
  ra: number,  // Right Ascension
  dec: number, // Declination
  locale: string = 'en'
): string => {
  const raFormatted = formatNumber(ra, locale, { 
    minimumFractionDigits: 6,
    maximumFractionDigits: 6 
  });
  const decFormatted = formatNumber(dec, locale, { 
    minimumFractionDigits: 6,
    maximumFractionDigits: 6,
    signDisplay: 'always'
  });
  
  return `RA: ${raFormatted}°, Dec: ${decFormatted}°`;
};

/**
 * Smart truncation that respects word boundaries in different languages
 */
export const truncateText = (
  text: string,
  maxLength: number,
  locale: string = 'en'
): string => {
  if (text.length <= maxLength) return text;
  
  // For Arabic, we might want different truncation rules
  if (locale === 'ar') {
    // Arabic text truncation - be more conservative with word boundaries
    const truncated = text.substring(0, maxLength - 3);
    const lastSpace = truncated.lastIndexOf(' ');
    
    if (lastSpace > maxLength * 0.7) {
      return truncated.substring(0, lastSpace) + '...';
    }
    return truncated + '...';
  }
  
  // Standard truncation for LTR languages
  const truncated = text.substring(0, maxLength - 3);
  const lastSpace = truncated.lastIndexOf(' ');
  
  if (lastSpace > maxLength * 0.8) {
    return truncated.substring(0, lastSpace) + '...';
  }
  return truncated + '...';
};
