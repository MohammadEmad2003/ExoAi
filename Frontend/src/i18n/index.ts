import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Import translation files
import enTranslations from './locales/en.json';
import esTranslations from './locales/es.json';
import arTranslations from './locales/ar.json';

const resources = {
  en: {
    translation: enTranslations
  },
  es: {
    translation: esTranslations
  },
  ar: {
    translation: arTranslations
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    debug: process.env.NODE_ENV === 'development',
    
    interpolation: {
      escapeValue: false, // React already does escaping
    },
    
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
    },
  });

// Set document direction and language based on language changes
i18n.on('languageChanged', (lng) => {
  const dir = lng === 'ar' ? 'rtl' : 'ltr';
  const isArabic = lng === 'ar';
  
  // Set direction and language
  document.documentElement.dir = dir;
  document.documentElement.lang = lng;
  
  // Add/remove Arabic class for CSS targeting
  if (isArabic) {
    document.documentElement.classList.add('lang-arabic');
  } else {
    document.documentElement.classList.remove('lang-arabic');
  }
  
  // Update meta tags if available
  const titleMeta = document.querySelector('title');
  const descMeta = document.querySelector('meta[name="description"]');
  
  // Store original meta content if not already stored
  if (!document.documentElement.dataset.originalTitle && titleMeta) {
    document.documentElement.dataset.originalTitle = titleMeta.textContent || '';
  }
  if (!document.documentElement.dataset.originalDesc && descMeta) {
    document.documentElement.dataset.originalDesc = descMeta.getAttribute('content') || '';
  }
  
  // Update title and description based on language (basic implementation)
  if (titleMeta) {
    const originalTitle = document.documentElement.dataset.originalTitle || '';
    if (isArabic && originalTitle) {
      titleMeta.textContent = 'ExoPlanetAI - اكتشف الكواكب الخارجية باستخدام نموذج الذكاء الاصطناعي المقاوم للهجمات والقابل للتفسير';
    } else {
      titleMeta.textContent = originalTitle;
    }
  }
  
  if (descMeta) {
    const originalDesc = document.documentElement.dataset.originalDesc || '';
    if (isArabic && originalDesc) {
      descMeta.setAttribute('content', 'نموذج تعلم آلي متقدم لاكتشاف الكواكب الخارجية في مجموعات بيانات كبلر وتيس من ناسا. استكشف نهجنا في الذكاء الاصطناعي والعروض التفاعلية والتصورات العلمية للاكتشاف الفلكي.');
    } else {
      descMeta.setAttribute('content', originalDesc);
    }
  }
  
  // Store language preference
  localStorage.setItem('preferred-language', lng);
});

// Set initial direction and language
const storedLang = localStorage.getItem('preferred-language');
const currentLang = storedLang || i18n.language || 'en';

// Initialize language if stored preference exists
if (storedLang && storedLang !== i18n.language) {
  i18n.changeLanguage(storedLang);
} else {
  // Trigger initial setup
  const dir = currentLang === 'ar' ? 'rtl' : 'ltr';
  document.documentElement.dir = dir;
  document.documentElement.lang = currentLang;
  
  if (currentLang === 'ar') {
    document.documentElement.classList.add('lang-arabic');
  }
}

export default i18n;
