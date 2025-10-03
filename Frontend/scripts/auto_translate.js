#!/usr/bin/env node

/**
 * auto_translate.js - Auto-translate missing keys using a translation service
 * Marks auto-translated entries for human review
 */

import fs from 'fs';
import path from 'path';

// Translation mapping for common terms
const TRANSLATION_MAP = {
  // Navigation
  "Home": "الرئيسية",
  "Demo": "العرض التوضيحي", 
  "Team": "الفريق",
  "Resources": "الموارد",
  
  // Common UI elements
  "Loading": "جارٍ التحميل",
  "Save": "حفظ",
  "Cancel": "إلغاء",
  "Delete": "حذف",
  "Edit": "تعديل",
  "View": "عرض",
  "Download": "تنزيل",
  "Upload": "رفع",
  "Search": "بحث",
  "Filter": "تصفية",
  "Sort": "ترتيب",
  "Settings": "الإعدادات",
  "Help": "مساعدة",
  "Close": "إغلاق",
  "Open": "فتح",
  "Yes": "نعم",
  "No": "لا",
  "OK": "موافق",
  "Error": "خطأ",
  "Success": "نجاح",
  "Warning": "تحذير",
  "Info": "معلومات",
  
  // Forms and inputs
  "Name": "الاسم",
  "Email": "البريد الإلكتروني",
  "Password": "كلمة المرور",
  "Submit": "إرسال",
  "Reset": "إعادة تعيين",
  "Required": "مطلوب",
  "Optional": "اختياري",
  
  // Model terms (keep English)
  "TabKANet": "TabKANet",
  "QSVC": "QSVC",
  
  // File formats (keep English)
  ".csv": ".csv",
  ".json": ".json",
  ".parquet": ".parquet",
  ".pt": ".pt",
  ".onnx": ".onnx",
  
  // Technical terms with Arabic
  "Machine Learning": "التعلم الآلي",
  "Deep Learning": "التعلم العميق",
  "Neural Network": "الشبكة العصبية",
  "Dataset": "مجموعة البيانات",
  "Training": "التدريب",
  "Validation": "التحقق",
  "Testing": "الاختبار",
  "Accuracy": "الدقة",
  "Precision": "الدقة",
  "Recall": "الاستدعاء",
  "Model": "النموذج",
  "Algorithm": "الخوارزمية",
  "Performance": "الأداء",
  "Optimization": "الأمثلة",
  "Parameters": "المعلمات",
  "Hyperparameters": "المعلمات الفائقة",
  "Configuration": "الإعداد",
  
  // Common phrases
  "Start Training": "بدء التدريب",
  "View Results": "عرض النتائج",
  "Export Model": "تصدير النموذج",
  "Upload Data": "رفع البيانات",
  "Select File": "اختر ملفًا",
  "Process Data": "معالجة البيانات",
  "Train Model": "تدريب النموذج",
  "Evaluate Model": "تقييم النموذج"
};

function translateText(text, targetLang = 'ar') {
  // First check our manual translation map
  if (TRANSLATION_MAP[text]) {
    return TRANSLATION_MAP[text];
  }
  
  // For demo purposes, return placeholder Arabic text
  // In production, you would integrate with Google Translate API or similar
  if (targetLang === 'ar') {
    // Check if it's a technical term or model name that should stay in English
    if (isModelName(text) || isTechnicalTerm(text)) {
      return text; // Keep English
    }
    
    // Simple auto-translation placeholder
    return `[AR: ${text}]`; // Placeholder for human translation
  }
  
  return text;
}

function isModelName(text) {
  const modelNames = ['TabKANet', 'QSVC', 'CNN', 'RNN', 'LSTM', 'GAN', 'VAE'];
  return modelNames.some(name => text.includes(name));
}

function isTechnicalTerm(text) {
  const techTerms = ['.csv', '.json', '.pt', '.onnx', 'CLI', 'API', 'HTTP', 'GPU', 'CPU'];
  return techTerms.some(term => text.includes(term));
}

function deepTranslate(obj, targetLang = 'ar', prefix = '') {
  const result = {};
  
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      result[key] = deepTranslate(value, targetLang, prefix ? `${prefix}.${key}` : key);
    } else if (typeof value === 'string') {
      const translated = translateText(value, targetLang);
      
      // Mark as auto-translated if it's different from original and not in our manual map
      if (translated !== value && !TRANSLATION_MAP[value]) {
        result[key] = {
          _value: translated,
          _autoTranslated: true,
          _originalValue: value,
          _needsReview: true
        };
      } else {
        result[key] = translated;
      }
    } else {
      result[key] = value;
    }
  }
  
  return result;
}

function flattenTranslatedObject(obj) {
  const result = {};
  
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      if (value._autoTranslated) {
        result[key] = value._value;
      } else {
        result[key] = flattenTranslatedObject(value);
      }
    } else {
      result[key] = value;
    }
  }
  
  return result;
}

async function main() {
  const LOCALES_DIR = 'src/i18n/locales';
  const EN_FILE = path.join(LOCALES_DIR, 'en.json');
  const AR_FILE = path.join(LOCALES_DIR, 'ar.json');
  
  try {
    console.log('🔄 Auto-translating missing keys...');
    
    // Read existing files
    if (!fs.existsSync(EN_FILE)) {
      throw new Error(`English locale file not found: ${EN_FILE}`);
    }
    
    const enData = JSON.parse(fs.readFileSync(EN_FILE, 'utf8'));
    let arData = {};
    
    if (fs.existsSync(AR_FILE)) {
      arData = JSON.parse(fs.readFileSync(AR_FILE, 'utf8'));
    }
    
    // Create merged translations
    const translatedData = deepTranslate(enData, 'ar');
    
    // Merge with existing Arabic translations (preserve manual translations)
    function mergeTranslations(existing, translated) {
      const result = { ...translated };
      
      for (const [key, value] of Object.entries(existing)) {
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          if (result[key] && typeof result[key] === 'object') {
            result[key] = mergeTranslations(value, result[key]);
          } else {
            result[key] = value; // Keep existing translation
          }
        } else {
          result[key] = value; // Keep existing translation
        }
      }
      
      return result;
    }
    
    const mergedData = mergeTranslations(arData, translatedData);
    const finalData = flattenTranslatedObject(mergedData);
    
    // Write the updated Arabic file
    fs.writeFileSync(AR_FILE, JSON.stringify(finalData, null, 2));
    
    // Count auto-translated entries for reporting
    const autoTranslatedCount = JSON.stringify(translatedData).match(/"_autoTranslated":true/g)?.length || 0;
    
    console.log(`✅ Translation complete!`);
    console.log(`📝 ${autoTranslatedCount} entries auto-translated and marked for review`);
    console.log(`📂 Updated file: ${AR_FILE}`);
    console.log(`💡 Review entries marked with [AR: ...] placeholders`);
    
  } catch (error) {
    console.error('❌ Auto-translation failed:', error.message);
    process.exit(1);
  }
}

// ES module check
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
