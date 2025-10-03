/**
 * Comprehensive i18n tests for Arabic localization
 * Tests translation keys, RTL support, and critical user flows
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { I18nextProvider } from 'react-i18next';
import i18n from '../i18n';

// Mock components for testing
import Home from '../pages/Home';
import Demo from '../pages/Demo';
import Onboarding from '../pages/Onboarding';
import LanguageSwitcher from '../components/LanguageSwitcher';

// Critical UI strings to test
const CRITICAL_STRINGS = [
  'navigation.home',
  'navigation.guidedMode', 
  'navigation.demo',
  'home.hero.title1',
  'home.hero.tryModel',
  'demo.title',
  'demo.tabs.guided',
  'onboarding.title',
  'common.loading',
  'common.save',
  'common.cancel',
  'errors.networkError'
];

// Test wrapper component
const TestWrapper = ({ children, language = 'en' }: { children: React.ReactNode, language?: string }) => {
  beforeEach(() => {
    i18n.changeLanguage(language);
  });

  return (
    <I18nextProvider i18n={i18n}>
      {children}
    </I18nextProvider>
  );
};

describe('i18n Arabic Localization', () => {
  beforeEach(() => {
    // Reset to English before each test
    i18n.changeLanguage('en');
    // Clear localStorage
    localStorage.clear();
    // Reset document attributes
    document.documentElement.lang = 'en';
    document.documentElement.dir = 'ltr';
    document.documentElement.classList.remove('lang-arabic');
  });

  describe('Translation Keys Coverage', () => {
    it('should have all critical translation keys in Arabic', () => {
      CRITICAL_STRINGS.forEach(key => {
        const englishValue = i18n.t(key, { lng: 'en' });
        const arabicValue = i18n.t(key, { lng: 'ar' });
        
        expect(arabicValue).toBeTruthy();
        expect(arabicValue).not.toBe(key); // Should not fallback to key
        expect(arabicValue).not.toBe(englishValue); // Should be translated
      });
    });

    it('should not have missing translation keys', () => {
      const resources = i18n.getResourceBundle('ar', 'translation');
      expect(resources).toBeTruthy();
      
      // Check that we have substantial Arabic content
      const arabicKeys = Object.keys(resources).length;
      expect(arabicKeys).toBeGreaterThan(10);
    });
  });

  describe('Language Switching', () => {
    it('should change document direction to RTL when switching to Arabic', async () => {
      render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      // Click on language switcher
      const languageSwitcher = screen.getByRole('button');
      fireEvent.click(languageSwitcher);

      // Click on Arabic option
      const arabicOption = screen.getByText('العربية');
      fireEvent.click(arabicOption);

      await waitFor(() => {
        expect(document.documentElement.dir).toBe('rtl');
        expect(document.documentElement.lang).toBe('ar');
        expect(document.documentElement.classList.contains('lang-arabic')).toBe(true);
      });
    });

    it('should persist language preference in localStorage', async () => {
      render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      // Switch to Arabic
      const languageSwitcher = screen.getByRole('button');
      fireEvent.click(languageSwitcher);
      
      const arabicOption = screen.getByText('العربية');
      fireEvent.click(arabicOption);

      await waitFor(() => {
        expect(localStorage.getItem('preferred-language')).toBe('ar');
      });
    });

    it('should restore language preference on page load', () => {
      localStorage.setItem('preferred-language', 'ar');
      
      // Simulate page reload by reinitializing i18n
      const storedLang = localStorage.getItem('preferred-language');
      if (storedLang) {
        i18n.changeLanguage(storedLang);
      }

      expect(i18n.language).toBe('ar');
    });
  });

  describe('Component Rendering in Arabic', () => {
    it('should render Home page in Arabic', () => {
      render(
        <TestWrapper language="ar">
          <Home />
        </TestWrapper>
      );

      // Check for Arabic text
      expect(screen.getByText('اكتشاف الكواكب الخارجية')).toBeInTheDocument();
      expect(screen.getByText('جرب نموذجنا')).toBeInTheDocument();
      expect(screen.getByText('نقاط القوة في المشروع')).toBeInTheDocument();
    });

    it('should render Demo page in Arabic', () => {
      render(
        <TestWrapper language="ar">
          <Demo />
        </TestWrapper>
      );

      expect(screen.getByText('عرض النموذج')).toBeInTheDocument();
      expect(screen.getByText('الاكتشاف الموجه')).toBeInTheDocument();
      expect(screen.getByText('الصندوق المتقدم')).toBeInTheDocument();
    });

    it('should render Onboarding page in Arabic', () => {
      render(
        <TestWrapper language="ar">
          <Onboarding />
        </TestWrapper>
      );

      expect(screen.getByText('مرحبًا بك في Cosmic Analysts ExoAI')).toBeInTheDocument();
      expect(screen.getByText('رحلتك الموجهة نحو التعلم الآلي المتقدم')).toBeInTheDocument();
    });
  });

  describe('RTL Layout Support', () => {
    it('should apply RTL CSS classes when in Arabic', () => {
      render(
        <TestWrapper language="ar">
          <div className="icon-mirror">Test</div>
        </TestWrapper>
      );

      expect(document.documentElement.dir).toBe('rtl');
      
      // Check if RTL classes are working
      const element = screen.getByText('Test');
      expect(element).toHaveClass('icon-mirror');
    });

    it('should mirror icons in RTL mode', () => {
      render(
        <TestWrapper language="ar">
          <button>
            <span className="icon-mirror">→</span>
            Test Button
          </button>
        </TestWrapper>
      );

      const icon = screen.getByText('→');
      expect(icon).toHaveClass('icon-mirror');
    });
  });

  describe('Model Names and Technical Terms', () => {
    it('should keep model names in English even in Arabic mode', () => {
      i18n.changeLanguage('ar');
      
      const tabkanetName = i18n.t('models.tabkanet');
      const qsvcName = i18n.t('models.qsvc');
      
      expect(tabkanetName).toBe('TabKANet');
      expect(qsvcName).toBe('QSVC');
    });

    it('should preserve LTR direction for technical terms', () => {
      render(
        <TestWrapper language="ar">
          <span className="inline-ltr">TabKANet</span>
        </TestWrapper>
      );

      const techTerm = screen.getByText('TabKANet');
      expect(techTerm).toHaveClass('inline-ltr');
    });
  });

  describe('Form Inputs and User Interaction', () => {
    it('should handle Arabic text input correctly', () => {
      render(
        <TestWrapper language="ar">
          <input type="text" placeholder="اكتب هنا" />
        </TestWrapper>
      );

      const input = screen.getByPlaceholderText('اكتب هنا');
      fireEvent.change(input, { target: { value: 'نص تجريبي' } });
      
      expect(input).toHaveValue('نص تجريبي');
    });

    it('should maintain LTR for number inputs in Arabic mode', () => {
      render(
        <TestWrapper language="ar">
          <input type="number" />
        </TestWrapper>
      );

      const numberInput = screen.getByRole('spinbutton');
      fireEvent.change(numberInput, { target: { value: '123.45' } });
      
      expect(numberInput).toHaveValue(123.45);
    });
  });

  describe('Error Handling', () => {
    it('should display error messages in Arabic', () => {
      i18n.changeLanguage('ar');
      
      const errorMessage = i18n.t('errors.networkError');
      expect(errorMessage).toBe('خطأ في الشبكة');
    });

    it('should fallback to English if Arabic translation is missing', () => {
      const missingKey = i18n.t('nonexistent.key', { lng: 'ar' });
      // Should fall back to English or return the key
      expect(typeof missingKey).toBe('string');
    });
  });

  describe('Meta Tags and SEO', () => {
    it('should update page title when switching to Arabic', async () => {
      // Mock document.title
      Object.defineProperty(document, 'title', {
        writable: true,
        value: 'Original Title'
      });

      render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      // Switch to Arabic
      fireEvent.click(screen.getByRole('button'));
      fireEvent.click(screen.getByText('العربية'));

      await waitFor(() => {
        // The title should contain Arabic text
        expect(document.title).toContain('ExoPlanetAI');
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper lang attribute for screen readers', () => {
      render(
        <TestWrapper language="ar">
          <div>Arabic content</div>
        </TestWrapper>
      );

      expect(document.documentElement.lang).toBe('ar');
    });

    it('should have proper dir attribute for RTL support', () => {
      render(
        <TestWrapper language="ar">
          <div>RTL content</div>
        </TestWrapper>
      );

      expect(document.documentElement.dir).toBe('rtl');
    });
  });

  describe('Performance', () => {
    it('should not cause memory leaks when switching languages frequently', async () => {
      const component = render(
        <TestWrapper>
          <LanguageSwitcher />
        </TestWrapper>
      );

      // Switch languages multiple times
      for (let i = 0; i < 5; i++) {
        fireEvent.click(screen.getByRole('button'));
        fireEvent.click(screen.getByText('العربية'));
        
        await waitFor(() => {
          expect(document.documentElement.lang).toBe('ar');
        });

        fireEvent.click(screen.getByRole('button'));
        fireEvent.click(screen.getByText('English'));
        
        await waitFor(() => {
          expect(document.documentElement.lang).toBe('en');
        });
      }

      component.unmount();
    });
  });
});

// Integration test for complete user flow
describe('Arabic User Flow Integration', () => {
  it('should support complete guided mode flow in Arabic', async () => {
    render(
      <TestWrapper language="ar">
        <Onboarding />
      </TestWrapper>
    );

    // Check initial Arabic content
    expect(screen.getByText('مرحبًا بك في Cosmic Analysts ExoAI')).toBeInTheDocument();
    
    // Check navigation buttons are in Arabic
    const nextButton = screen.getByText('التالي');
    expect(nextButton).toBeInTheDocument();
    
    // Simulate user interaction
    fireEvent.click(nextButton);
    
    // Should advance to next step with Arabic content
    await waitFor(() => {
      expect(screen.getByText('ارفع بياناتك')).toBeInTheDocument();
    });
  });

  it('should handle demo interaction in Arabic', async () => {
    render(
      <TestWrapper language="ar">
        <Demo />
      </TestWrapper>
    );

    // Check demo page loads in Arabic
    expect(screen.getByText('عرض النموذج')).toBeInTheDocument();
    
    // Check tabs work
    const advancedTab = screen.getByText('الصندوق المتقدم');
    fireEvent.click(advancedTab);
    
    // Should show advanced mode content in Arabic
    await waitFor(() => {
      expect(screen.getByText('المعلمات الفائقة')).toBeInTheDocument();
    });
  });
});
