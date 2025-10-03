import React from 'react';

interface TechnicalTermProps {
  children: React.ReactNode;
  className?: string;
}

/**
 * TechnicalTerm component ensures technical terms, model names, and 
 * file extensions are displayed LTR within RTL content
 */
export const TechnicalTerm: React.FC<TechnicalTermProps> = ({ 
  children, 
  className = "" 
}) => {
  return (
    <span 
      className={`inline-ltr ${className}`}
      style={{ 
        unicodeBidi: 'isolate', 
        direction: 'ltr' 
      }}
    >
      {children}
    </span>
  );
};

// Convenience component for model names specifically
export const ModelName: React.FC<TechnicalTermProps> = ({ 
  children, 
  className = "" 
}) => (
  <TechnicalTerm className={`font-semibold text-primary ${className}`}>
    {children}
  </TechnicalTerm>
);

export default TechnicalTerm;
