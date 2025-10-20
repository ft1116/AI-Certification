import React from 'react';

interface MarkdownRendererProps {
  content: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  const renderMarkdown = (text: string) => {
    // Split by lines to process each line
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let key = 0;

    lines.forEach((line, index) => {
      const trimmedLine = line.trim();
      
      // Skip empty lines
      if (!trimmedLine) {
        elements.push(<br key={key++} />);
        return;
      }

      // Headers
      if (trimmedLine.startsWith('### ')) {
        elements.push(
          <h3 key={key++} style={{ 
            fontSize: '1.1em', 
            fontWeight: 'bold', 
            margin: '12px 0 8px 0',
            color: '#1f2937'
          }}>
            {trimmedLine.substring(4)}
          </h3>
        );
      } else if (trimmedLine.startsWith('## ')) {
        elements.push(
          <h2 key={key++} style={{ 
            fontSize: '1.3em', 
            fontWeight: 'bold', 
            margin: '16px 0 10px 0',
            color: '#111827',
            borderBottom: '2px solid #e5e7eb',
            paddingBottom: '4px'
          }}>
            {trimmedLine.substring(3)}
          </h2>
        );
      } else if (trimmedLine.startsWith('# ')) {
        elements.push(
          <h1 key={key++} style={{ 
            fontSize: '1.5em', 
            fontWeight: 'bold', 
            margin: '20px 0 12px 0',
            color: '#111827'
          }}>
            {trimmedLine.substring(2)}
          </h1>
        );
      }
      // Bullet points
      else if (trimmedLine.startsWith('- ')) {
        const bulletText = trimmedLine.substring(2);
        elements.push(
          <div key={key++} style={{ 
            margin: '4px 0 4px 20px',
            display: 'flex',
            alignItems: 'flex-start'
          }}>
            <span style={{ marginRight: '8px', color: '#6b7280' }}>•</span>
            <span>{renderInlineMarkdown(bulletText)}</span>
          </div>
        );
      }
      // Regular paragraphs
      else {
        elements.push(
          <p key={key++} style={{ 
            margin: '8px 0',
            lineHeight: '1.6'
          }}>
            {renderInlineMarkdown(trimmedLine)}
          </p>
        );
      }
    });

    return elements;
  };

  const renderInlineMarkdown = (text: string) => {
    // Handle bold text **text**
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={index} style={{ fontWeight: 'bold', color: '#1f2937' }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  return (
    <div style={{ 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      lineHeight: '1.6',
      color: '#374151'
    }}>
      {renderMarkdown(content)}
    </div>
  );
};

export default MarkdownRenderer;
