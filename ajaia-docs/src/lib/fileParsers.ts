import mammoth from 'mammoth';

export function markdownToHtml(markdown: string): string {
  let html = markdown
    // Normalize newlines
    .replace(/\r\n/g, '\n')
    // Escape HTML entities to prevent raw injection
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Blockquotes
  html = html.replace(/^\&gt;\s?(.*$)/gim, '<blockquote>$1</blockquote>');

  // Bold & Italic
  html = html.replace(/\*\*\*(.*?)\*\*\*/gim, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
  html = html.replace(/__(.*?)__/gim, '<strong>$1</strong>');
  html = html.replace(/_(.*?)_/gim, '<em>$1</em>');

  // Code blocks
  html = html.replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>');
  html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');

  // Bullet Lists
  html = html.replace(/^\s*[-*+]\s+(.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]*?<\/li>)/gi, '<ul>$1</ul>');

  // Numbered Lists
  html = html.replace(/^\s*\d+\.\s+(.*$)/gim, '<li>$1</li>');

  // Paragraphs
  const lines = html.split('\n');
  const processedLines = lines.map((line) => {
    const trimmed = line.trim();
    if (!trimmed) return '';
    if (
      trimmed.startsWith('<h1') ||
      trimmed.startsWith('<h2') ||
      trimmed.startsWith('<h3') ||
      trimmed.startsWith('<ul') ||
      trimmed.startsWith('<ol') ||
      trimmed.startsWith('<li') ||
      trimmed.startsWith('<blockquote') ||
      trimmed.startsWith('<pre')
    ) {
      return trimmed;
    }
    return `<p>${trimmed}</p>`;
  });

  return processedLines.filter(Boolean).join('');
}

export function plainTextToHtml(text: string): string {
  return text
    .split(/\n\n+/)
    .map((paragraph) => `<p>${paragraph.replace(/\n/g, '<br/>').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>`)
    .join('');
}

export async function parseUploadedBuffer(
  buffer: Buffer,
  filename: string,
  mimeType: string
): Promise<{ title: string; content: string; format: string }> {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const baseTitle = filename.substring(0, filename.lastIndexOf('.')) || filename;

  if (ext === 'docx' || mimeType.includes('wordprocessingml')) {
    const result = await mammoth.convertToHtml({ buffer });
    return {
      title: baseTitle,
      content: result.value || '<p>Empty Word document</p>',
      format: 'docx',
    };
  }

  const rawText = buffer.toString('utf-8');

  if (ext === 'md' || ext === 'markdown') {
    return {
      title: baseTitle,
      content: markdownToHtml(rawText),
      format: 'markdown',
    };
  }

  if (ext === 'json') {
    try {
      const parsed = JSON.parse(rawText);
      return {
        title: parsed.title || baseTitle,
        content: parsed.content || `<p>${JSON.stringify(parsed, null, 2)}</p>`,
        format: 'json',
      };
    } catch {
      return {
        title: baseTitle,
        content: plainTextToHtml(rawText),
        format: 'json',
      };
    }
  }

  // Default: plain text (.txt) or general
  return {
    title: baseTitle,
    content: plainTextToHtml(rawText),
    format: 'text',
  };
}
