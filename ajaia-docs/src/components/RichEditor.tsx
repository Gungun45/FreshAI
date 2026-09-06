'use client';

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Underline from '@tiptap/extension-underline';
import Highlight from '@tiptap/extension-highlight';
import TextAlign from '@tiptap/extension-text-align';
import Placeholder from '@tiptap/extension-placeholder';
import { Document, DocumentRevision, UserRole } from '@/lib/types';
import { useAuth } from '@/context/AuthContext';
import { ShareModal } from './ShareModal';
import { FileUploadModal } from './FileUploadModal';
import { RevisionHistoryModal } from './RevisionHistoryModal';
import {
  Bold,
  Italic,
  Underline as UnderlineIcon,
  Strikethrough,
  Heading1,
  Heading2,
  Heading3,
  List,
  ListOrdered,
  Quote,
  Code,
  AlignLeft,
  AlignCenter,
  AlignRight,
  Highlighter,
  RotateCcw,
  RotateCw,
  Save,
  Share2,
  Download,
  UploadCloud,
  History,
  CheckCircle,
  AlertCircle,
  Clock,
  Eye,
  Edit3,
  Users,
  Sparkles,
} from 'lucide-react';
import Link from 'next/link';

interface RichEditorProps {
  initialDocument: Document;
  userRole: UserRole | null;
}

export function RichEditor({ initialDocument, userRole }: RichEditorProps) {
  const { currentUser } = useAuth();
  const [document, setDocument] = useState<Document>(initialDocument);
  const [title, setTitle] = useState(initialDocument.title);
  const [saveStatus, setSaveStatus] = useState<'saved' | 'saving' | 'unsaved' | 'error'>('saved');
  const [lastSavedTime, setLastSavedTime] = useState<string>(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  const [wordCount, setWordCount] = useState(initialDocument.wordCount || 0);
  const [charCount, setCharCount] = useState(initialDocument.characterCount || 0);

  // Modals
  const [isShareOpen, setIsShareOpen] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [exportMenuOpen, setExportMenuOpen] = useState(false);

  // Determine permissions based on current user and document
  const isOwner = document.ownerId === currentUser.id;
  const collaborator = document.collaborators.find((c) => c.userId === currentUser.id || c.email.toLowerCase() === currentUser.email.toLowerCase());
  const effectiveRole: UserRole = isOwner ? 'owner' : collaborator ? collaborator.role : 'viewer';
  const isReadOnly = effectiveRole === 'viewer';

  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const calculateCounts = (text: string) => {
    const plain = text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    const words = plain ? plain.split(/\s+/).length : 0;
    const chars = plain.length;
    setWordCount(words);
    setCharCount(chars);
  };

  const persistChanges = useCallback(
    async (newContent: string, newTitle: string) => {
      if (isReadOnly) return;
      setSaveStatus('saving');

      try {
        const res = await fetch(`/api/documents/${document.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: newTitle,
            content: newContent,
            userId: currentUser.id,
          }),
        });

        const data = await res.json();
        if (!res.ok || !data.success) {
          throw new Error(data.error || 'Failed to save');
        }

        setDocument(data.document);
        setSaveStatus('saved');
        setLastSavedTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      } catch (err) {
        console.error('Save error:', err);
        setSaveStatus('error');
      }
    },
    [document.id, currentUser.id, isReadOnly]
  );

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3],
        },
      }),
      Underline,
      Highlight.configure({ multicolor: true }),
      TextAlign.configure({
        types: ['heading', 'paragraph'],
      }),
      Placeholder.configure({
        placeholder: 'Type your thoughts, document plans, or notes here...',
      }),
    ],
    content: initialDocument.content,
    editable: !isReadOnly,
    immediatelyRender: false,
    onUpdate: ({ editor }) => {
      const html = editor.getHTML();
      calculateCounts(html);
      setSaveStatus('unsaved');

      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      saveTimeoutRef.current = setTimeout(() => {
        persistChanges(html, title);
      }, 1000);
    },
  });

  // Update editor editable state when role changes
  useEffect(() => {
    if (editor) {
      editor.setEditable(!isReadOnly);
    }
  }, [editor, isReadOnly]);

  // Title change handler
  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTitle = e.target.value;
    setTitle(newTitle);
    setSaveStatus('unsaved');

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      if (editor) {
        persistChanges(editor.getHTML(), newTitle);
      }
    }, 1000);
  };

  // Restore Revision Handler
  const handleRestoreRevision = (revision: DocumentRevision) => {
    if (!editor || isReadOnly) return;
    editor.commands.setContent(revision.content);
    setTitle(revision.title);
    calculateCounts(revision.content);
    persistChanges(revision.content, revision.title);
    setIsHistoryOpen(false);
  };

  // Import Content Handler
  const handleImportContent = (importedHtml: string) => {
    if (!editor || isReadOnly) return;
    editor.commands.insertContent(importedHtml);
    const html = editor.getHTML();
    calculateCounts(html);
    persistChanges(html, title);
  };

  // Export handlers
  const exportDocument = (format: 'md' | 'txt' | 'html') => {
    if (!editor) return;
    let fileContent = '';
    let mimeType = 'text/plain';
    let ext = format;

    if (format === 'html') {
      fileContent = `<!DOCTYPE html><html><head><title>${title}</title><meta charset="utf-8"></head><body>${editor.getHTML()}</body></html>`;
      mimeType = 'text/html';
    } else if (format === 'txt') {
      fileContent = editor.getText();
      mimeType = 'text/plain';
    } else if (format === 'md') {
      // Basic HTML to Markdown conversion for export
      let md = editor.getHTML()
        .replace(/<h1>(.*?)<\/h1>/gi, '# $1\n\n')
        .replace(/<h2>(.*?)<\/h2>/gi, '## $1\n\n')
        .replace(/<h3>(.*?)<\/h3>/gi, '### $1\n\n')
        .replace(/<strong>(.*?)<\/strong>/gi, '**$1**')
        .replace(/<em>(.*?)<\/em>/gi, '*$1*')
        .replace(/<code>(.*?)<\/code>/gi, '`$1`')
        .replace(/<blockquote>(.*?)<\/blockquote>/gi, '> $1\n\n')
        .replace(/<li>(.*?)<\/li>/gi, '- $1\n')
        .replace(/<\/ul>/gi, '\n')
        .replace(/<\/ol>/gi, '\n')
        .replace(/<p>(.*?)<\/p>/gi, '$1\n\n')
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<[^>]*>/g, '');
      fileContent = `# ${title}\n\n${md}`;
      mimeType = 'text/markdown';
    }

    const blob = new Blob([fileContent], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = window.document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]/gi, '_')}.${ext}`;
    window.document.body.appendChild(a);
    a.click();
    window.document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setExportMenuOpen(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Document App Bar */}
      <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-4 py-2.5 shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          {/* Left: Document Title & Metadata */}
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <Link
              href="/"
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors text-xs font-semibold flex items-center gap-1.5"
            >
              &larr; Back
            </Link>

            <div className="flex flex-col flex-1">
              <input
                type="text"
                value={title}
                disabled={isReadOnly}
                onChange={handleTitleChange}
                placeholder="Document Title..."
                className={`text-base sm:text-lg font-bold bg-transparent border-b border-transparent hover:border-slate-700 focus:border-indigo-500 focus:bg-slate-950/40 rounded px-1.5 py-0.5 text-slate-100 focus:outline-none transition-all ${
                  isReadOnly ? 'cursor-not-allowed opacity-90' : ''
                }`}
              />

              <div className="flex items-center gap-3 text-[11px] text-slate-400 pl-1.5">
                {/* Save status */}
                <span className="flex items-center gap-1">
                  {saveStatus === 'saved' && (
                    <>
                      <CheckCircle className="w-3 h-3 text-emerald-400" />
                      <span>Saved at {lastSavedTime}</span>
                    </>
                  )}
                  {saveStatus === 'saving' && (
                    <>
                      <Clock className="w-3 h-3 text-amber-400 animate-spin" />
                      <span className="text-amber-300">Saving changes...</span>
                    </>
                  )}
                  {saveStatus === 'unsaved' && <span className="text-slate-400">Unsaved edits...</span>}
                  {saveStatus === 'error' && (
                    <>
                      <AlertCircle className="w-3 h-3 text-rose-400" />
                      <span className="text-rose-400">Save failed</span>
                    </>
                  )}
                </span>

                <span className="text-slate-600">•</span>

                {/* Role badge */}
                <span
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-medium ${
                    effectiveRole === 'owner'
                      ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                      : effectiveRole === 'editor'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  }`}
                >
                  {effectiveRole === 'owner' ? (
                    <Sparkles className="w-2.5 h-2.5" />
                  ) : effectiveRole === 'editor' ? (
                    <Edit3 className="w-2.5 h-2.5" />
                  ) : (
                    <Eye className="w-2.5 h-2.5" />
                  )}
                  {effectiveRole.toUpperCase()} MODE
                </span>
              </div>
            </div>
          </div>

          {/* Right: Actions (Ingest, History, Export, Share, Manual Save) */}
          <div className="flex items-center gap-2 flex-wrap self-end sm:self-center">
            {/* Version History */}
            <button
              onClick={() => setIsHistoryOpen(true)}
              className="px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition-colors"
              title="View revision history"
            >
              <History className="w-3.5 h-3.5 text-violet-400" />
              <span className="hidden md:inline">History</span>
            </button>

            {/* Ingest / File Upload */}
            {!isReadOnly && (
              <button
                onClick={() => setIsUploadOpen(true)}
                className="px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition-colors"
                title="Import .md, .docx, .txt"
              >
                <UploadCloud className="w-3.5 h-3.5 text-cyan-400" />
                <span className="hidden md:inline">Import</span>
              </button>
            )}

            {/* Export Menu */}
            <div className="relative">
              <button
                onClick={() => setExportMenuOpen(!exportMenuOpen)}
                className="px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition-colors"
              >
                <Download className="w-3.5 h-3.5 text-emerald-400" />
                <span className="hidden md:inline">Export</span>
              </button>

              {exportMenuOpen && (
                <div className="absolute right-0 mt-1 w-44 rounded-xl bg-slate-900 border border-slate-700 shadow-2xl z-50 p-1.5 space-y-1 text-xs animate-in fade-in zoom-in-95">
                  <button
                    onClick={() => exportDocument('md')}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-800 text-slate-200"
                  >
                    Markdown (.md)
                  </button>
                  <button
                    onClick={() => exportDocument('txt')}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-800 text-slate-200"
                  >
                    Plain Text (.txt)
                  </button>
                  <button
                    onClick={() => exportDocument('html')}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-slate-800 text-slate-200"
                  >
                    HTML Document (.html)
                  </button>
                </div>
              )}
            </div>

            {/* Share Button */}
            <button
              onClick={() => setIsShareOpen(true)}
              className="px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md shadow-indigo-600/20"
            >
              <Share2 className="w-3.5 h-3.5" />
              <span>Share</span>
              {document.collaborators.length > 0 && (
                <span className="bg-indigo-900 text-indigo-200 text-[10px] px-1.5 py-0.2 rounded-full">
                  {document.collaborators.length}
                </span>
              )}
            </button>

            {/* Manual Save (for editors) */}
            {!isReadOnly && (
              <button
                onClick={() => {
                  if (editor) persistChanges(editor.getHTML(), title);
                }}
                className="p-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
                title="Save now"
              >
                <Save className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Read-Only Banner if Viewer */}
      {isReadOnly && (
        <div className="bg-cyan-950/40 border-b border-cyan-800/40 px-4 py-2 text-center text-xs text-cyan-300 flex items-center justify-center gap-2">
          <Eye className="w-3.5 h-3.5 text-cyan-400" />
          <span>You are viewing this document in <strong>Read-Only</strong> mode. You cannot edit content unless granted Editor access by the owner.</span>
        </div>
      )}

      {/* Google Docs Inspired Formatting Toolbar */}
      {!isReadOnly && editor && (
        <div className="sticky top-[61px] z-20 border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm px-4 py-1.5 shadow-sm overflow-x-auto">
          <div className="max-w-4xl mx-auto flex items-center gap-1 justify-start sm:justify-center">
            {/* Undo / Redo */}
            <button
              onClick={() => editor.chain().focus().undo().run()}
              disabled={!editor.can().undo()}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 disabled:opacity-30"
              title="Undo"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().redo().run()}
              disabled={!editor.can().redo()}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 disabled:opacity-30"
              title="Redo"
            >
              <RotateCw className="w-4 h-4" />
            </button>

            <div className="w-px h-5 bg-slate-800 mx-1" />

            {/* Headings */}
            <button
              onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('heading', { level: 1 }) ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Heading 1"
            >
              <Heading1 className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('heading', { level: 2 }) ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Heading 2"
            >
              <Heading2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('heading', { level: 3 }) ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Heading 3"
            >
              <Heading3 className="w-4 h-4" />
            </button>

            <div className="w-px h-5 bg-slate-800 mx-1" />

            {/* Formats: Bold, Italic, Underline, Strike */}
            <button
              onClick={() => editor.chain().focus().toggleBold().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('bold') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Bold (Ctrl+B)"
            >
              <Bold className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleItalic().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('italic') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Italic (Ctrl+I)"
            >
              <Italic className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleUnderline().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('underline') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Underline (Ctrl+U)"
            >
              <UnderlineIcon className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleStrike().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('strike') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Strikethrough"
            >
              <Strikethrough className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleHighlight().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('highlight') ? 'bg-amber-500/30 text-amber-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Highlight Text"
            >
              <Highlighter className="w-4 h-4" />
            </button>

            <div className="w-px h-5 bg-slate-800 mx-1" />

            {/* Lists */}
            <button
              onClick={() => editor.chain().focus().toggleBulletList().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('bulletList') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Bulleted List"
            >
              <List className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleOrderedList().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('orderedList') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Numbered List"
            >
              <ListOrdered className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleBlockquote().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('blockquote') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Quote"
            >
              <Quote className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().toggleCodeBlock().run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive('codeBlock') ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Code Block"
            >
              <Code className="w-4 h-4" />
            </button>

            <div className="w-px h-5 bg-slate-800 mx-1" />

            {/* Alignment */}
            <button
              onClick={() => editor.chain().focus().setTextAlign('left').run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive({ textAlign: 'left' }) ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Align Left"
            >
              <AlignLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().setTextAlign('center').run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive({ textAlign: 'center' }) ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Align Center"
            >
              <AlignCenter className="w-4 h-4" />
            </button>
            <button
              onClick={() => editor.chain().focus().setTextAlign('right').run()}
              className={`p-1.5 rounded-lg transition-colors ${
                editor.isActive({ textAlign: 'right' }) ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-400 hover:bg-slate-800'
              }`}
              title="Align Right"
            >
              <AlignRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Editor Main Canvas */}
      <main className="flex-1 py-8 px-4 sm:px-6 flex justify-center">
        <div className="w-full max-w-4xl bg-slate-900/70 border border-slate-800 rounded-2xl p-6 sm:p-12 shadow-2xl min-h-[700px] flex flex-col justify-between">
          <div className="prose prose-invert prose-indigo max-w-none focus:outline-none">
            <EditorContent editor={editor} className="min-h-[500px]" />
          </div>

          {/* Bottom Stats Footer */}
          <div className="pt-8 mt-8 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-3">
            <div className="flex items-center gap-4">
              <span><strong>{wordCount}</strong> words</span>
              <span><strong>{charCount}</strong> characters</span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-500">Document ID: <code className="font-mono text-slate-400">{document.id}</code></span>
            </div>
          </div>
        </div>
      </main>

      {/* Modals */}
      <ShareModal
        document={document}
        isOpen={isShareOpen}
        onClose={() => setIsShareOpen(false)}
        onUpdateDocument={(updated) => setDocument(updated)}
      />

      <FileUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        targetDocId={document.id}
        onImportContent={handleImportContent}
      />

      <RevisionHistoryModal
        document={document}
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onRestoreRevision={handleRestoreRevision}
        isReadOnly={isReadOnly}
      />
    </div>
  );
}
