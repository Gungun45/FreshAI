'use client';

import React, { useState, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { X, UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface FileUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  targetDocId?: string; // If provided, imports into existing doc
  onImportContent?: (importedHtml: string) => void;
  onDocCreated?: () => void;
}

export function FileUploadModal({
  isOpen,
  onClose,
  targetDocId,
  onImportContent,
  onDocCreated,
}: FileUploadModalProps) {
  const { currentUser } = useAuth();
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [mode, setMode] = useState<'new_doc' | 'insert'>(targetDocId ? 'insert' : 'new_doc');
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      const validExtensions = ['.txt', '.md', '.markdown', '.docx', '.json'];
      const hasValidExt = validExtensions.some((ext) => selected.name.toLowerCase().endsWith(ext));

      if (!hasValidExt) {
        setErrorMsg('Unsupported format. Please select a .txt, .md, .docx, or .json file.');
        setFile(null);
        return;
      }

      if (selected.size > 5 * 1024 * 1024) {
        setErrorMsg('File is too large. Maximum supported size is 5MB.');
        setFile(null);
        return;
      }

      setErrorMsg('');
      setFile(selected);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setErrorMsg('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('ownerId', currentUser.id);
      formData.append('createNewDoc', mode === 'new_doc' ? 'true' : 'false');

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to upload and parse file');
      }

      if (mode === 'new_doc' && data.createdDocument) {
        onClose();
        if (onDocCreated) onDocCreated();
        router.push(`/doc/${data.createdDocument.id}`);
      } else if (mode === 'insert' && onImportContent) {
        onImportContent(data.content);
        onClose();
      } else {
        onClose();
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error uploading file');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
              <UploadCloud className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-semibold text-base text-slate-100">Import &amp; Ingest File</h3>
              <p className="text-xs text-slate-400">Convert local files into rich-text documents</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Supported Types Info Banner */}
        <div className="p-3.5 bg-slate-950/80 border border-slate-800 rounded-xl space-y-1 text-xs">
          <div className="flex items-center gap-1.5 font-medium text-slate-200">
            <FileText className="w-3.5 h-3.5 text-cyan-400" />
            <span>Supported File Formats:</span>
          </div>
          <p className="text-slate-400 text-[11px]">
            <code className="text-cyan-300 font-mono">.md / .markdown</code> (Markdown with headings, lists, quotes),{' '}
            <code className="text-cyan-300 font-mono">.docx</code> (Microsoft Word documents),{' '}
            <code className="text-cyan-300 font-mono">.txt</code> (Plain text),{' '}
            <code className="text-cyan-300 font-mono">.json</code> (Structured documents). Max size: 5MB.
          </p>
        </div>

        {/* Action Destination selector */}
        {targetDocId && (
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-slate-300">Import Destination</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setMode('insert')}
                className={`px-3 py-2 rounded-xl text-xs font-medium border text-center transition-all ${
                  mode === 'insert'
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                Append into Active Draft
              </button>
              <button
                type="button"
                onClick={() => setMode('new_doc')}
                className={`px-3 py-2 rounded-xl text-xs font-medium border text-center transition-all ${
                  mode === 'new_doc'
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                Create New Document
              </button>
            </div>
          </div>
        )}

        {/* Dropzone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${
            file
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-slate-700 hover:border-indigo-500/60 bg-slate-950/40 hover:bg-slate-950/80'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.markdown,.docx,.json"
            onChange={handleFileChange}
            className="hidden"
          />

          {file ? (
            <div className="space-y-2 flex flex-col items-center">
              <CheckCircle2 className="w-10 h-10 text-emerald-400 animate-in zoom-in-50" />
              <div>
                <p className="text-sm font-semibold text-slate-100">{file.name}</p>
                <p className="text-xs text-slate-400">{(file.size / 1024).toFixed(1)} KB — Ready to parse</p>
              </div>
              <span className="text-[11px] text-indigo-400 underline">Click to choose another file</span>
            </div>
          ) : (
            <div className="space-y-2 flex flex-col items-center">
              <UploadCloud className="w-10 h-10 text-slate-400" />
              <div>
                <p className="text-sm font-medium text-slate-200">Click or drag &amp; drop file here</p>
                <p className="text-xs text-slate-500">.md, .docx, .txt, .json up to 5MB</p>
              </div>
            </div>
          )}
        </div>

        {errorMsg && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Actions */}
        <div className="pt-3 border-t border-slate-800 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            disabled={!file || isUploading}
            onClick={handleUpload}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-lg shadow-indigo-600/20"
          >
            {isUploading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            <span>{isUploading ? 'Parsing & Ingesting...' : mode === 'new_doc' ? 'Create Document' : 'Import Content'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
