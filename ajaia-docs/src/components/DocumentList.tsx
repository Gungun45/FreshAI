'use client';

import React, { useEffect, useState } from 'react';
import { Document } from '@/lib/types';
import { useAuth } from '@/context/AuthContext';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Plus,
  UploadCloud,
  FileText,
  Clock,
  Users,
  Trash2,
  Share2,
  Lock,
  Edit3,
  Eye,
  Search,
  Sparkles,
  ArrowUpDown,
  Filter,
} from 'lucide-react';
import { ShareModal } from './ShareModal';
import { FileUploadModal } from './FileUploadModal';

export function DocumentList() {
  const { currentUser } = useAuth();
  const router = useRouter();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [activeTab, setActiveTab] = useState<'all' | 'owned' | 'shared'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);

  // Modals
  const [selectedDocForShare, setSelectedDocForShare] = useState<Document | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const fetchDocuments = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`/api/documents?userId=${currentUser.id}&filter=${activeTab}`);
      const data = await res.json();
      if (data.success) {
        setDocuments(data.documents);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [currentUser.id, activeTab]);

  const handleCreateNew = async () => {
    setIsCreating(true);
    try {
      const res = await fetch('/api/documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'Untitled Document',
          ownerId: currentUser.id,
        }),
      });

      const data = await res.json();
      if (data.success && data.document) {
        router.push(`/doc/${data.document.id}`);
      }
    } catch (err) {
      console.error('Failed to create new document:', err);
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (docId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
      const res = await fetch(`/api/documents/${docId}?userId=${currentUser.id}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (data.success) {
        setDocuments((prev) => prev.filter((d) => d.id !== docId));
      } else {
        alert(data.error || 'Failed to delete');
      }
    } catch (err) {
      console.error('Delete error:', err);
    }
  };

  // Filtered by search
  const filteredDocs = documents.filter(
    (doc) =>
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (doc.ownerName && doc.ownerName.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="space-y-8">
      {/* Hero Welcome & Quick Action Bar */}
      <div className="bg-gradient-to-r from-indigo-950/60 via-slate-900 to-slate-900 border border-indigo-900/40 rounded-3xl p-6 sm:p-8 shadow-xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-semibold text-indigo-400">
              <Sparkles className="w-3.5 h-3.5" /> Logged in as {currentUser.name}
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
              Collaborative Document Studio
            </h1>
            <p className="text-sm text-slate-400 max-w-xl">
              Create rich formatted documents, ingest files (.md, .docx, .txt), and collaborate with team members using granular access permissions.
            </p>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              onClick={() => setIsUploadOpen(true)}
              className="flex-1 sm:flex-initial px-4 py-2.5 rounded-xl border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-semibold flex items-center justify-center gap-2 transition-colors shadow-sm"
            >
              <UploadCloud className="w-4 h-4 text-cyan-400" />
              <span>Import File</span>
            </button>

            <button
              disabled={isCreating}
              onClick={handleCreateNew}
              className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-semibold flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-600/20"
            >
              <Plus className="w-4 h-4" />
              <span>{isCreating ? 'Creating...' : 'New Document'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Filter Tabs & Search Header */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
        {/* Navigation Tabs */}
        <div className="flex items-center p-1 bg-slate-900 border border-slate-800 rounded-xl">
          <button
            onClick={() => setActiveTab('all')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'all'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All Docs
          </button>
          <button
            onClick={() => setActiveTab('owned')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'owned'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <span>Owned by Me</span>
          </button>
          <button
            onClick={() => setActiveTab('shared')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'shared'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Users className="w-3.5 h-3.5" />
            <span>Shared with Me</span>
          </button>
        </div>

        {/* Search Filter */}
        <div className="relative flex-1 max-w-xs">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search documents or authors..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
      </div>

      {/* Documents Grid */}
      {isLoading ? (
        <div className="py-20 text-center text-slate-400 text-sm">Loading documents...</div>
      ) : filteredDocs.length === 0 ? (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl py-16 px-4 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-slate-800 text-slate-400 flex items-center justify-center mx-auto">
            <FileText className="w-6 h-6" />
          </div>
          <div className="space-y-1 max-w-sm mx-auto">
            <p className="text-base font-semibold text-slate-200">No documents found</p>
            <p className="text-xs text-slate-400">
              {searchQuery
                ? 'Try searching with a different term.'
                : activeTab === 'shared'
                ? 'No documents have been shared with this user yet. Switch user or invite collaborators!'
                : 'Get started by creating a new document or importing an existing file.'}
            </p>
          </div>
          <button
            onClick={handleCreateNew}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold transition-colors"
          >
            Create New Document
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredDocs.map((doc) => {
            const isOwner = doc.ownerId === currentUser.id;
            const collab = doc.collaborators.find((c) => c.userId === currentUser.id);
            const formattedDate = new Date(doc.updatedAt).toLocaleDateString(undefined, {
              month: 'short',
              day: 'numeric',
            });

            return (
              <div
                key={doc.id}
                className="group relative bg-slate-900/70 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-2xl p-5 shadow-md hover:shadow-xl transition-all flex flex-col justify-between"
              >
                {/* Card Top */}
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="w-9 h-9 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center justify-center flex-shrink-0 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                      <FileText className="w-5 h-5" />
                    </div>

                    {/* Ownership / Permission Pill */}
                    <div>
                      {isOwner ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded-full">
                          <Sparkles className="w-2.5 h-2.5" /> Owner
                        </span>
                      ) : (
                        <span
                          className={`inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full border ${
                            collab?.role === 'editor'
                              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                              : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                          }`}
                        >
                          {collab?.role === 'editor' ? <Edit3 className="w-2.5 h-2.5" /> : <Eye className="w-2.5 h-2.5" />}
                          {collab?.role ? collab.role.toUpperCase() : 'SHARED'}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Title & Preview */}
                  <div>
                    <Link href={`/doc/${doc.id}`} className="block focus:outline-none">
                      <h3 className="font-semibold text-slate-100 text-base line-clamp-1 group-hover:text-indigo-300 transition-colors">
                        {doc.title}
                      </h3>
                      <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                        {doc.content.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim() || 'No preview available'}
                      </p>
                    </Link>
                  </div>
                </div>

                {/* Card Bottom / Metadata & Actions */}
                <div className="pt-4 mt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5 text-slate-500" />
                      {formattedDate}
                    </span>
                    <span>{doc.wordCount || 0} words</span>
                  </div>

                  <div className="flex items-center gap-1">
                    {/* Share trigger */}
                    <button
                      onClick={() => setSelectedDocForShare(doc)}
                      title="Share document"
                      className="p-1.5 rounded-lg text-slate-400 hover:text-indigo-300 hover:bg-slate-800 transition-colors"
                    >
                      <Share2 className="w-4 h-4" />
                    </button>

                    {/* Delete trigger (owner only) */}
                    {isOwner && (
                      <button
                        onClick={(e) => handleDelete(doc.id, e)}
                        title="Delete document"
                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Share Modal */}
      {selectedDocForShare && (
        <ShareModal
          document={selectedDocForShare}
          isOpen={Boolean(selectedDocForShare)}
          onClose={() => setSelectedDocForShare(null)}
          onUpdateDocument={(updated) => {
            setDocuments((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
            setSelectedDocForShare(updated);
          }}
        />
      )}

      {/* File Upload Modal */}
      <FileUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onDocCreated={() => fetchDocuments()}
      />
    </div>
  );
}
