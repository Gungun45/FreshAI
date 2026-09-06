'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { RichEditor } from '@/components/RichEditor';
import { Document, UserRole } from '@/lib/types';
import Link from 'next/link';
import { AlertCircle, FileText, Loader2 } from 'lucide-react';

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { currentUser } = useAuth();
  const [document, setDocument] = useState<Document | null>(null);
  const [userRole, setUserRole] = useState<UserRole | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const docId = params?.id as string;

  useEffect(() => {
    if (!docId) return;

    const fetchDoc = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`/api/documents/${docId}?userId=${currentUser.id}`);
        const data = await res.json();

        if (!res.ok || !data.success) {
          throw new Error(data.error || 'Failed to load document');
        }

        setDocument(data.document);
        setUserRole(data.userRole);
      } catch (err: any) {
        setError(err.message || 'Could not load document');
      } finally {
        setLoading(false);
      }
    };

    fetchDoc();
  }, [docId, currentUser.id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-400 space-y-3">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
        <p className="text-sm font-medium">Opening document workspace...</p>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 max-w-md w-full text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-rose-500/10 text-rose-400 flex items-center justify-center mx-auto">
            <AlertCircle className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-slate-100">Document Unavailable</h2>
          <p className="text-xs text-slate-400">{error || 'This document does not exist or has been deleted.'}</p>
          <Link
            href="/"
            className="inline-block px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold transition-colors"
          >
            Return to Documents
          </Link>
        </div>
      </div>
    );
  }

  return <RichEditor initialDocument={document} userRole={userRole} />;
}
