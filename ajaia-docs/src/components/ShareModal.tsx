'use client';

import React, { useState } from 'react';
import { Document, User } from '@/lib/types';
import { useAuth } from '@/context/AuthContext';
import { X, Users, UserPlus, Trash2, Copy, Check, ShieldCheck, Eye, Edit3 } from 'lucide-react';

interface ShareModalProps {
  document: Document;
  isOpen: boolean;
  onClose: () => void;
  onUpdateDocument: (updatedDoc: Document) => void;
}

export function ShareModal({ document, isOpen, onClose, onUpdateDocument }: ShareModalProps) {
  const { currentUser, allUsers } = useAuth();
  const [emailInput, setEmailInput] = useState('');
  const [selectedRole, setSelectedRole] = useState<'editor' | 'viewer'>('editor');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [copiedLink, setCopiedLink] = useState(false);

  if (!isOpen) return null;

  const isOwner = document.ownerId === currentUser.id;

  const handleShare = async (emailToShare?: string, roleToShare?: 'editor' | 'viewer') => {
    const targetEmail = (emailToShare || emailInput).trim();
    const role = roleToShare || selectedRole;

    if (!targetEmail) {
      setErrorMsg('Please enter a valid email address');
      return;
    }

    setIsSubmitting(true);
    setErrorMsg('');

    try {
      const res = await fetch(`/api/documents/${document.id}/share`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requestingUserId: currentUser.id,
          targetEmail,
          targetRole: role,
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to update permissions');
      }

      onUpdateDocument(data.document);
      setEmailInput('');
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRemoveCollaborator = async (targetUserId: string) => {
    if (!isOwner) return;
    setIsSubmitting(true);
    setErrorMsg('');

    try {
      const res = await fetch(
        `/api/documents/${document.id}/share?requestingUserId=${currentUser.id}&targetUserId=${targetUserId}`,
        {
          method: 'DELETE',
        }
      );

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to remove collaborator');
      }

      onUpdateDocument(data.document);
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCopyLink = () => {
    if (typeof window !== 'undefined') {
      navigator.clipboard.writeText(window.location.href);
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 2000);
    }
  };

  // Quick suggestions: other seeded users not already shared
  const availableTeammates = allUsers.filter(
    (u) =>
      u.id !== document.ownerId &&
      !document.collaborators.some((c) => c.userId === u.id || c.email.toLowerCase() === u.email.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
              <Users className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-semibold text-base text-slate-100">Share &quot;{document.title}&quot;</h3>
              <p className="text-xs text-slate-400">Manage permissions and team access</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Warning if not owner */}
        {!isOwner && (
          <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-300 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>You have access to this document, but only the owner can modify sharing permissions.</span>
          </div>
        )}

        {/* Add collaborator form (only if owner) */}
        {isOwner && (
          <div className="space-y-3">
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Add Collaborator
            </label>
            <div className="flex gap-2">
              <input
                type="email"
                placeholder="Enter teammate's email..."
                value={emailInput}
                onChange={(e) => setEmailInput(e.target.value)}
                className="flex-1 px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-700 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
              <select
                value={selectedRole}
                onChange={(e) => setSelectedRole(e.target.value as 'editor' | 'viewer')}
                className="px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs font-medium text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="editor">Editor (Can edit)</option>
                <option value="viewer">Viewer (Read-only)</option>
              </select>
              <button
                disabled={isSubmitting || !emailInput.trim()}
                onClick={() => handleShare()}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm"
              >
                <UserPlus className="w-3.5 h-3.5" />
                Invite
              </button>
            </div>

            {/* Quick Add Team Pills */}
            {availableTeammates.length > 0 && (
              <div className="space-y-1.5">
                <span className="text-[11px] text-slate-400">Quick invite teammates:</span>
                <div className="flex flex-wrap gap-1.5">
                  {availableTeammates.map((u) => (
                    <button
                      key={u.id}
                      onClick={() => handleShare(u.email, 'editor')}
                      className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-300 flex items-center gap-1.5 transition-colors"
                    >
                      <span className={`w-2 h-2 rounded-full ${u.avatarColor}`} />
                      <span>{u.name}</span>
                      <span className="text-[10px] text-slate-400">(Editor)</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {errorMsg && <p className="text-xs text-rose-400 font-medium">{errorMsg}</p>}
          </div>
        )}

        {/* Collaborators List */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
            People with access ({document.collaborators.length + 1})
          </label>
          <div className="max-h-52 overflow-y-auto space-y-2 pr-1">
            {/* Owner */}
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center text-xs font-bold">
                  {document.ownerName ? document.ownerName.charAt(0) : 'O'}
                </div>
                <div>
                  <div className="flex items-center gap-1.5">
                    <p className="text-xs font-semibold text-slate-100">{document.ownerName || 'Owner'}</p>
                    <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.2 rounded font-mono">Owner</span>
                  </div>
                  <p className="text-[11px] text-slate-400">{document.ownerEmail || 'document-owner'}</p>
                </div>
              </div>
              <span className="text-xs font-medium text-slate-400">Full Access</span>
            </div>

            {/* Collaborators */}
            {document.collaborators.map((c) => (
              <div
                key={c.userId || c.email}
                className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950/40 border border-slate-800/80"
              >
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-200 flex items-center justify-center text-xs font-bold">
                    {c.name ? c.name.charAt(0) : c.email.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <p className="text-xs font-medium text-slate-100">{c.name || c.email}</p>
                    <p className="text-[11px] text-slate-400">{c.email}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isOwner ? (
                    <select
                      value={c.role}
                      onChange={(e) => handleShare(c.email, e.target.value as 'editor' | 'viewer')}
                      className="px-2 py-1 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none"
                    >
                      <option value="editor">Editor</option>
                      <option value="viewer">Viewer</option>
                    </select>
                  ) : (
                    <span className="text-xs font-medium text-slate-400 flex items-center gap-1 capitalize">
                      {c.role === 'editor' ? <Edit3 className="w-3 h-3 text-emerald-400" /> : <Eye className="w-3 h-3 text-cyan-400" />}
                      {c.role}
                    </span>
                  )}

                  {isOwner && (
                    <button
                      onClick={() => handleRemoveCollaborator(c.userId)}
                      title="Remove access"
                      className="p-1 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer / Copy Link */}
        <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
          <button
            onClick={handleCopyLink}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-700 bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition-colors"
          >
            {copiedLink ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedLink ? 'Link Copied!' : 'Copy Document Link'}</span>
          </button>

          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-medium transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
