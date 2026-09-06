'use client';

import React from 'react';
import { Document, DocumentRevision } from '@/lib/types';
import { X, History, RotateCcw, Clock, Check } from 'lucide-react';

interface RevisionHistoryModalProps {
  document: Document;
  isOpen: boolean;
  onClose: () => void;
  onRestoreRevision: (revision: DocumentRevision) => void;
  isReadOnly: boolean;
}

export function RevisionHistoryModal({
  document,
  isOpen,
  onClose,
  onRestoreRevision,
  isReadOnly,
}: RevisionHistoryModalProps) {
  if (!isOpen) return null;

  const revisions = document.revisions || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-5 text-slate-100 max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 flex-shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-violet-500/20 text-violet-400 flex items-center justify-center">
              <History className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-semibold text-base text-slate-100">Version History</h3>
              <p className="text-xs text-slate-400">View and restore past saved versions of this document</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content list */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          {revisions.length === 0 ? (
            <div className="py-12 text-center text-slate-400 text-xs">
              No previous revisions saved yet. Changes are automatically snapshotted on save.
            </div>
          ) : (
            <div className="relative border-l-2 border-slate-800 ml-4 pl-4 space-y-4 my-2">
              {revisions
                .slice()
                .reverse()
                .map((rev, index) => {
                  const isLatest = index === 0;
                  const formattedTime = new Date(rev.timestamp).toLocaleString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  });

                  return (
                    <div key={rev.id} className="relative group">
                      {/* Timeline dot */}
                      <div
                        className={`absolute -left-[23px] top-1.5 w-3 h-3 rounded-full border-2 border-slate-900 ${
                          isLatest ? 'bg-emerald-400 ring-2 ring-emerald-400/30' : 'bg-slate-600'
                        }`}
                      />

                      <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-slate-700 transition-colors space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-slate-200">{rev.title}</span>
                            {isLatest && (
                              <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full font-medium flex items-center gap-1">
                                <Check className="w-2.5 h-2.5" /> Current
                              </span>
                            )}
                          </div>

                          {!isReadOnly && !isLatest && (
                            <button
                              onClick={() => onRestoreRevision(rev)}
                              className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/30 text-xs font-medium transition-colors"
                            >
                              <RotateCcw className="w-3 h-3" />
                              Restore
                            </button>
                          )}
                        </div>

                        <p className="text-xs text-slate-400 italic">&quot;{rev.summary || 'Content updated'}&quot;</p>

                        <div className="flex items-center gap-4 text-[11px] text-slate-500">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formattedTime}
                          </span>
                          <span>Saved by: <strong className="text-slate-400 font-medium">{rev.savedByName}</strong></span>
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="pt-3 border-t border-slate-800 flex justify-end flex-shrink-0">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
