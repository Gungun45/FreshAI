'use client';

import React from 'react';
import Link from 'next/link';
import { UserSwitcher } from './UserSwitcher';
import { FileText, Sparkles } from 'lucide-react';

interface NavbarProps {
  onNewDoc?: () => void;
  onOpenFileUpload?: () => void;
  showActions?: boolean;
}

export function Navbar({ onNewDoc, onOpenFileUpload, showActions = true }: NavbarProps) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Logo & Brand */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:scale-105 transition-transform">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-slate-100 tracking-tight">Ajaia Docs</span>
                <span className="inline-flex items-center gap-1 text-[11px] font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded-full">
                  <Sparkles className="w-3 h-3" /> AI-Native
                </span>
              </div>
              <span className="text-[11px] text-slate-400 hidden sm:inline">Collaborative Document Studio</span>
            </div>
          </Link>
        </div>

        {/* User Simulation Switcher */}
        <div className="flex items-center gap-3">
          <UserSwitcher />
        </div>
      </div>
    </header>
  );
}
