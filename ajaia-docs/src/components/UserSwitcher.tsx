'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import { ChevronDown, Check, UserCheck } from 'lucide-react';

export function UserSwitcher() {
  const { currentUser, allUsers, switchUser } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative inline-block text-left" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-900 hover:bg-slate-800 transition-colors text-sm font-medium text-slate-200 shadow-sm"
        title="Switch simulated user to test permissions"
      >
        <div
          className={`w-6 h-6 rounded-full ${currentUser.avatarColor} text-white flex items-center justify-center text-xs font-bold shadow-inner`}
        >
          {currentUser.initials}
        </div>
        <div className="flex flex-col text-left leading-tight hidden sm:block">
          <span className="font-semibold text-xs text-slate-100">{currentUser.name}</span>
          <span className="text-[10px] text-slate-400">{currentUser.role.split('(')[0]}</span>
        </div>
        <ChevronDown className="w-3.5 h-3.5 text-slate-400 ml-1" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-72 origin-top-right rounded-xl bg-slate-900 border border-slate-700 shadow-2xl z-50 p-2 divide-y divide-slate-800 animate-in fade-in zoom-in-95 duration-100">
          <div className="px-3 py-2 text-xs text-slate-400 font-semibold uppercase tracking-wider flex items-center justify-between">
            <span>Simulate User Role</span>
            <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded font-mono">Mock Auth</span>
          </div>
          <div className="py-1 space-y-1">
            {allUsers.map((user) => {
              const isSelected = user.id === currentUser.id;
              return (
                <button
                  key={user.id}
                  onClick={() => {
                    switchUser(user.id);
                    setIsOpen(false);
                  }}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left text-sm transition-colors ${
                    isSelected ? 'bg-indigo-950/60 text-indigo-200 border border-indigo-700/50' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <div
                      className={`w-7 h-7 rounded-full ${user.avatarColor} text-white flex items-center justify-center text-xs font-bold`}
                    >
                      {user.initials}
                    </div>
                    <div>
                      <p className="font-medium text-xs text-slate-100">{user.name}</p>
                      <p className="text-[11px] text-slate-400">{user.role}</p>
                    </div>
                  </div>
                  {isSelected && <Check className="w-4 h-4 text-indigo-400 flex-shrink-0" />}
                </button>
              );
            })}
          </div>
          <div className="px-3 py-2 text-[11px] text-slate-400 flex items-center gap-1.5">
            <UserCheck className="w-3.5 h-3.5 text-indigo-400" />
            <span>Switch anytime to test Owner vs Editor vs Viewer flows</span>
          </div>
        </div>
      )}
    </div>
  );
}
