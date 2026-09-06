'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { User } from '@/lib/types';
import { SEEDED_USERS } from '@/lib/users';

interface AuthContextType {
  currentUser: User;
  allUsers: User[];
  switchUser: (userId: string) => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [currentUser, setCurrentUser] = useState<User>(SEEDED_USERS[0]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    try {
      const savedUserId = localStorage.getItem('ajaia_active_user_id');
      if (savedUserId) {
        const found = SEEDED_USERS.find((u) => u.id === savedUserId);
        if (found) setCurrentUser(found);
      }
    } catch {
      // Ignore localStorage errors
    } finally {
      setIsLoading(false);
    }
  }, []);

  const switchUser = (userId: string) => {
    const found = SEEDED_USERS.find((u) => u.id === userId);
    if (found) {
      setCurrentUser(found);
      try {
        localStorage.setItem('ajaia_active_user_id', userId);
      } catch {
        // Ignore
      }
    }
  };

  return (
    <AuthContext.Provider
      value={{
        currentUser,
        allUsers: SEEDED_USERS,
        switchUser,
        isLoading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
