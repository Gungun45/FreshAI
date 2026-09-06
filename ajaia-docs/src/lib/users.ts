import { User } from './types';

export const SEEDED_USERS: User[] = [
  {
    id: 'user_vaibhavi',
    name: 'Vaibhavi Diwakar',
    email: 'taniyadiwaker6@gmail.com',
    role: 'Lead Product Engineer (Candidate)',
    avatarColor: 'bg-indigo-600',
    initials: 'VD',
  },
  {
    id: 'user_alex',
    name: 'Alex Chen',
    email: 'alex.chen@ajaia.io',
    role: 'Head of Product',
    avatarColor: 'bg-emerald-600',
    initials: 'AC',
  },
  {
    id: 'user_jordan',
    name: 'Jordan Taylor',
    email: 'jordan.taylor@ajaia.io',
    role: 'Staff Platform Engineer',
    avatarColor: 'bg-violet-600',
    initials: 'JT',
  },
  {
    id: 'user_sarah',
    name: 'Sarah Miller',
    email: 'sarah.miller@ajaia.io',
    role: 'Design Director',
    avatarColor: 'bg-amber-600',
    initials: 'SM',
  },
];

export function findUserById(id: string): User | undefined {
  return SEEDED_USERS.find((u) => u.id === id);
}

export function findUserByEmail(email: string): User | undefined {
  const clean = email.trim().toLowerCase();
  return SEEDED_USERS.find((u) => u.email.toLowerCase() === clean);
}

export function getAllUsers(): User[] {
  return SEEDED_USERS;
}
