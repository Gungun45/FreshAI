export type UserRole = 'owner' | 'editor' | 'viewer';

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  avatarColor: string;
  initials: string;
}

export interface Collaborator {
  userId: string;
  email: string;
  name: string;
  role: 'editor' | 'viewer';
  addedAt: string;
}

export interface DocumentRevision {
  id: string;
  timestamp: string;
  title: string;
  content: string;
  savedByUserId: string;
  savedByName: string;
  summary: string;
}

export interface Document {
  id: string;
  title: string;
  content: string;
  ownerId: string;
  ownerName?: string;
  ownerEmail?: string;
  collaborators: Collaborator[];
  createdAt: string;
  updatedAt: string;
  wordCount?: number;
  characterCount?: number;
  revisions?: DocumentRevision[];
}

export interface CreateDocumentDTO {
  title?: string;
  content?: string;
  ownerId: string;
}

export interface UpdateDocumentDTO {
  title?: string;
  content?: string;
  userId: string;
}

export interface ShareDocumentDTO {
  documentId: string;
  requestingUserId: string;
  targetEmail: string;
  targetRole: 'editor' | 'viewer';
}

export interface FileUploadResponse {
  success: boolean;
  filename: string;
  title: string;
  content: string;
  format: 'markdown' | 'text' | 'docx' | 'html' | 'json';
  error?: string;
}
