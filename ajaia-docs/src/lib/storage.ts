import fs from 'fs';
import path from 'path';
import { Document, DocumentRevision, UserRole } from './types';
import { findUserById, findUserByEmail, SEEDED_USERS } from './users';

const DATA_DIR = path.join(process.cwd(), 'data');
const DATA_FILE = path.join(DATA_DIR, 'documents.json');

// Ensure data directory and file exists with initial seed
function initializeStorage(): Document[] {
  try {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }

    if (!fs.existsSync(DATA_FILE)) {
      const initialDocs: Document[] = [
        {
          id: 'doc-seed-1',
          title: 'Product Requirements: AI Collaborative Workspace',
          content: `<h1>Product Requirements: AI Collaborative Workspace</h1><p>Welcome to <strong>Ajaia Docs</strong>, a high-performance, real-time collaborative workspace engineered for modern product teams.</p><h2>1. Executive Summary</h2><p>This product bridges rich-text authoring, multi-party collaboration, and file ingestion with frictionless permission sharing.</p><h3>Key Value Propositions</h3><ul><li><strong>Instant Rich-Text Authoring:</strong> Headings, rich formatting, quotes, and structured lists.</li><li><strong>Frictionless File Ingestion:</strong> Import Markdown (.md), Plain Text (.txt), and Word (.docx) directly into editable drafts.</li><li><strong>Role-Based Access Control:</strong> Seamlessly grant <em>Editor</em> or <em>Viewer</em> permissions across cross-functional teams.</li></ul><blockquote>"A focused, well-reasoned solution is better than an overextended build." — Ajaia Engineering Principles</blockquote><p>Feel free to edit this document, invite teammates, or upload existing notes.</p>`,
          ownerId: 'user_vaibhavi',
          ownerName: 'Vaibhavi Diwakar',
          ownerEmail: 'taniyadiwaker6@gmail.com',
          collaborators: [
            {
              userId: 'user_alex',
              email: 'alex.chen@ajaia.io',
              name: 'Alex Chen',
              role: 'editor',
              addedAt: new Date(Date.now() - 3600000 * 24).toISOString(),
            },
            {
              userId: 'user_sarah',
              email: 'sarah.miller@ajaia.io',
              name: 'Sarah Miller',
              role: 'viewer',
              addedAt: new Date(Date.now() - 3600000 * 12).toISOString(),
            },
          ],
          createdAt: new Date(Date.now() - 3600000 * 48).toISOString(),
          updatedAt: new Date().toISOString(),
          wordCount: 110,
          characterCount: 820,
          revisions: [
            {
              id: 'rev-1',
              timestamp: new Date(Date.now() - 3600000 * 24).toISOString(),
              title: 'Product Requirements: AI Collaborative Workspace',
              content: '<h1>Product Requirements</h1><p>Initial draft outlining product features.</p>',
              savedByUserId: 'user_vaibhavi',
              savedByName: 'Vaibhavi Diwakar',
              summary: 'Initial document creation',
            },
          ],
        },
        {
          id: 'doc-seed-2',
          title: 'Ajaia Architecture & System RFC #104',
          content: `<h1>Ajaia Architecture &amp; System RFC #104</h1><p><strong>Author:</strong> Alex Chen (Head of Product)<br/><strong>Status:</strong> Under Review</p><h2>1. Architecture Overview</h2><p>We leverage a modern full-stack TypeScript architecture using Next.js App Router, atomic file persistence, and client-side rich text state management.</p><h3>Storage &amp; Data Consistency</h3><p>Persistence is managed via structured atomic file I/O with JSON serialization, providing durability across reboots without external operational dependencies.</p><h3>Permissions Model</h3><ol><li><strong>Owner:</strong> Full rights (edit, rename, delete, manage permissions).</li><li><strong>Editor:</strong> Read and write content modifications.</li><li><strong>Viewer:</strong> Read-only access with interface controls locked.</li></ol>`,
          ownerId: 'user_alex',
          ownerName: 'Alex Chen',
          ownerEmail: 'alex.chen@ajaia.io',
          collaborators: [
            {
              userId: 'user_vaibhavi',
              email: 'taniyadiwaker6@gmail.com',
              name: 'Vaibhavi Diwakar',
              role: 'editor',
              addedAt: new Date(Date.now() - 3600000 * 6).toISOString(),
            },
            {
              userId: 'user_jordan',
              email: 'jordan.taylor@ajaia.io',
              name: 'Jordan Taylor',
              role: 'editor',
              addedAt: new Date(Date.now() - 3600000 * 4).toISOString(),
            },
          ],
          createdAt: new Date(Date.now() - 3600000 * 24).toISOString(),
          updatedAt: new Date(Date.now() - 3600000 * 2).toISOString(),
          wordCount: 88,
          characterCount: 650,
        },
        {
          id: 'doc-seed-3',
          title: 'Q3 Product Roadmap & Sprint Backlog',
          content: `<h1>Q3 Product Roadmap &amp; Sprint Backlog</h1><p>Prepared by Jordan Taylor for platform engineering alignment.</p><h2>Milestone Deliverables</h2><ul><li>Document Ingestion Engine (.docx, .md, .txt)</li><li>Live Presence &amp; Real-time active editing simulation</li><li>Granular Role-based Access Sharing</li><li>Automated Continuous Integration &amp; Test Suite</li></ul>`,
          ownerId: 'user_jordan',
          ownerName: 'Jordan Taylor',
          ownerEmail: 'jordan.taylor@ajaia.io',
          collaborators: [
            {
              userId: 'user_vaibhavi',
              email: 'taniyadiwaker6@gmail.com',
              name: 'Vaibhavi Diwakar',
              role: 'viewer',
              addedAt: new Date().toISOString(),
            },
          ],
          createdAt: new Date(Date.now() - 3600000 * 10).toISOString(),
          updatedAt: new Date(Date.now() - 3600000 * 1).toISOString(),
          wordCount: 42,
          characterCount: 310,
        },
      ];

      fs.writeFileSync(DATA_FILE, JSON.stringify(initialDocs, null, 2), 'utf-8');
      return initialDocs;
    }

    const fileContent = fs.readFileSync(DATA_FILE, 'utf-8');
    return JSON.parse(fileContent);
  } catch (error) {
    console.error('Failed to read or initialize storage:', error);
    return [];
  }
}

function saveDocuments(docs: Document[]): void {
  try {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }
    const tempFile = `${DATA_FILE}.tmp.${Date.now()}`;
    fs.writeFileSync(tempFile, JSON.stringify(docs, null, 2), 'utf-8');
    fs.renameSync(tempFile, DATA_FILE);
  } catch (error) {
    console.error('Failed to save documents:', error);
    // Fallback direct write
    fs.writeFileSync(DATA_FILE, JSON.stringify(docs, null, 2), 'utf-8');
  }
}

function countWords(htmlContent: string): { words: number; chars: number } {
  const plainText = htmlContent.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  const words = plainText ? plainText.split(/\s+/).length : 0;
  const chars = plainText.length;
  return { words, chars };
}

export function getUserRoleForDocument(doc: Document, userId: string): UserRole | null {
  if (doc.ownerId === userId) return 'owner';
  const collab = doc.collaborators.find((c) => c.userId === userId);
  if (collab) return collab.role;
  return null;
}

export const Storage = {
  list(userId?: string, filter: 'all' | 'owned' | 'shared' = 'all'): Document[] {
    const docs = initializeStorage();
    if (!userId) return docs;

    return docs.filter((doc) => {
      const isOwner = doc.ownerId === userId;
      const isCollaborator = doc.collaborators.some((c) => c.userId === userId);

      if (filter === 'owned') return isOwner;
      if (filter === 'shared') return isCollaborator && !isOwner;
      return isOwner || isCollaborator;
    });
  },

  getById(id: string): Document | null {
    const docs = initializeStorage();
    return docs.find((d) => d.id === id) || null;
  },

  create(data: { title?: string; content?: string; ownerId: string }): Document {
    const docs = initializeStorage();
    const owner = findUserById(data.ownerId) || SEEDED_USERS[0];
    const initialContent = data.content || '<p>Start typing your document here...</p>';
    const counts = countWords(initialContent);

    const newDoc: Document = {
      id: `doc-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      title: data.title?.trim() || 'Untitled Document',
      content: initialContent,
      ownerId: owner.id,
      ownerName: owner.name,
      ownerEmail: owner.email,
      collaborators: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      wordCount: counts.words,
      characterCount: counts.chars,
      revisions: [
        {
          id: `rev-${Date.now()}`,
          timestamp: new Date().toISOString(),
          title: data.title?.trim() || 'Untitled Document',
          content: initialContent,
          savedByUserId: owner.id,
          savedByName: owner.name,
          summary: 'Document created',
        },
      ],
    };

    docs.unshift(newDoc);
    saveDocuments(docs);
    return newDoc;
  },

  update(id: string, data: { title?: string; content?: string; userId: string }): Document | null {
    const docs = initializeStorage();
    const index = docs.findIndex((d) => d.id === id);
    if (index === -1) return null;

    const doc = docs[index];
    const userRole = getUserRoleForDocument(doc, data.userId);

    if (!userRole || userRole === 'viewer') {
      throw new Error('Unauthorized: Viewer permissions do not allow modifying document content.');
    }

    const updater = findUserById(data.userId) || SEEDED_USERS[0];
    const updatedContent = data.content !== undefined ? data.content : doc.content;
    const updatedTitle = data.title !== undefined ? data.title.trim() : doc.title;
    const counts = countWords(updatedContent);

    const revisions = doc.revisions || [];
    // Append revision snapshot
    revisions.push({
      id: `rev-${Date.now()}`,
      timestamp: new Date().toISOString(),
      title: updatedTitle,
      content: updatedContent,
      savedByUserId: updater.id,
      savedByName: updater.name,
      summary: data.title && data.title !== doc.title ? 'Renamed document & updated content' : 'Saved content edits',
    });

    // Limit to latest 20 revisions
    const trimmedRevisions = revisions.slice(-20);

    const updatedDoc: Document = {
      ...doc,
      title: updatedTitle,
      content: updatedContent,
      updatedAt: new Date().toISOString(),
      wordCount: counts.words,
      characterCount: counts.chars,
      revisions: trimmedRevisions,
    };

    docs[index] = updatedDoc;
    saveDocuments(docs);
    return updatedDoc;
  },

  delete(id: string, requestingUserId: string): boolean {
    const docs = initializeStorage();
    const doc = docs.find((d) => d.id === id);
    if (!doc) return false;

    if (doc.ownerId !== requestingUserId) {
      throw new Error('Unauthorized: Only the document owner can delete this document.');
    }

    const filtered = docs.filter((d) => d.id !== id);
    saveDocuments(filtered);
    return true;
  },

  share(docId: string, requestingUserId: string, targetEmail: string, role: 'editor' | 'viewer'): Document {
    const docs = initializeStorage();
    const index = docs.findIndex((d) => d.id === docId);
    if (index === -1) throw new Error('Document not found');

    const doc = docs[index];
    const requestRole = getUserRoleForDocument(doc, requestingUserId);

    if (requestRole !== 'owner') {
      throw new Error('Unauthorized: Only document owners can manage sharing permissions.');
    }

    const targetUser = findUserByEmail(targetEmail);
    const collaboratorName = targetUser ? targetUser.name : targetEmail.split('@')[0];
    const collaboratorUserId = targetUser ? targetUser.id : `guest_${targetEmail.replace(/[^a-zA-Z0-9]/g, '_')}`;

    if (collaboratorUserId === doc.ownerId) {
      throw new Error('Cannot share with the document owner.');
    }

    // Check if already in collaborators
    const existingIndex = doc.collaborators.findIndex((c) => c.email.toLowerCase() === targetEmail.toLowerCase() || c.userId === collaboratorUserId);

    if (existingIndex >= 0) {
      doc.collaborators[existingIndex].role = role;
    } else {
      doc.collaborators.push({
        userId: collaboratorUserId,
        email: targetEmail,
        name: collaboratorName,
        role: role,
        addedAt: new Date().toISOString(),
      });
    }

    doc.updatedAt = new Date().toISOString();
    docs[index] = doc;
    saveDocuments(docs);
    return doc;
  },

  removeCollaborator(docId: string, requestingUserId: string, targetUserId: string): Document {
    const docs = initializeStorage();
    const index = docs.findIndex((d) => d.id === docId);
    if (index === -1) throw new Error('Document not found');

    const doc = docs[index];
    if (doc.ownerId !== requestingUserId) {
      throw new Error('Unauthorized: Only document owners can remove collaborators.');
    }

    doc.collaborators = doc.collaborators.filter((c) => c.userId !== targetUserId && c.email !== targetUserId);
    doc.updatedAt = new Date().toISOString();
    docs[index] = doc;
    saveDocuments(docs);
    return doc;
  },
};
