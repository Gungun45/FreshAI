import { describe, it, expect, beforeEach } from 'vitest';
import { Storage, getUserRoleForDocument } from '../src/lib/storage';
import { markdownToHtml, plainTextToHtml } from '../src/lib/fileParsers';
import { SEEDED_USERS } from '../src/lib/users';

describe('Ajaia Collaborative Docs - Core Logic & Permissions', () => {
  const ownerUser = SEEDED_USERS[0]; // Vaibhavi Diwakar
  const editorUser = SEEDED_USERS[1]; // Alex Chen
  const viewerUser = SEEDED_USERS[3]; // Sarah Miller

  it('should initialize and list seeded documents', () => {
    const docs = Storage.list();
    expect(Array.isArray(docs)).toBe(true);
    expect(docs.length).toBeGreaterThanOrEqual(1);
  });

  it('should create a new document with correct owner and counts', () => {
    const newDoc = Storage.create({
      title: 'Automated Test Architecture RFC',
      content: '<h1>System Architecture</h1><p>Testing robust persistence and validation.</p>',
      ownerId: ownerUser.id,
    });

    expect(newDoc.id).toBeDefined();
    expect(newDoc.title).toBe('Automated Test Architecture RFC');
    expect(newDoc.ownerId).toBe(ownerUser.id);
    expect(newDoc.wordCount).toBeGreaterThan(0);
    expect(newDoc.characterCount).toBeGreaterThan(0);
    expect(newDoc.revisions?.length).toBe(1);
  });

  it('should allow owner and editor to update document content', () => {
    const doc = Storage.create({
      title: 'Draft Feature Spec',
      content: '<p>Initial draft.</p>',
      ownerId: ownerUser.id,
    });

    // Share with editorUser
    Storage.share(doc.id, ownerUser.id, editorUser.email, 'editor');

    // Update by editor
    const updated = Storage.update(doc.id, {
      title: 'Draft Feature Spec (Reviewed)',
      content: '<p>Updated draft content by editor.</p>',
      userId: editorUser.id,
    });

    expect(updated).not.toBeNull();
    expect(updated?.title).toBe('Draft Feature Spec (Reviewed)');
    expect(updated?.content).toContain('Updated draft content');
  });

  it('should REJECT update attempts by a Viewer (read-only enforcement)', () => {
    const doc = Storage.create({
      title: 'Confidential Strategy Note',
      content: '<p>Read-only note.</p>',
      ownerId: ownerUser.id,
    });

    // Share with viewerUser as viewer
    Storage.share(doc.id, ownerUser.id, viewerUser.email, 'viewer');

    // Attempt to update by viewer must throw Unauthorized error
    expect(() => {
      Storage.update(doc.id, {
        content: '<p>Hacked content</p>',
        userId: viewerUser.id,
      });
    }).toThrow(/Unauthorized/);
  });

  it('should correctly calculate user roles', () => {
    const doc = Storage.create({
      title: 'Role Test Doc',
      content: '<p>Role testing</p>',
      ownerId: ownerUser.id,
    });

    Storage.share(doc.id, ownerUser.id, editorUser.email, 'editor');
    Storage.share(doc.id, ownerUser.id, viewerUser.email, 'viewer');

    const freshDoc = Storage.getById(doc.id)!;
    expect(getUserRoleForDocument(freshDoc, ownerUser.id)).toBe('owner');
    expect(getUserRoleForDocument(freshDoc, editorUser.id)).toBe('editor');
    expect(getUserRoleForDocument(freshDoc, viewerUser.id)).toBe('viewer');
  });

  it('should parse Markdown correctly into structured HTML', () => {
    const md = '# Title\n\n## Subheading\n\nThis is **bold** and *italic* text.\n\n- Item 1\n- Item 2';
    const html = markdownToHtml(md);

    expect(html).toContain('<h1>Title</h1>');
    expect(html).toContain('<h2>Subheading</h2>');
    expect(html).toContain('<strong>bold</strong>');
    expect(html).toContain('<em>italic</em>');
    expect(html).toContain('<li>Item 1</li>');
  });

  it('should convert Plain Text into HTML paragraphs', () => {
    const text = 'First paragraph.\n\nSecond paragraph.';
    const html = plainTextToHtml(text);

    expect(html).toContain('<p>First paragraph.</p>');
    expect(html).toContain('<p>Second paragraph.</p>');
  });
});
