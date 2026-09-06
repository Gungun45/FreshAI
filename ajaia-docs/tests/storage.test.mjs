import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Storage, getUserRoleForDocument } from '../src/lib/storage.ts';
import { markdownToHtml, plainTextToHtml } from '../src/lib/fileParsers.ts';
import { SEEDED_USERS } from '../src/lib/users.ts';

describe('Ajaia Collaborative Docs - Core Storage & Permissions', () => {
  const ownerUser = SEEDED_USERS[0]; // Vaibhavi Diwakar
  const editorUser = SEEDED_USERS[1]; // Alex Chen
  const viewerUser = SEEDED_USERS[3]; // Sarah Miller

  it('initializes and lists seeded documents with ownership tracking', () => {
    const docs = Storage.list();
    assert.ok(Array.isArray(docs), 'Documents must be an array');
    assert.ok(docs.length >= 1, 'Should have seeded documents');
  });

  it('creates a new document with owner and calculates word/char counts', () => {
    const newDoc = Storage.create({
      title: 'Automated RFC Spec',
      content: '<h1>Architecture Overview</h1><p>Verifying full stack persistence.</p>',
      ownerId: ownerUser.id,
    });

    assert.ok(newDoc.id, 'Document ID should exist');
    assert.strictEqual(newDoc.title, 'Automated RFC Spec');
    assert.strictEqual(newDoc.ownerId, ownerUser.id);
    assert.ok(newDoc.wordCount > 0, 'Word count should be positive');
    assert.ok(newDoc.characterCount > 0, 'Char count should be positive');
  });

  it('allows owner and editor to update document content', () => {
    const doc = Storage.create({
      title: 'Draft Feature Spec',
      content: '<p>Initial draft.</p>',
      ownerId: ownerUser.id,
    });

    // Share with editor
    Storage.share(doc.id, ownerUser.id, editorUser.email, 'editor');

    // Update by editor
    const updated = Storage.update(doc.id, {
      title: 'Draft Feature Spec (Reviewed)',
      content: '<p>Updated content by editor.</p>',
      userId: editorUser.id,
    });

    assert.ok(updated, 'Updated document should not be null');
    assert.strictEqual(updated?.title, 'Draft Feature Spec (Reviewed)');
    assert.ok(updated?.content.includes('Updated content by editor'));
  });

  it('REJECTS update attempts by a Viewer (read-only authorization enforcement)', () => {
    const doc = Storage.create({
      title: 'Confidential Strategy Note',
      content: '<p>Read only note.</p>',
      ownerId: ownerUser.id,
    });

    // Share as viewer
    Storage.share(doc.id, ownerUser.id, viewerUser.email, 'viewer');

    // Attempt to update by viewer must throw Unauthorized error
    assert.throws(
      () => {
        Storage.update(doc.id, {
          content: '<p>Hacked content</p>',
          userId: viewerUser.id,
        });
      },
      /Unauthorized/,
      'Must reject viewer modifications'
    );
  });

  it('correctly maps user roles for owner, editor, and viewer', () => {
    const doc = Storage.create({
      title: 'Role Verification Doc',
      content: '<p>Testing permissions mapping</p>',
      ownerId: ownerUser.id,
    });

    Storage.share(doc.id, ownerUser.id, editorUser.email, 'editor');
    Storage.share(doc.id, ownerUser.id, viewerUser.email, 'viewer');

    const freshDoc = Storage.getById(doc.id);
    assert.ok(freshDoc);
    assert.strictEqual(getUserRoleForDocument(freshDoc, ownerUser.id), 'owner');
    assert.strictEqual(getUserRoleForDocument(freshDoc, editorUser.id), 'editor');
    assert.strictEqual(getUserRoleForDocument(freshDoc, viewerUser.id), 'viewer');
  });

  it('parses Markdown into structured semantic HTML', () => {
    const md = '# Header 1\n\n## Header 2\n\n**Bold Text** and *Italic Text*\n\n- Point A\n- Point B';
    const html = markdownToHtml(md);

    assert.ok(html.includes('<h1>Header 1</h1>'));
    assert.ok(html.includes('<h2>Header 2</h2>'));
    assert.ok(html.includes('<strong>Bold Text</strong>'));
    assert.ok(html.includes('<em>Italic Text</em>'));
    assert.ok(html.includes('<li>Point A</li>'));
  });

  it('converts Plain Text into formatted paragraphs', () => {
    const text = 'Line 1\n\nLine 2';
    const html = plainTextToHtml(text);
    assert.ok(html.includes('<p>Line 1</p>'));
    assert.ok(html.includes('<p>Line 2</p>'));
  });
});
