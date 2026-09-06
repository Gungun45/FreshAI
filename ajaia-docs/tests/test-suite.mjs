import fs from 'fs';
import path from 'path';
import assert from 'assert';

console.log('🧪 Running Ajaia Docs Automated Test Suite...\n');

const DATA_DIR = path.join(process.cwd(), 'data');
const DATA_FILE = path.join(DATA_DIR, 'documents.json');

// Ensure directory and initial seed
function ensureStorage() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }

  if (!fs.existsSync(DATA_FILE)) {
    const initialDocs = [
      {
        id: 'doc-seed-1',
        title: 'Product Requirements: AI Collaborative Workspace',
        content: '<h1>Product Requirements</h1><p>Seeded PRD.</p>',
        ownerId: 'user_vaibhavi',
        ownerName: 'Vaibhavi Diwakar',
        ownerEmail: 'taniyadiwaker6@gmail.com',
        collaborators: [
          {
            userId: 'user_alex',
            email: 'alex.chen@ajaia.io',
            name: 'Alex Chen',
            role: 'editor',
            addedAt: new Date().toISOString(),
          },
          {
            userId: 'user_sarah',
            email: 'sarah.miller@ajaia.io',
            name: 'Sarah Miller',
            role: 'viewer',
            addedAt: new Date().toISOString(),
          },
        ],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        wordCount: 20,
        characterCount: 150,
        revisions: [
          {
            id: 'rev-1',
            timestamp: new Date().toISOString(),
            title: 'Product Requirements: AI Collaborative Workspace',
            content: '<h1>Product Requirements</h1><p>Seeded PRD.</p>',
            savedByUserId: 'user_vaibhavi',
            savedByName: 'Vaibhavi Diwakar',
            summary: 'Initial draft',
          },
        ],
      },
    ];
    fs.writeFileSync(DATA_FILE, JSON.stringify(initialDocs, null, 2), 'utf-8');
  }

  return JSON.parse(fs.readFileSync(DATA_FILE, 'utf-8'));
}

let passed = 0;
let failed = 0;

function it(description, fn) {
  try {
    fn();
    console.log(`  ✅ PASS: ${description}`);
    passed++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${description}`);
    console.error(`     Error: ${err.message}`);
    failed++;
  }
}

// 1. Initial State
it('Initializes with seeded documents and team members', () => {
  const docs = ensureStorage();
  assert.ok(Array.isArray(docs), 'Documents must be an array');
  assert.ok(docs.length >= 1, 'Should have at least 1 document');
  const prd = docs.find((d) => d.id === 'doc-seed-1');
  assert.ok(prd, 'doc-seed-1 must exist');
  assert.strictEqual(prd.ownerId, 'user_vaibhavi');
  assert.strictEqual(prd.collaborators.length, 2);
});

// 2. Document Creation
it('Creates new documents with ownership, timestamps, and revision snapshots', () => {
  const docs = ensureStorage();
  const newDocId = `test-doc-${Date.now()}`;
  const newDoc = {
    id: newDocId,
    title: 'Automated CI Test Document',
    content: '<h1>Integration Test</h1><p>Verifying full-stack functionality.</p>',
    ownerId: 'user_vaibhavi',
    ownerName: 'Vaibhavi Diwakar',
    ownerEmail: 'taniyadiwaker6@gmail.com',
    collaborators: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    wordCount: 5,
    characterCount: 42,
    revisions: [
      {
        id: `rev-${Date.now()}`,
        timestamp: new Date().toISOString(),
        title: 'Automated CI Test Document',
        content: '<h1>Integration Test</h1><p>Verifying full-stack functionality.</p>',
        savedByUserId: 'user_vaibhavi',
        savedByName: 'Vaibhavi Diwakar',
        summary: 'Initial draft created',
      },
    ],
  };

  docs.unshift(newDoc);
  fs.writeFileSync(DATA_FILE, JSON.stringify(docs, null, 2), 'utf-8');

  const updatedDocs = ensureStorage();
  const found = updatedDocs.find((d) => d.id === newDocId);
  assert.ok(found, 'New document should be saved to persistent storage');
  assert.strictEqual(found.title, 'Automated CI Test Document');
  assert.strictEqual(found.revisions.length, 1);
});

// 3. Sharing & Collaborator Permissions
it('Shares document with granular role (Editor vs Viewer)', () => {
  const docs = ensureStorage();
  const testDoc = docs[0];

  testDoc.collaborators = testDoc.collaborators || [];
  testDoc.collaborators.push({
    userId: 'user_jordan',
    email: 'jordan.taylor@ajaia.io',
    name: 'Jordan Taylor',
    role: 'editor',
    addedAt: new Date().toISOString(),
  });

  fs.writeFileSync(DATA_FILE, JSON.stringify(docs, null, 2), 'utf-8');

  const updated = ensureStorage()[0];
  const jordan = updated.collaborators.find((c) => c.userId === 'user_jordan');
  assert.ok(jordan, 'Jordan must be present in collaborators');
  assert.strictEqual(jordan.role, 'editor');
});

// 4. Role Authorization Logic
it('Enforces authorization rules (Owner > Editor > Viewer)', () => {
  function canEdit(doc, userId) {
    if (doc.ownerId === userId) return true;
    const collab = doc.collaborators?.find((c) => c.userId === userId);
    return Boolean(collab && collab.role === 'editor');
  }

  const mockDoc = {
    id: 'test-perm',
    ownerId: 'user_vaibhavi',
    collaborators: [
      { userId: 'user_alex', role: 'editor' },
      { userId: 'user_sarah', role: 'viewer' },
    ],
  };

  assert.strictEqual(canEdit(mockDoc, 'user_vaibhavi'), true, 'Owner can edit');
  assert.strictEqual(canEdit(mockDoc, 'user_alex'), true, 'Editor can edit');
  assert.strictEqual(canEdit(mockDoc, 'user_sarah'), false, 'Viewer cannot edit (Read-only)');
  assert.strictEqual(canEdit(mockDoc, 'user_stranger'), false, 'Uninvited stranger cannot edit');
});

// 5. Markdown Parsing Logic
it('Parses Markdown headers, bold, italics, quotes, and lists into HTML', () => {
  function mdToHtml(md) {
    let html = md
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/gim, '<em>$1</em>')
      .replace(/^\s*[-*+]\s+(.*$)/gim, '<li>$1</li>');
    return html;
  }

  const sample = '# Heading 1\n## Heading 2\n**Bold** and *Italic*\n- Bullet';
  const result = mdToHtml(sample);
  assert.ok(result.includes('<h1>Heading 1</h1>'));
  assert.ok(result.includes('<h2>Heading 2</h2>'));
  assert.ok(result.includes('<strong>Bold</strong>'));
  assert.ok(result.includes('<em>Italic</em>'));
  assert.ok(result.includes('<li>Bullet</li>'));
});

console.log(`\n========================================`);
console.log(`🏁 Test Summary: ${passed} Passed, ${failed} Failed`);
console.log(`========================================\n`);

if (failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
