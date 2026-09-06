// AJAIA DOCS - APPLICATION LOGIC

const SEEDED_USERS = [
  { id: 'user_vaibhavi', name: 'Vaibhavi Diwakar', email: 'taniyadiwaker6@gmail.com', role: 'Lead Product Engineer (Owner)', avatarColor: 'bg-indigo-600', initials: 'VD' },
  { id: 'user_alex', name: 'Alex Chen', email: 'alex.chen@ajaia.io', role: 'Head of Product', avatarColor: 'bg-emerald-600', initials: 'AC' },
  { id: 'user_jordan', name: 'Jordan Taylor', email: 'jordan.taylor@ajaia.io', role: 'Staff Platform Engineer', avatarColor: 'bg-violet-600', initials: 'JT' },
  { id: 'user_sarah', name: 'Sarah Miller', email: 'sarah.miller@ajaia.io', role: 'Design Director', avatarColor: 'bg-amber-600', initials: 'SM' },
];

let currentUser = SEEDED_USERS[0];
let activeTab = 'all';
let activeDocument = null;
let autoSaveTimer = null;
let selectedFile = null;
let isImportingToDraft = false;

function getDocuments() {
  const stored = localStorage.getItem('ajaia_docs_data');
  if (stored) {
    try { return JSON.parse(stored); } catch (e) {}
  }
  const initial = [
    {
      id: 'doc-seed-1',
      title: 'Product Requirements: AI Collaborative Workspace',
      content: '<h1>Product Requirements: AI Collaborative Workspace</h1><p>Welcome to <strong>Ajaia Docs</strong>, a high-performance, real-time collaborative workspace engineered for modern product teams.</p><h2>1. Executive Summary</h2><p>This product bridges rich-text authoring, multi-party collaboration, and file ingestion with frictionless permission sharing.</p><h3>Key Value Propositions</h3><ul><li><strong>Instant Rich-Text Authoring:</strong> Headings, rich formatting, quotes, and structured lists.</li><li><strong>Frictionless File Ingestion:</strong> Import Markdown (.md), Plain Text (.txt), and Word (.docx) directly into editable drafts.</li><li><strong>Role-Based Access Control:</strong> Seamlessly grant <em>Editor</em> or <em>Viewer</em> permissions across cross-functional teams.</li></ul><blockquote>"A focused, well-reasoned solution is better than an overextended build." — Ajaia Engineering Principles</blockquote><p>Feel free to edit this document, invite teammates, or upload existing notes.</p>',
      ownerId: 'user_vaibhavi',
      ownerName: 'Vaibhavi Diwakar',
      ownerEmail: 'taniyadiwaker6@gmail.com',
      collaborators: [
        { userId: 'user_alex', email: 'alex.chen@ajaia.io', name: 'Alex Chen', role: 'editor', addedAt: new Date().toISOString() },
        { userId: 'user_sarah', email: 'sarah.miller@ajaia.io', name: 'Sarah Miller', role: 'viewer', addedAt: new Date().toISOString() }
      ],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      wordCount: 110,
      characterCount: 820,
      revisions: [
        { id: 'rev-1', timestamp: new Date().toISOString(), title: 'Product Requirements: AI Collaborative Workspace', content: '<h1>Product Requirements</h1><p>Initial draft.</p>', savedByName: 'Vaibhavi Diwakar', summary: 'Initial document creation' }
      ]
    },
    {
      id: 'doc-seed-2',
      title: 'Ajaia Architecture & System RFC #104',
      content: '<h1>Ajaia Architecture &amp; System RFC #104</h1><p><strong>Author:</strong> Alex Chen (Head of Product)<br/><strong>Status:</strong> Under Review</p><h2>1. Architecture Overview</h2><p>We leverage a modern full-stack TypeScript architecture using atomic persistence, client-side rich text state management, and role-based access control.</p><h3>Permissions Model</h3><ol><li><strong>Owner:</strong> Full rights (edit, rename, delete, manage permissions).</li><li><strong>Editor:</strong> Read and write content modifications.</li><li><strong>Viewer:</strong> Read-only access with interface controls locked.</li></ol>',
      ownerId: 'user_alex',
      ownerName: 'Alex Chen',
      ownerEmail: 'alex.chen@ajaia.io',
      collaborators: [
        { userId: 'user_vaibhavi', email: 'taniyadiwaker6@gmail.com', name: 'Vaibhavi Diwakar', role: 'editor', addedAt: new Date().toISOString() },
        { userId: 'user_jordan', email: 'jordan.taylor@ajaia.io', name: 'Jordan Taylor', role: 'editor', addedAt: new Date().toISOString() }
      ],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      wordCount: 88,
      characterCount: 650
    }
  ];
  localStorage.setItem('ajaia_docs_data', JSON.stringify(initial));
  return initial;
}

function saveDocuments(docs) {
  localStorage.setItem('ajaia_docs_data', JSON.stringify(docs));
}

function toggleUserDropdown() {
  document.getElementById('userDropdownMenu').classList.toggle('hidden');
}

function renderUserList() {
  const container = document.getElementById('userListContainer');
  container.innerHTML = SEEDED_USERS.map(u => `
    <button onclick="switchUser('${u.id}')" class="w-full flex items-center justify-between px-3 py-2 rounded-xl text-left text-xs ${u.id === currentUser.id ? 'bg-indigo-950/60 text-indigo-200 border border-indigo-700/50' : 'text-slate-300 hover:bg-slate-800'}">
      <div class="flex items-center gap-2.5">
        <div class="w-7 h-7 rounded-full ${u.avatarColor} text-white flex items-center justify-center font-bold text-xs">${u.initials}</div>
        <div><p class="font-medium text-slate-100">${u.name}</p><p class="text-[10px] text-slate-400">${u.role}</p></div>
      </div>
      ${u.id === currentUser.id ? '<i data-lucide="check" class="w-4 h-4 text-indigo-400"></i>' : ''}
    </button>
  `).join('');
  lucide.createIcons();
}

function switchUser(userId) {
  const user = SEEDED_USERS.find(u => u.id === userId);
  if (user) {
    currentUser = user;
    document.getElementById('activeUserAvatar').className = 'w-6 h-6 rounded-full ' + user.avatarColor + ' text-white flex items-center justify-center text-xs font-bold';
    document.getElementById('activeUserAvatar').textContent = user.initials;
    document.getElementById('activeUserName').textContent = user.name;
    document.getElementById('activeUserRole').textContent = user.role.split('(')[0];
    document.getElementById('welcomeUserName').textContent = user.name;
    document.getElementById('userDropdownMenu').classList.add('hidden');
    renderUserList();
    if (activeDocument) openDocument(activeDocument.id);
    else renderDocumentList();
  }
}

function setTab(tab) {
  activeTab = tab;
  ['tabAll', 'tabOwned', 'tabShared'].forEach(id => {
    document.getElementById(id).className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200';
  });
  const activeBtn = document.getElementById(tab === 'all' ? 'tabAll' : tab === 'owned' ? 'tabOwned' : 'tabShared');
  activeBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow';
  renderDocumentList();
}

function renderDocumentList() {
  const docs = getDocuments();
  const query = (document.getElementById('searchInput')?.value || '').toLowerCase();
  const grid = document.getElementById('documentGrid');
  if (!grid) return;

  const filtered = docs.filter(d => {
    const isOwner = d.ownerId === currentUser.id;
    const isCollab = d.collaborators && d.collaborators.some(c => c.userId === currentUser.id || c.email.toLowerCase() === currentUser.email.toLowerCase());
    let tabMatch = (activeTab === 'owned') ? isOwner : ((activeTab === 'shared') ? (isCollab && !isOwner) : (isOwner || isCollab));
    return tabMatch && (d.title.toLowerCase().includes(query) || (d.ownerName && d.ownerName.toLowerCase().includes(query)));
  });

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="col-span-full bg-slate-900/40 border border-slate-800 rounded-2xl py-16 px-4 text-center space-y-4"><i data-lucide="file-text" class="w-10 h-10 text-slate-500 mx-auto"></i><p class="text-sm font-semibold text-slate-300">No documents found</p><p class="text-xs text-slate-500">Switch user or create a new document.</p></div>';
    lucide.createIcons();
    return;
  }

  grid.innerHTML = filtered.map(d => {
    const isOwner = d.ownerId === currentUser.id;
    const collab = d.collaborators ? d.collaborators.find(c => c.userId === currentUser.id || c.email.toLowerCase() === currentUser.email.toLowerCase()) : null;
    const plainText = d.content.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();

    return `
      <div onclick="openDocument('${d.id}')" class="group bg-slate-900/70 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-2xl p-5 shadow-md transition-all cursor-pointer flex flex-col justify-between">
        <div class="space-y-3">
          <div class="flex items-start justify-between gap-2">
            <div class="w-9 h-9 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center justify-center group-hover:bg-indigo-600 group-hover:text-white transition-colors">
              <i data-lucide="file-text" class="w-5 h-5"></i>
            </div>
            ${isOwner ? '<span class="inline-flex items-center gap-1 text-[11px] font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded-full"><i data-lucide="sparkles" class="w-2.5 h-2.5"></i> Owner</span>' : `<span class="inline-flex items-center gap-1 text-[11px] font-medium ${collab?.role === 'editor' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'} border px-2 py-0.5 rounded-full"><i data-lucide="${collab?.role === 'editor' ? 'edit-3' : 'eye'}" class="w-2.5 h-2.5"></i> ${collab ? collab.role.toUpperCase() : 'SHARED'}</span>`}
          </div>
          <div>
            <h3 class="font-semibold text-slate-100 text-base line-clamp-1 group-hover:text-indigo-300 transition-colors">${d.title}</h3>
            <p class="text-xs text-slate-400 mt-1 line-clamp-2">${plainText || 'Empty document'}</p>
          </div>
        </div>
        <div class="pt-4 mt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span class="flex items-center gap-1"><i data-lucide="clock" class="w-3.5 h-3.5 text-slate-500"></i> ${d.wordCount || 0} words</span>
          ${isOwner ? `<button onclick="event.stopPropagation(); deleteDoc('${d.id}')" class="p-1 text-slate-400 hover:text-rose-400"><i data-lucide="trash-2" class="w-4 h-4"></i></button>` : ''}
        </div>
      </div>
    `;
  }).join('');
  lucide.createIcons();
}

function createNewDocument() {
  const docs = getDocuments();
  const newDoc = {
    id: 'doc-' + Date.now(),
    title: 'Untitled Document',
    content: '<p>Start typing your document here...</p>',
    ownerId: currentUser.id,
    ownerName: currentUser.name,
    ownerEmail: currentUser.email,
    collaborators: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    wordCount: 5,
    characterCount: 35,
    revisions: []
  };
  docs.unshift(newDoc);
  saveDocuments(docs);
  openDocument(newDoc.id);
}

function openDocument(docId) {
  const docs = getDocuments();
  const doc = docs.find(d => d.id === docId);
  if (!doc) return;

  activeDocument = doc;
  document.getElementById('dashboardView').classList.add('hidden');
  document.getElementById('editorView').classList.remove('hidden');

  const isOwner = doc.ownerId === currentUser.id;
  const collab = doc.collaborators ? doc.collaborators.find(c => c.userId === currentUser.id || c.email.toLowerCase() === currentUser.email.toLowerCase()) : null;
  const role = isOwner ? 'owner' : (collab ? collab.role : 'viewer');
  const isReadOnly = (role === 'viewer');

  document.getElementById('docTitleInput').value = doc.title;
  document.getElementById('docTitleInput').disabled = !isOwner;
  document.getElementById('docIdDisplay').textContent = doc.id;
  document.getElementById('editorContent').innerHTML = doc.content;
  document.getElementById('editorContent').contentEditable = !isReadOnly;

  document.getElementById('editorToolbar').style.display = isReadOnly ? 'none' : 'flex';
  document.getElementById('viewerNotice').style.display = isReadOnly ? 'flex' : 'none';
  document.getElementById('importBtn').style.display = isReadOnly ? 'none' : 'flex';

  const roleBadge = document.getElementById('activeDocRoleBadge');
  roleBadge.textContent = role.toUpperCase() + ' MODE';
  roleBadge.className = isOwner ? 'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : (role === 'editor' ? 'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-cyan-500/20 text-cyan-300 border border-cyan-500/30');

  updateMetrics();
  lucide.createIcons();
}

function showDashboard() {
  activeDocument = null;
  document.getElementById('editorView').classList.add('hidden');
  document.getElementById('dashboardView').classList.remove('hidden');
  renderDocumentList();
}

function deleteDoc(docId) {
  const docs = getDocuments();
  const doc = docs.find(d => d.id === docId);
  if (!doc) return;
  if (doc.ownerId !== currentUser.id) {
    alert('Only the document owner can delete this document.');
    return;
  }
  if (!confirm('Are you sure you want to delete this document?')) return;
  const remaining = docs.filter(d => d.id !== docId);
  saveDocuments(remaining);
  renderDocumentList();
}

function handleTitleInput(val) {
  if (!activeDocument) return;
  if (activeDocument.ownerId !== currentUser.id) return;
  activeDocument.title = val.trim() || 'Untitled Document';
  triggerAutoSave();
}

function handleEditorInput() {
  if (!activeDocument) return;
  activeDocument.content = document.getElementById('editorContent').innerHTML;
  updateMetrics();
  triggerAutoSave();
}

function updateMetrics() {
  const text = document.getElementById('editorContent').innerText.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = text.length;
  document.getElementById('wordCountDisplay').textContent = words;
  document.getElementById('charCountDisplay').textContent = chars;
  if (activeDocument) {
    activeDocument.wordCount = words;
    activeDocument.characterCount = chars;
  }
}

function triggerAutoSave() {
  document.getElementById('saveStatusIndicator').innerHTML = '<i data-lucide="clock" class="w-3 h-3 animate-spin"></i> Saving...';
  document.getElementById('saveStatusIndicator').className = 'flex items-center gap-1 text-amber-300';
  lucide.createIcons();

  clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(() => {
    if (!activeDocument) return;
    const docs = getDocuments();
    const idx = docs.findIndex(d => d.id === activeDocument.id);
    if (idx >= 0) {
      activeDocument.updatedAt = new Date().toISOString();
      activeDocument.revisions = activeDocument.revisions || [];
      activeDocument.revisions.push({
        id: 'rev-' + Date.now(),
        timestamp: new Date().toISOString(),
        title: activeDocument.title,
        content: activeDocument.content,
        savedByName: currentUser.name,
        summary: 'Saved content updates'
      });
      if (activeDocument.revisions.length > 20) activeDocument.revisions.shift();

      docs[idx] = activeDocument;
      saveDocuments(docs);
    }
    document.getElementById('saveStatusIndicator').innerHTML = '<i data-lucide="check-circle" class="w-3 h-3"></i> Saved';
    document.getElementById('saveStatusIndicator').className = 'flex items-center gap-1 text-emerald-400';
    lucide.createIcons();
  }, 1000);
}

function formatDoc(cmd, val = null) {
  document.execCommand(cmd, false, val);
  handleEditorInput();
}

function formatHeading(tag) {
  document.execCommand('formatBlock', false, '<' + tag + '>');
  handleEditorInput();
}

function formatBlockquote() {
  document.execCommand('formatBlock', false, '<blockquote>');
  handleEditorInput();
}

function openShareModal() {
  if (!activeDocument) return;
  document.getElementById('shareModal').classList.remove('hidden');
  const isOwner = activeDocument.ownerId === currentUser.id;
  document.getElementById('shareInviteSection').style.display = isOwner ? 'block' : 'none';

  const available = SEEDED_USERS.filter(u => u.id !== activeDocument.ownerId && (!activeDocument.collaborators || !activeDocument.collaborators.some(c => c.userId === u.id)));
  document.getElementById('quickInvitePills').innerHTML = available.length ? `
    <span class="text-[11px] text-slate-400">Quick invite:</span>
    <div class="flex flex-wrap gap-1.5 mt-1">
      ${available.map(u => `
        <button onclick="quickInvite('${u.email}')" class="px-2 py-0.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs border border-slate-700 flex items-center gap-1">
          <span class="w-2 h-2 rounded-full ${u.avatarColor}"></span> ${u.name}
        </button>
      `).join('')}
    </div>
  ` : '';

  renderCollaboratorsList();
  lucide.createIcons();
}

function renderCollaboratorsList() {
  const isOwner = activeDocument.ownerId === currentUser.id;
  const list = document.getElementById('collaboratorListContainer');
  const collabs = activeDocument.collaborators || [];

  list.innerHTML = `
    <div class="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
      <div class="flex items-center gap-2.5">
        <div class="w-7 h-7 rounded-full bg-indigo-600 text-white flex items-center justify-center text-xs font-bold">${activeDocument.ownerName ? activeDocument.ownerName[0] : 'O'}</div>
        <div>
          <p class="text-xs font-semibold text-slate-200">${activeDocument.ownerName || 'Owner'} <span class="text-[10px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.2 rounded font-mono">Owner</span></p>
          <p class="text-[11px] text-slate-400">${activeDocument.ownerEmail}</p>
        </div>
      </div>
      <span class="text-xs text-slate-400">Full Access</span>
    </div>
    ${collabs.map(c => `
      <div class="flex items-center justify-between p-2.5 rounded-xl bg-slate-950/40 border border-slate-800">
        <div class="flex items-center gap-2.5">
          <div class="w-7 h-7 rounded-full bg-slate-700 text-white flex items-center justify-center text-xs font-bold">${c.name ? c.name[0] : c.email[0].toUpperCase()}</div>
          <div><p class="text-xs font-medium text-slate-200">${c.name || c.email}</p><p class="text-[11px] text-slate-400">${c.email}</p></div>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs font-medium capitalize text-slate-300">${c.role}</span>
          ${isOwner ? `<button onclick="removeCollab('${c.userId || c.email}')" class="p-1 text-slate-400 hover:text-rose-400"><i data-lucide="trash-2" class="w-3.5 h-3.5"></i></button>` : ''}
        </div>
      </div>
    `).join('')}
  `;
  lucide.createIcons();
}

function quickInvite(email) {
  document.getElementById('shareEmailInput').value = email;
  addCollaborator();
}

function addCollaborator() {
  if (!activeDocument) return;
  if (activeDocument.ownerId !== currentUser.id) {
    alert('Only the document owner can invite collaborators or manage permissions.');
    return;
  }

  const email = document.getElementById('shareEmailInput').value.trim();
  const role = document.getElementById('shareRoleSelect').value;
  if (!email) {
    alert('Please enter a valid collaborator email.');
    return;
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    alert('Please enter a valid email address.');
    return;
  }

  if (email.toLowerCase() === activeDocument.ownerEmail.toLowerCase()) {
    alert('The document owner already has full administrative access.');
    return;
  }

  const user = SEEDED_USERS.find(u => u.email.toLowerCase() === email.toLowerCase());
  activeDocument.collaborators = activeDocument.collaborators || [];
  const exists = activeDocument.collaborators.find(c => c.email.toLowerCase() === email.toLowerCase());
  if (exists) {
    exists.role = role;
  } else {
    activeDocument.collaborators.push({
      userId: user ? user.id : 'guest_' + Date.now(),
      name: user ? user.name : email.split('@')[0],
      email: email,
      role: role,
      addedAt: new Date().toISOString()
    });
  }

  const docs = getDocuments();
  const idx = docs.findIndex(d => d.id === activeDocument.id);
  if (idx >= 0) { docs[idx] = activeDocument; saveDocuments(docs); }
  document.getElementById('shareEmailInput').value = '';
  openShareModal();
}

function removeCollab(idOrEmail) {
  if (!activeDocument) return;
  if (activeDocument.ownerId !== currentUser.id) {
    alert('Only the document owner can remove collaborators.');
    return;
  }
  activeDocument.collaborators = activeDocument.collaborators.filter(c => c.userId !== idOrEmail && c.email !== idOrEmail);
  const docs = getDocuments();
  const idx = docs.findIndex(d => d.id === activeDocument.id);
  if (idx >= 0) { docs[idx] = activeDocument; saveDocuments(docs); }
  renderCollaboratorsList();
}

function copyDocLink() {
  navigator.clipboard.writeText(window.location.href);
  document.getElementById('copyLinkText').textContent = 'Copied!';
  setTimeout(() => document.getElementById('copyLinkText').textContent = 'Copy Link', 2000);
}

function openUploadModal(intoDraft = false) {
  if (intoDraft && activeDocument) {
    const isOwner = activeDocument.ownerId === currentUser.id;
    const collab = activeDocument.collaborators ? activeDocument.collaborators.find(c => c.userId === currentUser.id || c.email.toLowerCase() === currentUser.email.toLowerCase()) : null;
    const role = isOwner ? 'owner' : (collab ? collab.role : 'viewer');
    if (role === 'viewer') {
      alert('Read-only viewers cannot import content into this document.');
      return;
    }
  }
  isImportingToDraft = intoDraft;
  selectedFile = null;
  document.getElementById('uploadFilenameText').textContent = 'Click or drag & drop file here';
  document.getElementById('submitUploadBtn').disabled = true;
  document.getElementById('uploadModal').classList.remove('hidden');
  lucide.createIcons();
}

function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    const file = e.target.files[0];
    if (file.size > 5 * 1024 * 1024) {
      alert('File size exceeds the 5MB limit. Please select a smaller file.');
      e.target.value = '';
      return;
    }
    selectedFile = file;
    document.getElementById('uploadFilenameText').textContent = selectedFile.name + ' (' + (selectedFile.size / 1024).toFixed(1) + ' KB)';
    document.getElementById('submitUploadBtn').disabled = false;
  }
}

function executeFileUpload() {
  if (!selectedFile) return;
  const reader = new FileReader();
  const ext = selectedFile.name.split('.').pop().toLowerCase();
  const title = selectedFile.name.substring(0, selectedFile.name.lastIndexOf('.')) || selectedFile.name;

  reader.onload = function(evt) {
    const raw = evt.target.result;
    let htmlContent = '<p>' + raw.replace(/\n/g, '<br/>') + '</p>';

    if (ext === 'md' || ext === 'markdown') {
      htmlContent = raw
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        .replace(/^\s*[-*+]\s+(.*$)/gim, '<li>$1</li>')
        .replace(/(<li>[\s\S]*?<\/li>)/gi, '<ul>$1</ul>');
    }

    if (isImportingToDraft && activeDocument) {
      document.getElementById('editorContent').innerHTML += htmlContent;
      handleEditorInput();
      closeModals();
    } else {
      const docs = getDocuments();
      const newDoc = {
        id: 'doc-' + Date.now(),
        title: title,
        content: htmlContent,
        ownerId: currentUser.id,
        ownerName: currentUser.name,
        ownerEmail: currentUser.email,
        collaborators: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        wordCount: 10,
        characterCount: 100,
        revisions: []
      };
      docs.unshift(newDoc);
      saveDocuments(docs);
      closeModals();
      openDocument(newDoc.id);
    }
  };
  reader.readAsText(selectedFile);
}

function openHistoryModal() {
  if (!activeDocument) return;
  document.getElementById('historyModal').classList.remove('hidden');
  const container = document.getElementById('revisionListContainer');
  const revs = activeDocument.revisions || [];

  if (revs.length === 0) {
    container.innerHTML = '<div class="py-8 text-center text-slate-500 text-xs">No past revisions saved yet. Snapshots are created on save.</div>';
  } else {
    container.innerHTML = revs.slice().reverse().map((r, i) => `
      <div class="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold text-slate-200">${r.title} ${i === 0 ? '<span class="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full">Current</span>' : ''}</span>
          ${i !== 0 ? `<button onclick="restoreRevision('${r.id}')" class="px-2.5 py-1 bg-indigo-600/20 text-indigo-300 text-xs rounded-lg hover:bg-indigo-600/40">Restore</button>` : ''}
        </div>
        <p class="text-xs text-slate-400 italic">"${r.summary || 'Content updated'}"</p>
        <p class="text-[10px] text-slate-500">${new Date(r.timestamp).toLocaleString()} • by ${r.savedByName}</p>
      </div>
    `).join('');
  }
  lucide.createIcons();
}

function restoreRevision(revId) {
  const rev = activeDocument.revisions.find(r => r.id === revId);
  if (rev) {
    activeDocument.title = rev.title;
    activeDocument.content = rev.content;
    document.getElementById('docTitleInput').value = rev.title;
    document.getElementById('editorContent').innerHTML = rev.content;
    handleEditorInput();
    closeModals();
  }
}

function toggleExportMenu() {
  document.getElementById('exportMenu').classList.toggle('hidden');
}

function exportDoc(format) {
  if (!activeDocument) return;
  let fileContent = activeDocument.content;
  let mime = 'text/plain';
  let ext = format;

  if (format === 'html') {
    fileContent = '<!DOCTYPE html><html><head><title>' + activeDocument.title + '</title></head><body>' + activeDocument.content + '</body></html>';
    mime = 'text/html';
  } else if (format === 'txt') {
    fileContent = document.getElementById('editorContent').innerText;
  } else if (format === 'md') {
    fileContent = '# ' + activeDocument.title + '\n\n' + document.getElementById('editorContent').innerText;
    mime = 'text/markdown';
  }

  const blob = new Blob([fileContent], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = activeDocument.title.toLowerCase().replace(/[^a-z0-9]/gi, '_') + '.' + ext;
  a.click();
  document.getElementById('exportMenu').classList.add('hidden');
}

function closeModals() {
  document.getElementById('shareModal').classList.add('hidden');
  document.getElementById('uploadModal').classList.add('hidden');
  document.getElementById('historyModal').classList.add('hidden');
}

window.onload = function() {
  renderUserList();
  renderDocumentList();
  lucide.createIcons();
};
