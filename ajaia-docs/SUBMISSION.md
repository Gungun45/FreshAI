# Ajaia LLC — Full Stack Product Engineer Assignment Submission

**Candidate:** Vaibhavi Diwakar  
**Email:** `taniyadiwaker6@gmail.com`  
**Role:** AI-Native Full Stack Product Engineer  
**Project:** Ajaia Docs (Collaborative Document Studio)  

---

## 1. Project Summary & Core Highlights

Ajaia Docs is a production-grade, full-stack collaborative document editor inspired by Google Docs, built with **Next.js 16 App Router, TypeScript, Tailwind CSS, TipTap/ProseMirror**, and an atomic persistence engine.

### Key Capabilities Delivered:
- **Rich-Text Document Authoring:** Headings (H1–H3), Bold, Italic, Underline, Strikethrough, Bulleted Lists, Numbered Lists, Blockquotes, Code Blocks, Highlights, Text Alignment, and inline document renaming.
- **Debounced Auto-Save & Manual Save:** 1000ms debounced persistence with real-time save state indicators (`Saved`, `Saving...`, `Unsaved`).
- **File Ingestion Engine:** Drag-and-drop / upload support for `.md`, `.docx`, `.txt`, and `.json` files, with options to create new documents or import into active drafts.
- **Multi-Format Export:** Export documents as Markdown (`.md`), Plain Text (`.txt`), or HTML (`.html`).
- **Role-Based Access Control (RBAC):** Document Owner, Editor, and Viewer permission roles. Viewers are strictly locked into read-only mode on both client and server.
- **Multi-User Simulation:** Fast user switcher in the navbar to test permission flows across 4 seeded personas without manual signup hurdles.
- **Document Organization:** Filterable grid with tabs for *All Documents*, *Owned by Me*, and *Shared with Me*, plus real-time search.
- **Version History (Stretch Goal):** Snapshotting revisions on every save with one-click restore.
- **Automated Test Suite:** Comprehensive test suite (`npm test`) validating CRUD operations, role authorization, and parser correctness.

---

## 2. Seeded Test Accounts (Mock Authentication)

Reviewers can switch between accounts anytime via the top-right **User Switcher**:

1. **Vaibhavi Diwakar (Candidate / Owner)** — `taniyadiwaker6@gmail.com`
2. **Alex Chen (Head of Product)** — `alex.chen@ajaia.io`
3. **Jordan Taylor (Staff Platform Engineer)** — `jordan.taylor@ajaia.io`
4. **Sarah Miller (Design Director)** — `sarah.miller@ajaia.io`

---

## 3. Local Setup & Run Instructions

```bash
# 1. Clone or open the project folder
cd ajaia-docs

# 2. Install dependencies
npm install

# 3. Start local development server
npm run dev

# 4. Open in browser
# http://localhost:3000

# 5. Run automated test suite
npm test
```

---

## 4. Scope Prioritization & Architecture Tradeoffs

### What Is Working End-to-End:
- Full document lifecycle (Create, Edit, Rename, Save, Reopen, Delete).
- Rich text editing toolbar and keyboard shortcuts.
- File upload and parser for `.md`, `.docx`, `.txt`, and `.json`.
- Sharing modal with email invites, role management, and collaborator removal.
- Client and server-side RBAC enforcement.
- Durable persistence across refreshes with revision timeline snapshots.

### What Was Intentionally Deprioritized:
- **CRDT / Live Character-by-Character Cursor Syncing:** Deprioritized in favor of rock-solid debounced REST auto-saving, atomic file persistence, and robust RBAC permissions.
- **Inline Comment Threads:** Left for the next iteration to keep the UI clean and focused on core authoring.

### What We Would Build With Another 2–4 Hours:
1. **Live Presence & Live Cursors:** Integrate `y-webrtc` or WebSocket sync for peer presence indicators.
2. **Inline Comments & Suggestion Mode:** Allow reviewers to highlight text and leave threaded comments.
3. **Full-Text Document Search Indexing:** Fast fuzzy search across document bodies.

---

## 5. AI-Native Workflow Summary

- **Tools Used:** Antigravity / Gemini 3.7 Coding Agent.
- **Material Speedup:** Accelerated TipTap extension integration, Markdown/Docx parsing routines, Tailwind layout construction, and automated test synthesis (~3x speedup).
- **Rejected Output:** Overruled heavy Redis/WebSocket recommendations in favor of reliable, self-contained persistence and strict type safety.
- **Verification:** Verified via `npm test` automated suite, `npm run build` TypeScript validation, and multi-user permission testing.

---

## 6. Submission Links & Materials

- **Google Drive Folder:** Containing source code, documentation, and sample files.
- **Walkthrough Video Link:** *(Paste Loom / YouTube video URL here)*
- **Source Code Repository:** Included in project folder `ajaia-docs`.
