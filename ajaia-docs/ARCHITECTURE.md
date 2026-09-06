# Architecture & Design Note — Ajaia Collaborative Docs

## 1. Executive Summary & Goals

The objective of this project was to design and ship a cohesive, resilient, full-stack collaborative document editor inspired by Google Docs within a strict 4–6 hour timebox.

Rather than attempting to shallowly clone every edge-case feature of Google Docs, the architecture prioritizes **depth and reliability** in the core workflows that matter most to product teams:
1. **Fluid Rich-Text Editing & Document Authoring**
2. **Deterministic File Ingestion & Parsing (.md, .docx, .txt, .json)**
3. **Role-Based Access Control & Sharing (Owner vs. Editor vs. Viewer)**
4. **Resilient Persistence & Revision History**

---

## 2. System Architecture & Component Design

```
+---------------------------------------------------------------------------------+
|                                Frontend (Next.js / React)                       |
|                                                                                 |
|  +-------------------+  +-----------------------+  +-------------------------+  |
|  |   DocumentList    |  |      RichEditor       |  |       ShareModal        |  |
|  |  (Filter, Search) |  |   (TipTap, Toolbar)   |  |   (RBAC, Permissions)   |  |
|  +-------------------+  +-----------------------+  +-------------------------+  |
|            |                        |                           |               |
|            +------------------------+---------------------------+               |
|                                     |                                           |
|                           [ AuthContext / MockAuth ]                            |
+-------------------------------------|-------------------------------------------+
                                      | HTTP REST API
+-------------------------------------v-------------------------------------------+
|                               Next.js API Routes                                |
|                                                                                 |
|   /api/documents    /api/documents/[id]    /api/documents/[id]/share   /api/upload|
+-------------------------------------|-------------------------------------------+
                                      |
+-------------------------------------v-------------------------------------------+
|                              Backend Core Services                              |
|                                                                                 |
|  +-------------------------------------+  +----------------------------------+  |
|  |           Storage Service           |  |          File Parsers            |  |
|  |   - CRUD Operations                 |  |   - Markdown -> HTML             |  |
|  |   - RBAC Validation & Authorization |  |   - DOCX -> Mammoth Engine       |  |
|  |   - Revision Snapshotting           |  |   - Plain Text -> Paragraphs     |  |
|  +-------------------------------------+  +----------------------------------+  |
|                                     |                                           |
+-------------------------------------|-------------------------------------------+
                                      | Atomic File I/O
+-------------------------------------v-------------------------------------------+
|                       Persistence Layer (data/documents.json)                    |
+---------------------------------------------------------------------------------+
```

---

## 3. Key Design Decisions & Tradeoffs

### A. Rich Text Engine: TipTap / ProseMirror
- **Decision:** Built the editor on top of TipTap (ProseMirror headless core) rather than naive `contenteditable` or full-blown heavyweight suites.
- **Rationale:** TipTap provides structured JSON and semantic HTML output, making serialization clean and deterministic. It enables precise control over bold, italic, underline, strike, headings (H1-H3), lists, quotes, and alignments without dirty browser-specific DOM quirks.
- **Tradeoff:** TipTap requires hydration handling on the client side (`useEditor`), which was cleanly encapsulated inside the `RichEditor` component.

### B. Persistence Layer: Atomic File Store vs External Database
- **Decision:** Implemented an atomic JSON-based file store with staging `.tmp` rename patterns.
- **Rationale:** Avoids external DB infrastructure hurdles (Docker, PostgreSQL server setup, remote network latency) for reviewers while providing 100% durable persistence across page refreshes and server restarts.
- **Tradeoff:** For enterprise scale (>100k concurrent writes), a PostgreSQL database with row-level locking or optimistic concurrency control would be adopted.

### C. Multi-User Simulation (Mock Auth Context)
- **Decision:** Provided a fast User Switcher in the top navigation with seeded personas (Vaibhavi Diwakar, Alex Chen, Jordan Taylor, Sarah Miller) rather than a rigid single-user signup wall.
- **Rationale:** Reviewers can test collaboration, document transfer, and permission boundaries (e.g., verifying a Viewer cannot edit content) in 2 seconds without creating burner email accounts.

### D. File Ingestion & Transformation Engine
- **Decision:** Supported `.md`, `.docx`, `.txt`, and `.json` files.
- **Rationale:** Teams frequently migrate notes from Notion/Obsidian (Markdown) or Word (.docx). Supporting multi-format ingestion directly turns static files into live editable collaborative documents.

---

## 4. What Is Working End-to-End

- ✅ **Document CRUD:** Create, read, update, rename, and delete documents with instant URL routing.
- ✅ **Rich Text Toolbar:** Full formatting suite (Bold, Italic, Underline, Strike, Headings, Lists, Quotes, Code, Alignment, Highlight).
- ✅ **Auto-Save & Metrics:** 1000ms debounced auto-saving, visual save status pill, and real-time word/char counter.
- ✅ **Multi-Format Ingestion:** Drag-and-drop or file upload of `.md`, `.docx`, `.txt`, and `.json` to create new docs or append to drafts.
- ✅ **Exporting:** Download documents as `.md`, `.txt`, or `.html`.
- ✅ **Granular RBAC Sharing:** Owner, Editor, and Viewer permission levels with server-side authorization enforcement.
- ✅ **Document Filtering:** Clear visual distinction and tabs for *All Documents*, *Owned by Me*, and *Shared with Me*.
- ✅ **Version History:** Snapshotting revisions on every save with one-click restore.
- ✅ **Automated Test Suite:** Node test suite verifying data integrity and permission security rules.

---

## 5. What Was Deprioritized & Future Roadmap (Next 2–4 Hours)

### Deliberate Scope Cuts:
1. **CRDT / Operational Transformation (OT) Live Cursor Syncing:**
   - *Reason:* Replicating Yjs/WebSocket peer-to-peer real-time character-by-character typing within 4 hours risks state corruption under edge network conditions. We prioritized rock-solid document persistence, auto-save debouncing, and role permissions first.
2. **Comment Threads & Mentions:**
   - *Reason:* Comment UI adds complexity that distracts from the core editing and ingestion pipeline.

### What We Would Build With Another 2–4 Hours:
1. **Yjs + WebSockets / Live Cursors:** Integrate `y-webrtc` or `y-websocket` with TipTap for live cursor presence and multi-user typing.
2. **Inline Comments & Suggestion Mode:** Allow viewers to highlight text and leave threaded comments.
3. **Full-Text Vector Search:** Implement client/server indexing for instant search across document content.
