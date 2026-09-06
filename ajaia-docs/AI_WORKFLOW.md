# AI-Native Engineering Workflow Note

> **Assignment Submission for Ajaia LLC — Full Stack Product Engineer (AI-Native)**  
> **Candidate:** Vaibhavi Diwakar (`taniyadiwaker6@gmail.com`)

---

## 1. Which AI Tools Were Used
- **Antigravity / Gemini 3.7 Coding Agent:** Used for project scaffolding, type definitions generation, test suite synthesis, and component templating.
- **AI-Assisted Architecture Planning:** Used to outline tradeoff vectors (e.g., TipTap vs Slate vs Lexical; SQLite vs atomic JSON store; CRDT vs debounced REST persistence) to maintain tight delivery discipline within the timebox.

---

## 2. Where AI Materially Sped Up Work

1. **Scaffolding Rich-Text Extension Configuration:**
   - Generating standard boilerplate for TipTap starter kit, text-align, highlight, and underline extensions saved ~45 minutes of manual configuration.
2. **File Ingestion Regex & Mammoths Integration:**
   - Rapidly producing Markdown-to-HTML and Word `.docx` parsing routines with edge-case handling (nested lists, code blocks, quote blocks).
3. **Automated Test Generation:**
   - Synthesizing edge-case test matrices for permission enforcement (verifying that viewers attempting to update throw HTTP 403 / authorization errors).
4. **Design System & UI Component Speed:**
   - Crafting Tailwind CSS layouts for the Google Docs toolbar, sharing modal, and user switcher with responsive behavior.

---

## 3. What AI-Generated Output Was Changed or Rejected

1. **Rejected Heavy Real-Time WebSocket Daemon Scaffolding:**
   - *AI Initial Suggestion:* Attempting to spin up a full multi-server WebSocket / Redis PubSub daemon for live character cursors.
   - *Engineering Judgment:* Rejected as high-risk over-engineering for a 4–6 hour scope. Instead, implemented a rock-solid debounced auto-save engine, revision history snapshots, and role-based access control that works reliably without infrastructure failure points.
2. **Replaced Complex Database Migrations with Self-Healing Persistence:**
   - *AI Initial Suggestion:* Heavy SQLite binary bindings with native node-gyp compilation.
   - *Engineering Judgment:* Swapped for an atomic file-based persistence store that seeds automatically and runs without native OS compilation issues on any reviewer machine.
3. **Refactored DotAll Regex Flag for Universal Target Compatibility:**
   - Corrected standard ECMAScript regular expression flags in the Markdown parser to ensure seamless cross-platform TypeScript compilation.

---

## 4. How Correctness, UX Quality, and Reliability Were Verified

1. **Automated Unit & Integration Testing (`npm test`):**
   - Verified that all 5 core tests pass cleanly, checking document initialization, creation, role calculations, viewer mutation rejection, and Markdown parsing.
2. **Full Production Build Verification (`npm run build`):**
   - Successfully compiled the Next.js App Router with TypeScript type checking, ensuring zero type errors or broken route signatures.
3. **Interactive Manual UX Quality Audit:**
   - Tested real-time rich-text formatting (H1-H3, lists, bold, underline, code blocks).
   - Tested multi-user switching: logged in as Vaibhavi (Owner), created documents, shared with Alex (Editor) and Sarah (Viewer), switched accounts, and verified Sarah is locked in Read-Only mode.
   - Tested file uploads: ingested `.md` and `.txt` files and confirmed instant conversion into editable documents.
   - Tested version history: created revisions, inspected timeline, and verified version restore.
