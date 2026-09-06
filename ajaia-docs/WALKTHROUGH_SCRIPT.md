# 3–5 Minute Walkthrough Video Recording Script

> **Candidate:** Vaibhavi Diwakar  
> **Role:** Full Stack Product Engineer (AI-Native) — Ajaia LLC  
> **Topic:** Collaborative Document Editor Demo & Technical Walkthrough

---

## ⏱️ Video Breakdown (Total Time: ~4:00 Minutes)

### 0:00 – 0:45 | Introduction & Objective
- **On Screen:** Start on the Home Dashboard (`http://localhost:3000`).
- **Talking Points:**
  - *"Hi everyone, I'm Vaibhavi Diwakar. Today I'm presenting my full-stack submission for the Ajaia Product Engineer assignment: a lightweight, collaborative document editor inspired by Google Docs."*
  - *"My goal was to ship a high-quality, production-ready product slice that excels at document authoring, file ingestion, role-based sharing, and state persistence."*

---

### 0:45 – 2:00 | Core User Flows & End-to-End Features
- **On Screen Actions:**
  1. **Create & Edit Document:**
     - Click **"New Document"**.
     - Rename document title in the header to *"Ajaia Q3 Roadmap"*.
     - Type headings, apply **Bold**, *Italic*, <u>Underline</u>, Bulleted lists, and Quotes using the toolbar.
     - Show the live **Word / Character count** updating at the bottom.
     - Highlight the **"Saved" status indicator** showing auto-save debouncing.
  2. **File Ingestion:**
     - Click **"Import"** or the Ingestion button.
     - Drag/upload a `.md` or `.docx` file.
     - Show how the parser converts markdown/word formatting directly into the active editor.
  3. **Export Document:**
     - Click **"Export"** and download the document as Markdown (`.md`) or HTML.

---

### 2:00 – 3:00 | Sharing, Permissions & Multi-User Simulation
- **On Screen Actions:**
  1. Open the **Share Modal**.
  2. Show that **Vaibhavi Diwakar** is the Owner.
  3. Invite **Alex Chen** as an *Editor* and **Sarah Miller** as a *Viewer*.
  4. Use the top-right **User Switcher** to switch active persona to **Sarah Miller (Viewer)**:
     - Show the **"Read-Only Mode"** banner.
     - Demonstrate that the editor locks and formatting tools are disabled for viewers.
  5. Go back to the Dashboard:
     - Show the **"Owned by Me"** vs **"Shared with Me"** tabbed filters updating dynamically.

---

### 3:00 – 3:45 | Architecture Decisions & Scope Prioritization
- **On Screen:** Briefly show `ARCHITECTURE.md` or the terminal running `npm test`.
- **Talking Points:**
  - **Stack:** Built with Next.js 16 App Router, TypeScript, Tailwind CSS, TipTap/ProseMirror, and an atomic file persistence layer.
  - **Prioritized:** Reliable debounced auto-save, structured document parsing, and strict server-side RBAC validation (Owner vs Editor vs Viewer).
  - **Intentionally Deprioritized:** Real-time peer-to-peer live cursor syncing (CRDTs) was scoped out to avoid network edge race conditions within the 4–6 hour timebox.
  - **Next 2–4 Hours:** If given another 2–4 hours, I would introduce WebSocket live cursor presence and inline comment threads.

---

### 3:45 – 4:15 | AI-Native Workflow & Closing
- **Talking Points:**
  - *"As an AI-native engineer, I used Gemini and modern AI coding agents to accelerate boilerplate creation, parser regex routines, and unit test generation."*
  - *"Crucially, I maintained engineering judgment: rejecting over-engineered Redis/WebSocket suggestions and validating correctness via our automated test suite and strict type checking."*
  - *"Thank you for your time and review! I look forward to the next steps with the Ajaia team."*
