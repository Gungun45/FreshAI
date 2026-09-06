# Ajaia Docs — AI-Native Collaborative Document Editor

> **Assessment Submission for Ajaia LLC — Full Stack Product Engineer (AI-Native)**  
> **Candidate:** Vaibhavi Diwakar (`taniyadiwaker6@gmail.com`)

---

## 🌟 Overview

**Ajaia Docs** is a lightweight, full-stack collaborative document workspace inspired by Google Docs. Engineered with strict attention to product usability, robust data persistence, role-based access control, and seamless file ingestion, it provides a cohesive experience for product teams to author, collaborate, and manage documentation.

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- **Node.js**: v18.0.0 or later (Tested on Node v24 LTS)
- **npm**: v9.0.0 or later

### Installation & Run

1. **Navigate to the project directory:**
   ```bash
   cd ajaia-docs
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   Open [http://localhost:3000](http://localhost:3000)

5. **Run automated test suite:**
   ```bash
   npm test
   ```

---

## 🔑 Seeded Test Accounts (Mock Authentication)

To allow instant evaluation of the multi-user permission model without complex OAuth signups, the application includes 4 pre-configured team accounts accessible via the top-right **User Switcher**:

| User Name | Email | Role / Title |
| :--- | :--- | :--- |
| **Vaibhavi Diwakar** | `taniyadiwaker6@gmail.com` | Lead Product Engineer (Candidate / Owner) |
| **Alex Chen** | `alex.chen@ajaia.io` | Head of Product |
| **Jordan Taylor** | `jordan.taylor@ajaia.io` | Staff Platform Engineer |
| **Sarah Miller** | `sarah.miller@ajaia.io` | Design Director |

*Switch between users at any time to verify "Owned by Me" vs "Shared with Me" filters, write vs read-only modes, and permission granting.*

---

## 🛠 Core Product Capabilities

### 1. Document Creation & Rich-Text Editing
- **Interactive Formatting Toolbar:** Bold, Italic, Underline, Strikethrough, Heading 1, Heading 2, Heading 3, Bulleted Lists, Numbered Lists, Blockquotes, Code Blocks, Text Highlights, and Alignment.
- **Inline Renaming:** Rename document title directly in the header with instant auto-save.
- **Debounced Auto-Save & Manual Save:** Changes are automatically persisted with visual status pills (`Saved`, `Saving...`, `Unsaved`).
- **Live Metrics:** Real-time word count and character count statistics.

### 2. Multi-Format File Ingestion & Export
- **Supported Upload Formats:**
  - `.md` / `.markdown`: Converts markdown headers, quotes, lists, and bold/italics into editable rich text.
  - `.docx`: Converts Microsoft Word files into semantic HTML.
  - `.txt`: Preserves paragraphs and formatting structure.
  - `.json`: Ingests structured JSON document payloads.
- **Flexible Destinations:** Choose between creating a new document or importing into an active draft.
- **Export Options:** Export documents anytime as `.md`, `.txt`, or `.html`.

### 3. Role-Based Access Control (RBAC) & Sharing
- **Document Owner:** Can edit, rename, invite collaborators, change permission roles, or delete documents.
- **Editor:** Can read and modify document content and title.
- **Viewer:** Read-only mode with editor controls locked and warning banner displayed.
- **Visual Separation:** Tabbed filtering between *All Documents*, *Owned by Me*, and *Shared with Me*.

### 4. Persistence & Revision History
- **Atomic File Storage:** Durable JSON file storage with atomic writes and zero native compilation dependencies.
- **Revision Timeline:** Automatic snapshotting of past revisions on save, with one-click version restore.

---

## 🧪 Automated Testing

Run the automated test suite verifying CRUD operations, permissions enforcement, and markdown parser logic:
```bash
npm test
```

Test coverage includes:
- Seeded storage initialization
- Document creation with word/character count computation
- Editor update permissions validation
- **403 Unauthorized enforcement** on Viewer modification attempts
- Markdown to semantic HTML parser conversion
- Plain text to paragraph segmentation

---

## 📂 Project Structure

```
ajaia-docs/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── documents/
│   │   │   │   ├── route.ts            # GET (list), POST (create)
│   │   │   │   └── [id]/
│   │   │   │       ├── route.ts        # GET (doc), PUT (update), DELETE
│   │   │   │       └── share/route.ts  # POST (share), DELETE (collab)
│   │   │   ├── upload/route.ts         # Multipart file ingestion API
│   │   │   └── users/route.ts          # Seeded team members API
│   │   ├── doc/[id]/page.tsx           # Document editor page
│   │   ├── layout.tsx                  # Root layout & providers
│   │   ├── page.tsx                    # Main dashboard & documents view
│   │   └── globals.css                 # TipTap typography & theme
│   ├── components/
│   │   ├── Navbar.tsx                  # Brand bar & user switcher
│   │   ├── UserSwitcher.tsx            # Multi-user simulation selector
│   │   ├── DocumentList.tsx            # Filterable document grid
│   │   ├── RichEditor.tsx              # Google Docs-inspired rich text editor
│   │   ├── ShareModal.tsx              # Collaborator & role management
│   │   ├── FileUploadModal.tsx         # File ingestion modal (.md, .docx, .txt)
│   │   └── RevisionHistoryModal.tsx    # Version snapshots and restore
│   ├── context/
│   │   └── AuthContext.tsx             # Simulated session & user state
│   └── lib/
│       ├── fileParsers.ts              # Markdown, docx, and txt converters
│       ├── storage.ts                  # Atomic storage repository & permissions
│       ├── types.ts                    # TypeScript domain interfaces
│       └── users.ts                    # Seeded users configuration
├── data/
│   └── documents.json                  # Persistent JSON document database
├── tests/
│   └── test-suite.mjs                  # Automated test suite
├── ARCHITECTURE.md                     # Engineering tradeoffs & architecture
├── AI_WORKFLOW.md                      # AI-native development methodology
├── SUBMISSION.md                       # Ajaia submission markdown document
└── WALKTHROUGH_SCRIPT.md               # 3-5 min video recording script
```

---

## 🚢 Deployment

### Deploying to Vercel (Recommended)
1. Push this repository to GitHub.
2. Import the project into [Vercel](https://vercel.com).
3. Framework Preset: **Next.js**
4. Click **Deploy**.
