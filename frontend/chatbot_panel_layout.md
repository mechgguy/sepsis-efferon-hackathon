# Chatbot Panel — Real Layout Reference

Exact structure, colours, and measurements taken from
`frontend/components/research/research-chatbot.jsx`,
`frontend/css/components/app/pages/explore.scss`, and
`frontend/css/index.scss`.

---

## 1. Colour Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `panel-bg` | `#44546a` | Panel background, floating AI button background |
| `accent` | `#4effd0` | Turquoise — borders, scrollbar, spinner, streaming cursor, active tab indicator |
| `text-primary` | `#ffffff` | All text on panel |
| `text-muted` | `#aaaaaa` | Timestamps, secondary metadata |
| `text-italic-accent` | `#4effd0` | Intermediate/system message text |
| `shadow` | `rgba(68, 84, 106, 0.3)` | Panel drop-shadow |
| Header height | — | `$header-main-height: 55px` |

Tailwind extended palette (defined but secondary):
```
accent-orange  #ff7849
accent-blue    #4da6ff
accent-purple  #a78bfa
accent-green   #34d399
```

---

## 2. Overall Positioning

The panel is a **fixed right-side drawer**, 400 px wide, full viewport height minus the 55 px header.

```
Viewport
┌──────────────────────────────────────────────────────────┐
│  header (55px)                                           │
├─────────────────────────────────┬────────────────────────┤
│                                 │  Chatbot Panel         │
│  Main content                   │  position: fixed       │
│  width: calc(100% - 400px)      │  top: 55px             │
│  (shrinks when chat is open)    │  right: 0              │
│                                 │  width: 400px          │
│                                 │  height: calc(         │
│                                 │    100vh - 55px)       │
│                                 │                        │
└─────────────────────────────────┴────────────────────────┘
```

Mobile (≤768 px): panel width shrinks to **300 px**.

When the panel is **closed**, a floating "AI" button is shown instead:
```
position: fixed
bottom: 20px
right: 8px
z-index: 999999
```

---

## 3. Floating AI Button (panel closed)

```
┌──────────────┐
│ 🔵  AI       │   ← favicon.ico (20×20) + "AI" uppercase
└──────────────┘
```

**CSS:**
```
background:     #44546a
color:          #ffffff
border:         1px solid #ffffff
border-radius:  6px
padding:        12px 16px
font-size:      14px
font-weight:    500
letter-spacing: 0.1em
text-transform: uppercase
backdrop-filter: blur(8px)

:hover →
  border-color: #4effd0
  box-shadow: 0 6px 14px rgba(68,84,106,0.3),
              0 0 20px rgba(78,255,208,0.6),
              0 0 40px rgba(78,255,208,0.4)
```

---

## 4. Panel Shell (`.research-chatbot-dropdown`)

```
background:     #44546a
border:         2px solid #4effd0
border-radius:  12px
box-shadow:     0 8px 32px rgba(68,84,106,0.3)
backdrop-filter: blur(8px)
color:          #ffffff
font-family:    Inter, sans-serif
font-size:      14px
height:         calc(100vh - 85px)   /* 55px header + 15px top + 15px bottom */
overflow-y:     auto

scrollbar: thin, #4effd0 thumb on rgba(255,255,255,0.15) track
```

---

## 5. DOM Structure

```
.research-chatbot-dropdown           ← panel shell
└── .research-chatbot-container
    │
    ├── [close button]               ← top-right corner, ✕
    │
    ├── [history panel]              ← slides in from top (if user logged in)
    │   .research-chatbot-history-panel
    │   └── .research-chatbot-history-list
    │       └── .research-chatbot-history-item (×N)
    │           └── .research-chatbot-history-item-title
    │
    ├── [history toggle button]      ← clock SVG icon
    │   .research-chatbot-history-toggle
    │
    ├── .research-chatbot-tabs       ← tab bar
    │   ├── .research-chatbot-tab   "Chat"
    │   └── .research-chatbot-tab   "Documents"
    │       (active tab: .research-chatbot-tab-active)
    │
    ├── [CHAT TAB — when activeTab === 'chat']
    │   ├── .research-chatbot-messages     ← scrollable message list
    │   │   └── .research-chatbot-message .research-chatbot-message-{user|assistant|system}
    │   │       ├── .research-chatbot-message-content
    │   │       │   ├── [plain text or ReactMarkdown / MessageRenderer]
    │   │       │   ├── [spinner + italic text]  ← intermediate messages
    │   │       │   └── .research-chatbot-streaming-indicator  ▋
    │   │       └── .research-chatbot-message-time   (HH:MM)
    │   │
    │   └── [typing indicator — while isLoading]
    │       .research-chatbot-typing → <span/><span/><span/>
    │
    ├── [DOCUMENTS TAB — when activeTab === 'documents']
    │   .research-chatbot-documents
    │   ├── .research-chatbot-documents-header
    │   │   ├── <h3>Uploaded Documents</h3>
    │   │   └── .research-chatbot-documents-count   "N documents"
    │   │
    │   ├── [empty state]
    │   │   .research-chatbot-documents-empty
    │   │   ├── .research-chatbot-documents-empty-icon   📄
    │   │   ├── .research-chatbot-documents-empty-text   "No documents uploaded yet"
    │   │   └── .research-chatbot-documents-empty-hint   "Use the upload button (•)..."
    │   │
    │   └── .research-chatbot-documents-list
    │       └── .research-chatbot-document-item (×N)
    │           ├── .research-chatbot-document-icon   📕 🖼️ 📘 📊 🗺️ 📄 📎
    │           ├── .research-chatbot-document-info
    │           │   ├── .research-chatbot-document-name
    │           │   ├── .research-chatbot-document-meta   "size • date"
    │           │   ├── [button] "AI Extraction"          ← if extractable & not done
    │           │   ├── [button] "View Metadata"          ← if extraction complete
    │           │   ├── [progress bar] "Extracting... N%" ← while isExtracting
    │           │   └── [error] ❌ message + Retry button
    │           └── .research-chatbot-document-remove   × (remove button)
    │
    └── [INPUT SECTION — chat tab only]
        .research-chatbot-input-container
        │
        ├── [token row — if pending files or @datasets]
        │   .research-chatbot-tokens-container
        │   └── .research-chatbot-tokens-list
        │       ├── .research-chatbot-token .research-chatbot-token-dataset   "@name ×"
        │       └── .research-chatbot-token .research-chatbot-token-file      "@filename ×"
        │
        ├── [RAG banner — while indexing]
        │   .research-chatbot-rag-banner
        │   "⣿ Processing document — chat will be available once indexing is complete"
        │
        └── .research-chatbot-input-wrapper
            └── .research-chatbot-input-row
                ├── .research-chatbot-input-area
                │   ├── <textarea .research-chatbot-input>
                │   │   minHeight: 140px  maxHeight: 200px  overflowY: auto
                │   └── .dataset-dropdown  ← autocomplete (@-mention)
                └── .research-chatbot-actions
                    ├── .research-chatbot-action-button-upload   •
                    ├── .research-chatbot-action-button-send     →
                    └── .research-chatbot-action-button-new      +
```

---

## 6. Message Bubbles

### User message  (`.research-chatbot-message-user`)
```
align-self: flex-end  (right-aligned in the list)
plain text, white-space: pre-wrap
```

### Assistant message  (`.research-chatbot-message-assistant`)
```
align-self: flex-start
rendered via MessageRenderer / ReactMarkdown with remarkGfm
```

### System / intermediate  (`.research-chatbot-message-intermediate`)
```
.research-chatbot-message-content:
  background: transparent
  border: none
  padding: 0
  box-shadow: none

.research-chatbot-spinner-text-inline:
  font-size:   16px
  color:       #4effd0
  font-style:  italic
  font-weight: 500

Spinner: SVG progress-ring, stroke #4effd0, stroke-dashoffset animated
Completed: ✓ in #4effd0
```

### Streaming cursor
```
.research-chatbot-streaming-indicator  →  ▋  (block cursor appended while streaming)
```

### Typing indicator (loading)
```
.research-chatbot-typing  →  three <span> bouncing dots
```

### Timestamp
```
.research-chatbot-message-time
color: #888  font-size: ~11px  (HH:MM format)
```

---

## 7. Extraction Progress Bar

Shown while a PDF/image is being processed:

```
┌──────────────────────────────────┐
│ [████████░░░░░░░░░░░░░] 42%      │
│  Extracting... 42%               │
└──────────────────────────────────┘
```

Classes:
```
.research-chatbot-extraction-progress
└── .research-chatbot-extraction-progress-bar
    └── .research-chatbot-extraction-progress-fill   ← width: `${progress}%`
.research-chatbot-extraction-progress-text  →  "Extracting... N%"
```

The fill bar is turquoise `#4effd0`.

---

## 8. Close Button (`.research-chatbot-close-btn`)

Position: `absolute; top: 2px; right: 2px`

```
background:     rgba(78,255,208,0.15)
border:         1px solid #4effd0
border-radius:  6px
width: 24px  height: 24px
color: #ffffff  font-size: 12px
content: ✕

:hover →
  background:  rgba(78,255,208,0.25)
  box-shadow:  0 0 10px rgba(78,255,208,0.3)
  transform:   scale(1.1)
```

---

## 9. Streamlit Equivalents

| Original element | Streamlit equivalent |
|-----------------|---------------------|
| `research-chatbot-dropdown` (fixed panel) | `st.sidebar` or a right-column layout |
| `research-chatbot-tabs` | `st.tabs(["Chat", "Documents"])` |
| `research-chatbot-messages` scrollable list | `st.chat_message("user"/"assistant")` in a loop |
| Streaming indicator ▋ | `st.write_stream()` or manual placeholder |
| Typing dots | `st.spinner("Thinking...")` |
| Intermediate spinner + italic text | `st.status()` / `st.write()` inside `st.spinner` |
| Extraction progress bar | `st.progress(value, text="Extracting...")` |
| Token chips (pending files) | `st.pills()` or `st.multiselect` tags |
| RAG banner | `st.warning()` or `st.info()` |
| Textarea + send button | `st.chat_input()` (pinned at page bottom) |
| Upload button • | `st.file_uploader()` in Documents tab |
| New conversation + | `st.button("New conversation")` → clear `st.session_state.messages` |
| History sidebar | `st.session_state.history` list in a `st.expander` |
| Document list cards | `st.container(border=True)` per doc |
| Metadata modal | `st.dialog()` or `st.expander(expanded=True)` |

### Colour mapping for Streamlit custom CSS

```python
st.markdown("""
<style>
/* Panel background injected into the main container */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #44546a;
}
/* Accent colour for borders and active elements */
:root {
    --accent: #4effd0;
}
/* Chat input border */
div[data-testid="stChatInput"] textarea {
    border: 1px solid #4effd0;
    background-color: rgba(68,84,106,0.6);
    color: #ffffff;
}
/* Tab active indicator */
button[data-testid="stTab"][aria-selected="true"] {
    border-bottom: 2px solid #4effd0;
    color: #4effd0;
}
</style>
""", unsafe_allow_html=True)
```

---

## 10. Minimal CSS to Reproduce the Panel in Streamlit

Place this block at the top of `app.py` inside `st.markdown(... unsafe_allow_html=True)`:

```css
/* Panel shell */
.stApp { background: #1e1e1e; }

section[data-testid="stSidebar"] > div:first-child {
    background: #44546a;
    border-right: 2px solid #4effd0;
    box-shadow: 0 8px 32px rgba(68,84,106,0.3);
}

/* Headings and body text */
section[data-testid="stSidebar"] * { color: #ffffff; }

/* Progress bars */
div[data-testid="stProgress"] > div { background: #4effd0; }

/* Chat input */
div[data-testid="stChatInput"] textarea {
    background: rgba(68,84,106,0.8);
    border: 1px solid #4effd0;
    color: #ffffff;
}

/* Active tab */
button[data-testid="stTab"][aria-selected="true"] {
    border-bottom: 2px solid #4effd0;
    color: #4effd0;
}

/* Scrollbar */
* {
    scrollbar-color: #4effd0 rgba(255,255,255,0.1);
    scrollbar-width: thin;
}
```
