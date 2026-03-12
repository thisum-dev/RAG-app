/**
 * static/script.js
 *
 * Handles all frontend logic:
 *   - PDF upload via the Fetch API (multipart/form-data)
 *   - Chat request/response via the Fetch API (JSON)
 *   - Maintaining a per-tab session_id using crypto.randomUUID()
 *   - Dynamically rendering user/assistant messages and sources
 */

// ── Session ID ──────────────────────────────────────────────────────────────
// Generate once per page load. Persists across messages in the same tab.
const SESSION_ID = crypto.randomUUID();

// ── DOM references ───────────────────────────────────────────────────────────
const pdfFileInput   = document.getElementById("pdf-file");
const uploadBtn      = document.getElementById("upload-btn");
const uploadStatus   = document.getElementById("upload-status");
const chatWindow     = document.getElementById("chat-window");
const userInput      = document.getElementById("user-input");
const sendBtn        = document.getElementById("send-btn");

// ── Upload handler ───────────────────────────────────────────────────────────
uploadBtn.addEventListener("click", async () => {
  const file = pdfFileInput.files[0];
  if (!file) {
    setUploadStatus("Please select a PDF file first.", false);
    return;
  }

  uploadBtn.disabled = true;
  setUploadStatus("Uploading and indexing… this may take a moment.", null);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      setUploadStatus(
        `✅ "${data.file}" indexed successfully (${data.chunks_indexed} chunks).`,
        true
      );
    } else {
      setUploadStatus(`❌ Error: ${data.error}`, false);
    }
  } catch (err) {
    setUploadStatus(`❌ Network error: ${err.message}`, false);
  } finally {
    uploadBtn.disabled = false;
  }
});

/** Show upload status text with colour coding. */
function setUploadStatus(message, success) {
  uploadStatus.textContent = message;
  uploadStatus.className = success === true  ? "status-ok"
                         : success === false ? "status-err"
                         : "";               // null → neutral (e.g. loading)
}

// ── Chat handlers ─────────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", (e) => {
  // Send on Enter (but allow Shift+Enter for newlines if ever needed)
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const question = userInput.value.trim();
  if (!question) return;

  // Render user's bubble immediately
  appendMessage("user", question);
  userInput.value = "";
  setInputLocked(true);

  // Show animated typing indicator while waiting
  const typingEl = appendTypingIndicator();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, session_id: SESSION_ID }),
    });

    const data = await response.json();

    // Remove typing indicator before rendering the real answer
    typingEl.remove();

    if (response.ok) {
      appendBotMessage(data.answer, data.sources || []);
    } else {
      appendBotMessage(`⚠️ ${data.error}`, []);
    }
  } catch (err) {
    typingEl.remove();
    appendBotMessage(`⚠️ Network error: ${err.message}`, []);
  } finally {
    setInputLocked(false);
    userInput.focus();
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

/** Append a simple user or bot text bubble. */
function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  chatWindow.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

/**
 * Append a bot message with an optional "Sources" toggle.
 * Sources are shown as a collapsible list below the bubble.
 */
function appendBotMessage(text, sources) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg bot";

  // Answer bubble
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  // Sources section (only if there are any)
  if (sources && sources.length > 0) {
    const toggleBtn = document.createElement("button");
    toggleBtn.className = "sources-toggle";
    toggleBtn.textContent = `▸ Show ${sources.length} source(s)`;

    const sourcesList = document.createElement("ul");
    sourcesList.className = "sources-list";

    sources.forEach((src, i) => {
      const li = document.createElement("li");
      li.textContent = `[${i + 1}] ${src}…`;
      sourcesList.appendChild(li);
    });

    // Toggle open/close
    toggleBtn.addEventListener("click", () => {
      const isOpen = sourcesList.classList.toggle("open");
      toggleBtn.textContent = isOpen
        ? `▾ Hide sources`
        : `▸ Show ${sources.length} source(s)`;
    });

    wrapper.appendChild(toggleBtn);
    wrapper.appendChild(sourcesList);
  }

  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

/** Show an animated typing indicator while waiting for the server. */
function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "msg bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing";

  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("div");
    dot.className = "dot";
    bubble.appendChild(dot);
  }

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

/** Enable or disable the chat input + send button. */
function setInputLocked(locked) {
  userInput.disabled = locked;
  sendBtn.disabled   = locked;
}

/** Scroll the chat window to the latest message. */
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
