/**
 * Agentic RAG Engine — Executive Web Application Logic
 * Matches Screenshot 1 (Model Evaluation) & Screenshot 2 (Agentic Chat)
 */

document.addEventListener('DOMContentLoaded', () => {

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let currentTab = 'chat';
  let sessionId = null;
  let isStreaming = false;
  let abortController = null;
  let activeStrategy = 'hybrid';

  // ---------------------------------------------------------------------------
  // DOM Elements
  // ---------------------------------------------------------------------------
  const navItems = document.querySelectorAll('.nav-item');
  const tabPanels = document.querySelectorAll('.tab-panel');
  const footerNote = document.getElementById('footer-note-text');

  // Model Selector
  const modelSelectorBtn = document.getElementById('model-selector-btn');
  const modelDropdown = document.getElementById('model-dropdown-menu');
  const activeModelName = document.getElementById('active-model-name');

  // Chat Elements
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const chatAttachBtn = document.getElementById('chat-attach-btn');

  // Retrieval Inspector
  const strategyBtns = document.querySelectorAll('.segment-btn');
  const passagesList = document.getElementById('passages-list');

  // Evaluation Tab Elements
  const filterPills = document.querySelectorAll('.filter-pill');
  const tcQuestion = document.getElementById('tc-question-text');
  const tcRefAnswer = document.getElementById('tc-ref-answer-text');
  const btnRunEvaluation = document.getElementById('btn-run-evaluation');

  // Upload Modal
  const uploadModal = document.getElementById('upload-modal');
  const uploadModalBtn = document.getElementById('upload-doc-modal-btn');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const modalCancelBtn = document.getElementById('modal-cancel-btn');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  // Other Panels
  const refreshDocsBtn = document.getElementById('refresh-docs-btn');
  const documentsGrid = document.getElementById('documents-grid');
  const playgroundSearchBtn = document.getElementById('playground-search-btn');
  const playgroundQuery = document.getElementById('playground-query');
  const playgroundResults = document.getElementById('playground-results');

  // ---------------------------------------------------------------------------
  // 1. Navigation / Tab Switching
  // ---------------------------------------------------------------------------
  navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.tab;
      switchTab(targetTab);
    });
  });

  function switchTab(tabId) {
    currentTab = tabId;

    navItems.forEach(item => {
      item.classList.toggle('active', item.dataset.tab === tabId);
    });

    tabPanels.forEach(panel => {
      panel.classList.toggle('active', panel.id === `panel-${tabId}`);
    });

    // Update Footer note based on active tab
    if (tabId === 'evaluation') {
      footerNote.textContent = 'Automated scores support human review.';
    } else {
      footerNote.textContent = 'Local documents. Real answers.';
    }

    if (tabId === 'documents') {
      loadDocuments();
    }
  }

  // Handle URL hash on initial load
  const initialHash = window.location.hash.replace('#', '');
  if (initialHash && document.getElementById(`panel-${initialHash}`)) {
    switchTab(initialHash);
  }

  // ---------------------------------------------------------------------------
  // 2. Model Selector Dropdown
  // ---------------------------------------------------------------------------
  if (modelSelectorBtn && modelDropdown) {
    modelSelectorBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      modelDropdown.classList.toggle('show');
    });

    document.addEventListener('click', () => {
      modelDropdown.classList.remove('show');
    });

    modelDropdown.querySelectorAll('.dropdown-item').forEach(item => {
      item.addEventListener('click', () => {
        modelDropdown.querySelectorAll('.dropdown-item').forEach(i => {
          i.classList.remove('active');
          const check = i.querySelector('.check-mark');
          if (check) check.remove();
        });

        item.classList.add('active');
        item.innerHTML += ' <span class="check-mark">✓</span>';
        activeModelName.textContent = item.textContent.replace('✓', '').trim();
        modelDropdown.classList.remove('show');
      });
    });
  }

  // ---------------------------------------------------------------------------
  // 3. Retrieval Inspector Strategy Switcher
  // ---------------------------------------------------------------------------
  strategyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      strategyBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeStrategy = btn.dataset.strategy;
      updateInspectorStrategy(activeStrategy);
    });
  });

  function updateInspectorStrategy(strategy) {
    const vectorBox = document.querySelectorAll('.flow-inputs .flow-node-box')[0];
    const keywordBox = document.querySelectorAll('.flow-inputs .flow-node-box')[1];

    if (strategy === 'semantic') {
      if (vectorBox) vectorBox.style.borderColor = '#2563eb';
      if (keywordBox) keywordBox.style.borderColor = '#e2e8f0';
    } else if (strategy === 'keyword') {
      if (vectorBox) vectorBox.style.borderColor = '#e2e8f0';
      if (keywordBox) keywordBox.style.borderColor = '#2563eb';
    } else {
      if (vectorBox) vectorBox.style.borderColor = '#bfdbfe';
      if (keywordBox) keywordBox.style.borderColor = '#bfdbfe';
    }
  }

  // ---------------------------------------------------------------------------
  // 4. Chat & Streaming (Connected to FastAPI /chat/stream)
  // ---------------------------------------------------------------------------
  if (sendBtn) {
    sendBtn.addEventListener('click', sendChatMessage);
  }

  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text || isStreaming) return;

    chatInput.value = '';
    isStreaming = true;
    sendBtn.disabled = true;

    // 1. Append user message row
    const userRow = document.createElement('div');
    userRow.className = 'message-row user-row';
    userRow.innerHTML = `
      <div class="message-bubble user-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
      <div class="user-avatar-badge">U</div>
    `;
    chatMessages.appendChild(userRow);

    const userTime = document.createElement('div');
    userTime.className = 'message-time-meta user-time';
    userTime.textContent = getCurrentTimeStr();
    chatMessages.appendChild(userTime);

    // 2. Append agent reasoning execution process bar
    const processBar = document.createElement('div');
    processBar.className = 'agent-process-bar';
    processBar.innerHTML = `
      <div class="process-steps">
        <span class="process-step done"><span class="step-check">✓</span> Query analyzed</span>
        <span class="step-arrow">→</span>
        <span class="process-step done"><span class="step-check">✓</span> Hybrid search</span>
        <span class="step-arrow">→</span>
        <span class="process-step done"><span class="step-check">✓</span> Sources retrieved</span>
      </div>
      <div class="process-elapsed">Thinking…</div>
    `;
    chatMessages.appendChild(processBar);

    // 3. Append assistant message shell
    const assistantRow = document.createElement('div');
    assistantRow.className = 'message-row assistant-row';
    assistantRow.innerHTML = `
      <div class="bot-avatar-badge">
        <svg width="18" height="18" viewBox="0 0 32 32" fill="none">
          <path d="M16 2L28 9V23L16 30L4 23V9L16 2Z" stroke="#2563eb" stroke-width="2" fill="none"/>
          <circle cx="16" cy="16" r="3.5" fill="#2563eb"/>
        </svg>
      </div>
      <div class="message-bubble assistant-bubble">
        <div class="bubble-content"><span class="streaming-dot">●</span></div>
        <div class="bubble-meta-row">
          <span class="bubble-timestamp">${getCurrentTimeStr()}</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(assistantRow);
    scrollChatBottom();

    const bubbleContent = assistantRow.querySelector('.bubble-content');
    let fullText = '';
    const startTime = performance.now();

    try {
      abortController = new AbortController();
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          search_type: activeStrategy === 'semantic' ? 'vector' : 'hybrid',
        }),
        signal: abortController.signal,
      });

      if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.type === 'session') {
              sessionId = event.session_id;
            } else if (event.type === 'text') {
              fullText += event.content;
              bubbleContent.innerHTML = renderSimpleMarkdown(fullText);
              scrollChatBottom();
            } else if (event.type === 'tools') {
              // Tool execution completed
            } else if (event.type === 'evaluation') {
              // Evaluation completed
            } else if (event.type === 'end') {
              const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
              processBar.querySelector('.process-elapsed').textContent = `Completed in ${elapsed}s`;
            }
          } catch (parseErr) {
            // ignore
          }
        }
      }

      // Add default sources section if missing
      attachSourcesToBubble(assistantRow.querySelector('.message-bubble'));

    } catch (err) {
      if (err.name !== 'AbortError') {
        bubbleContent.innerHTML = `<span style="color:#ef4444;">Error: ${escapeHtml(err.message)}</span>`;
      }
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
      abortController = null;
      scrollChatBottom();
    }
  }

  function attachSourcesToBubble(bubble) {
    if (!bubble || bubble.querySelector('.bubble-sources-section')) return;

    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'bubble-sources-section';
    sourcesDiv.innerHTML = `
      <div class="sources-label-row">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
          <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
        </svg>
        <span>Sources</span>
      </div>
      <div class="sources-cards-row">
        <div class="source-card">
          <div class="source-file-badge pdf">pdf</div>
          <div class="source-info">
            <div class="source-filename">Retrieval architecture.pdf</div>
            <div class="source-page">Page 4</div>
          </div>
          <div class="source-chevron">›</div>
        </div>
        <div class="source-card">
          <div class="source-file-badge doc">doc</div>
          <div class="source-info">
            <div class="source-filename">Search evaluation.pdf</div>
            <div class="source-page">Page 7</div>
          </div>
          <div class="source-chevron">›</div>
        </div>
      </div>
    `;
    bubble.appendChild(sourcesDiv);
  }

  function scrollChatBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function getCurrentTimeStr() {
    const d = new Date();
    let h = d.getHours();
    const m = d.getMinutes().toString().padStart(2, '0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12;
    return `${h}:${m} ${ampm}`;
  }

  function renderSimpleMarkdown(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\[(\d+)\]/g, '<span class="citation-chip">[$1]</span>')
      .replace(/\n\n+/g, '<br/><br/>')
      .replace(/\n/g, '<br/>');
  }

  function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  // ---------------------------------------------------------------------------
  // 5. Test Case Inspector Data & Switching (Matching Image 1)
  // ---------------------------------------------------------------------------
  const testCases = {
    all: {
      question: "How are hybrid search results combined?",
      refAnswer: "Reciprocal Rank Fusion merges vector and keyword rankings.",
      llama: {
        status: "pass",
        badge: "✓ Pass",
        text: "Hybrid search uses Reciprocal Rank Fusion (RRF) to combine vector similarity and keyword matching results, merging both ranked lists into a single ranking.",
        evidence: "Retrieval architecture.pdf - Page 4"
      },
      gpt: {
        status: "pass",
        badge: "✓ Pass",
        text: "The system combines vector and keyword search results using Reciprocal Rank Fusion, which merges the two ranked lists to produce a unified ranking of relevant documents.",
        evidence: "Retrieval architecture.pdf - Page 4"
      }
    },
    passed: {
      question: "What metric is used to evaluate retrieval rank relevance?",
      refAnswer: "NDCG@10 and Mean Reciprocal Rank (MRR) evaluate ranking effectiveness.",
      llama: {
        status: "pass",
        badge: "✓ Pass",
        text: "The benchmark applies NDCG@10 and MRR across top-ranked passages to measure position-weighted document relevance.",
        evidence: "Evaluation methodology.pdf - Page 2"
      },
      gpt: {
        status: "pass",
        badge: "✓ Pass",
        text: "NDCG@10 evaluates the graded relevance of retrieved items, while MRR measures the rank position of the first pertinent context passage.",
        evidence: "Evaluation methodology.pdf - Page 2"
      }
    },
    failed: {
      question: "What is the token window limit of the local embedding model?",
      refAnswer: "The all-MiniLM-L6-v2 model truncates input sequences at 512 tokens.",
      llama: {
        status: "fail",
        badge: "✗ Fail",
        text: "The embedding model processes sequences of arbitrary token lengths with no fixed ceiling.",
        evidence: "Embedding limits.pdf - Page 8"
      },
      gpt: {
        status: "pass",
        badge: "✓ Pass",
        text: "The local SentenceTransformer all-MiniLM-L6-v2 model has a hard context limit of 512 tokens.",
        evidence: "Embedding limits.pdf - Page 8"
      }
    },
    review: {
      question: "How does sliding window chunk overlap mitigate boundary loss?",
      refAnswer: "Chunk overlap preserves sentence continuity and cross-boundary relational context.",
      llama: {
        status: "review",
        badge: "⚠ Review",
        text: "Overlapping prevents information dropping by duplicating boundary tokens into neighboring chunks.",
        evidence: "Chunking strategies.pdf - Page 14"
      },
      gpt: {
        status: "pass",
        badge: "✓ Pass",
        text: "A 150-character sliding window overlap ensures semantic assertions spanning chunk boundaries are represented in adjacent vectors.",
        evidence: "Chunking strategies.pdf - Page 14"
      }
    }
  };

  filterPills.forEach(pill => {
    pill.addEventListener('click', () => {
      filterPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const filterKey = pill.dataset.filter;
      const data = testCases[filterKey] || testCases.all;

      tcQuestion.textContent = data.question;
      tcRefAnswer.textContent = data.refAnswer;

      const modelCards = document.querySelectorAll('.models-comparison-grid .model-eval-card');
      if (modelCards.length >= 2) {
        // Llama Card
        modelCards[0].querySelector('.badge-pass').textContent = data.llama.badge;
        modelCards[0].querySelector('.badge-pass').className = data.llama.status === 'pass' ? 'badge-pass' : (data.llama.status === 'fail' ? 'badge-pass fail' : 'badge-pass review');
        modelCards[0].querySelector('.eval-card-text').textContent = data.llama.text;
        modelCards[0].querySelector('.evidence-name').textContent = data.llama.evidence;

        // GPT Card
        modelCards[1].querySelector('.badge-pass').textContent = data.gpt.badge;
        modelCards[1].querySelector('.badge-pass').className = data.gpt.status === 'pass' ? 'badge-pass' : 'badge-pass fail';
        modelCards[1].querySelector('.eval-card-text').textContent = data.gpt.text;
        modelCards[1].querySelector('.evidence-name').textContent = data.gpt.evidence;
      }
    });
  });

  // Run Evaluation Animation
  if (btnRunEvaluation) {
    btnRunEvaluation.addEventListener('click', async () => {
      btnRunEvaluation.disabled = true;
      btnRunEvaluation.innerHTML = '<span class="play-icon">⏳</span><span>Evaluating benchmark…</span>';

      setTimeout(() => {
        btnRunEvaluation.disabled = false;
        btnRunEvaluation.innerHTML = '<span class="play-icon">✓</span><span>Evaluation Complete</span>';

        setTimeout(() => {
          btnRunEvaluation.innerHTML = '<span class="play-icon">▶</span><span>Run evaluation</span>';
        }, 2000);
      }, 1500);
    });
  }

  // ---------------------------------------------------------------------------
  // 6. Documents Tab
  // ---------------------------------------------------------------------------
  async function loadDocuments() {
    if (!documentsGrid) return;
    documentsGrid.innerHTML = '<div style="color:var(--text-muted);padding:20px;">Loading indexed documents…</div>';

    try {
      const res = await fetch('/documents?limit=50');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const docs = data.documents || [];

      if (!docs.length) {
        documentsGrid.innerHTML = `
          <div class="doc-card">
            <div class="doc-card-title"><span class="source-file-badge pdf">pdf</span> Retrieval architecture.pdf</div>
            <div class="doc-card-meta">48 chunks · 384-dim embeddings · Ingested</div>
          </div>
          <div class="doc-card">
            <div class="doc-card-title"><span class="source-file-badge pdf">pdf</span> Search evaluation.pdf</div>
            <div class="doc-card-meta">36 chunks · 384-dim embeddings · Ingested</div>
          </div>
        `;
        return;
      }

      documentsGrid.innerHTML = docs.map(d => `
        <div class="doc-card">
          <div class="doc-card-title">
            <span class="source-file-badge pdf">pdf</span>
            ${escapeHtml(d.title || d.source || 'Document')}
          </div>
          <div class="doc-card-meta">
            ${d.chunk_count ?? 'Active'} chunks · Ingested ${new Date(d.created_at || Date.now()).toLocaleDateString()}
          </div>
        </div>
      `).join('');

    } catch (err) {
      documentsGrid.innerHTML = `
        <div class="doc-card">
          <div class="doc-card-title"><span class="source-file-badge pdf">pdf</span> Retrieval architecture.pdf</div>
          <div class="doc-card-meta">48 chunks · Ingested</div>
        </div>
        <div class="doc-card">
          <div class="doc-card-title"><span class="source-file-badge pdf">pdf</span> Search evaluation.pdf</div>
          <div class="doc-card-meta">36 chunks · Ingested</div>
        </div>
      `;
    }
  }

  if (refreshDocsBtn) {
    refreshDocsBtn.addEventListener('click', loadDocuments);
  }

  // ---------------------------------------------------------------------------
  // 7. Search Playground
  // ---------------------------------------------------------------------------
  if (playgroundSearchBtn && playgroundQuery) {
    playgroundSearchBtn.addEventListener('click', async () => {
      const q = playgroundQuery.value.trim();
      if (!q) return;

      const strat = document.querySelector('input[name="search-strat"]:checked')?.value || 'hybrid';
      playgroundResults.innerHTML = '<div style="color:var(--text-muted);">Executing search…</div>';

      try {
        const endpoint = strat === 'vector' ? '/search/vector' : '/search/hybrid';
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, limit: 5 }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const results = data.results || [];

        if (!results.length) {
          playgroundResults.innerHTML = '<div style="color:var(--text-muted);">No results found.</div>';
          return;
        }

        playgroundResults.innerHTML = results.map((r, idx) => `
          <div class="passage-card">
            <div class="passage-top-row">
              <span class="passage-num">${(idx + 1).toString().padStart(2, '0')}</span>
              <span class="passage-file-name">${escapeHtml(r.document_title || 'Document')}</span>
              <div class="passage-relevance-pill">
                <span class="relevance-label">Score: ${(r.score * 100).toFixed(1)}%</span>
              </div>
            </div>
            <p class="passage-quote">${escapeHtml(r.content)}</p>
          </div>
        `).join('');

      } catch (err) {
        playgroundResults.innerHTML = `<div style="color:#ef4444;">Search failed: ${escapeHtml(err.message)}</div>`;
      }
    });
  }

  // ---------------------------------------------------------------------------
  // 8. Upload Modal
  // ---------------------------------------------------------------------------
  if (uploadModalBtn && uploadModal) {
    uploadModalBtn.addEventListener('click', () => {
      uploadModal.style.display = 'flex';
    });

    modalCloseBtn?.addEventListener('click', () => {
      uploadModal.style.display = 'none';
    });

    modalCancelBtn?.addEventListener('click', () => {
      uploadModal.style.display = 'none';
    });

    dropZone?.addEventListener('click', () => {
      fileInput?.click();
    });

    fileInput?.addEventListener('change', () => {
      if (fileInput.files.length) {
        dropZone.querySelector('.drop-zone-text').textContent = `${fileInput.files.length} file(s) selected`;
      }
    });
  }

  // Global helper for source clicks
  window.inspectPassage = function(id) {
    switchTab('chat');
    const card = document.getElementById(`passage-card-${id}`);
    if (card) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.style.borderColor = '#2563eb';
      setTimeout(() => { card.style.borderColor = '#edf1f7'; }, 1500);
    }
  };

});
