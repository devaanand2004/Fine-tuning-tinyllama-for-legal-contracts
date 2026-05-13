// ══════════════════════════════════════════════════════════════
// LegallyBound — Frontend Application Logic (SSE-optimised)
// ══════════════════════════════════════════════════════════════

const API = '';  // same origin

// ── State ────────────────────────────────────────────────────
let contractId   = null;
let contractName = null;
let numChunks    = 0;
let relevantChunks = 0;

// ── DOM refs ─────────────────────────────────────────────────
const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const uploadZone    = $('#upload-zone');
const fileInput     = $('#file-input');
const fileInfo      = $('#file-info');
const fileIcon      = $('#file-icon');
const fileName      = $('#file-name');
const fileMeta      = $('#file-meta');
const btnAnalyze    = $('#btn-analyze');
const btnClear      = $('#btn-clear');
const uploadProgress = $('#upload-progress');
const riskLoader    = $('#risk-loader');
const loaderText    = $('#risk-loader .loader__text');
const riskSummary   = $('#risk-summary-container');
const riskResults   = $('#risk-results');

const searchInput   = $('#search-input');
const btnSearch     = $('#btn-search');
const searchProgress = $('#search-progress');
const searchLoader  = $('#search-loader');
const searchResults = $('#search-results');

const toast         = $('#toast');
const toastIcon     = $('#toast-icon');
const toastText     = $('#toast-text');


// ══════════════════════════════════════════════════════════════
// TABS
// ══════════════════════════════════════════════════════════════
$$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.tab-btn').forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
        $$('.tab-content').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
        $(`#panel-${btn.dataset.tab}`).classList.add('active');
    });
});


// ══════════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════════
function showToast(message, type = 'success') {
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toastIcon.textContent = icons[type] || '✅';
    toastText.textContent = message;
    toast.className = `toast toast--${type} show`;
    setTimeout(() => toast.classList.remove('show'), 4000);
}


// ══════════════════════════════════════════════════════════════
// FILE UPLOAD
// ══════════════════════════════════════════════════════════════

// Drag & drop
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});


async function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt'].includes(ext)) {
        showToast('Only PDF and TXT files are supported', 'error');
        return;
    }

    // Show file info
    fileIcon.textContent = ext === 'pdf' ? '📕' : '📝';
    fileName.textContent = file.name;
    fileMeta.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    fileInfo.classList.add('visible');
    uploadProgress.classList.add('visible');

    // Upload
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res  = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();

        contractId     = data.contract_id;
        contractName   = data.filename;
        numChunks      = data.num_chunks;
        relevantChunks = data.relevant_chunks;

        fileMeta.textContent = `${data.num_chunks} chunks (${data.relevant_chunks} relevant) • ${(data.text_length / 1024).toFixed(1)} KB`;
        btnAnalyze.disabled  = false;

        // Enable search
        searchInput.disabled = false;
        searchInput.placeholder = `Ask about "${data.filename}"…`;
        btnSearch.disabled   = false;
        searchResults.innerHTML = `
            <div class="empty-state">
                <div class="empty-state__icon">✨</div>
                <div class="empty-state__title">Contract ready</div>
                <p>Ask any question about your contract in the search bar above</p>
            </div>`;

        showToast(`Uploaded "${data.filename}" — ${data.relevant_chunks} of ${data.num_chunks} chunks will be analyzed`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        uploadProgress.classList.remove('visible');
    }
}


// Clear
btnClear.addEventListener('click', () => {
    contractId   = null;
    contractName = null;
    fileInput.value = '';
    fileInfo.classList.remove('visible');
    btnAnalyze.disabled = true;
    riskSummary.innerHTML = '';
    riskResults.innerHTML = '';
    searchInput.disabled = true;
    searchInput.placeholder = 'Ask a question about your contract…';
    btnSearch.disabled = true;
    searchResults.innerHTML = `
        <div class="empty-state">
            <div class="empty-state__icon">🔍</div>
            <div class="empty-state__title">Upload a contract first</div>
            <p>Then search across all its clauses using natural language</p>
        </div>`;
});


// ══════════════════════════════════════════════════════════════
// RISK EXTRACTION — SSE STREAMING
// ══════════════════════════════════════════════════════════════
btnAnalyze.addEventListener('click', () => {
    if (!contractId) return;
    startStreamingRiskExtraction(contractId);
});


function startStreamingRiskExtraction(cid) {
    btnAnalyze.disabled = true;
    riskLoader.classList.add('visible');
    riskSummary.innerHTML = '';
    riskResults.innerHTML = '';

    const allRisks = [];
    let totalRelevant = 0;
    let totalSkipped  = 0;
    let startTime     = Date.now();

    // Show initial progress
    updateProgressHeader(0, 0, 0, 0);

    const evtSource = new EventSource(`${API}/api/risks/stream?contract_id=${cid}`);

    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
            case 'meta':
                totalRelevant = data.relevant;
                totalSkipped  = data.skipped;
                loaderText.textContent = `Analyzing ${totalRelevant} relevant clauses (${totalSkipped} boilerplate skipped)…`;
                updateProgressHeader(0, totalRelevant, totalSkipped, data.total);
                break;

            case 'risk':
                allRisks.push(data);
                // Update progress text
                loaderText.textContent = `Chunk ${data.index + 1} of ${data.total} — ${data.clause_type} (${data.time_seconds}s)`;
                // Append the card in real time
                appendRiskCard(data, allRisks.length - 1);
                // Update summary counts
                updateProgressHeader(allRisks.length, totalRelevant, totalSkipped, totalRelevant + totalSkipped);
                break;

            case 'done':
                evtSource.close();
                riskLoader.classList.remove('visible');
                btnAnalyze.disabled = false;

                const totalTime = data.total_time || ((Date.now() - startTime) / 1000);
                showToast(
                    `Done! Analyzed ${data.analyzed} clauses in ${totalTime.toFixed(0)}s (${data.skipped} skipped)`,
                    'success'
                );
                // Final summary update
                updateProgressHeader(allRisks.length, totalRelevant, totalSkipped, totalRelevant + totalSkipped, true);
                break;

            case 'error':
                evtSource.close();
                riskLoader.classList.remove('visible');
                btnAnalyze.disabled = false;
                showToast(data.message, 'error');
                break;
        }
    };

    evtSource.onerror = () => {
        evtSource.close();
        riskLoader.classList.remove('visible');
        btnAnalyze.disabled = false;
        showToast('Connection lost during risk extraction', 'error');
    };
}


function updateProgressHeader(analyzed, totalRelevant, skipped, totalChunks, final = false) {
    // Count by level from existing cards
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    document.querySelectorAll('.risk-card').forEach(card => {
        if (card.classList.contains('risk-card--HIGH'))   counts.HIGH++;
        if (card.classList.contains('risk-card--MEDIUM')) counts.MEDIUM++;
        if (card.classList.contains('risk-card--LOW'))    counts.LOW++;
    });

    const statusText = final
        ? `${analyzed} clauses analyzed • ${skipped} boilerplate skipped`
        : `${analyzed} / ${totalRelevant} clauses analyzed • ${skipped} skipped`;

    riskSummary.innerHTML = `
        <div class="results-header">
            <span class="results-header__title">📊 Risk Overview</span>
            <span class="results-header__count">${statusText}</span>
        </div>
        <div class="risk-summary">
            <div class="risk-stat risk-stat--high">
                <div class="risk-stat__value">${counts.HIGH}</div>
                <div class="risk-stat__label">High Risk</div>
            </div>
            <div class="risk-stat risk-stat--medium">
                <div class="risk-stat__value">${counts.MEDIUM}</div>
                <div class="risk-stat__label">Medium Risk</div>
            </div>
            <div class="risk-stat risk-stat--low">
                <div class="risk-stat__value">${counts.LOW}</div>
                <div class="risk-stat__label">Low Risk</div>
            </div>
        </div>
        ${!final ? `
        <div class="stream-progress">
            <div class="stream-progress__bar">
                <div class="stream-progress__fill" style="width: ${totalRelevant > 0 ? (analyzed / totalRelevant * 100) : 0}%"></div>
            </div>
            <span class="stream-progress__label">${totalRelevant > 0 ? Math.round(analyzed / totalRelevant * 100) : 0}%</span>
        </div>` : ''}`;
}


function appendRiskCard(r, index) {
    const riskIcons = { HIGH: '🔴', MEDIUM: '🟡', LOW: '🟢' };

    const card = document.createElement('div');
    card.className = `risk-card risk-card--${r.risk_level}`;
    card.style.animation = 'slideInRight 0.4s ease';
    card.innerHTML = `
        <div class="risk-card__header">
            <span class="risk-card__clause">
                ${riskIcons[r.risk_level] || '⚪'} ${escapeHtml(r.clause_type)}
            </span>
            <div style="display:flex; align-items:center; gap:0.5rem;">
                <span class="relevance-badge" title="Legal relevance score">📎 ${(r.relevance * 100).toFixed(0)}%</span>
                <span class="risk-badge risk-badge--${r.risk_level}">
                    ${r.risk_level} • ${(r.confidence * 100).toFixed(0)}%
                </span>
            </div>
        </div>
        <div class="risk-card__summary">${escapeHtml(r.summary)}</div>
        <div class="risk-card__reason">💡 ${escapeHtml(r.risk_reason)}</div>
        <button class="risk-card__toggle" onclick="toggleExcerpt(${index})">
            ▸ Show excerpt
        </button>
        <div class="risk-card__excerpt" id="excerpt-${index}">${escapeHtml(r.excerpt)}</div>
    `;

    riskResults.appendChild(card);

    // Scroll to the new card
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


function toggleExcerpt(index) {
    const el  = document.getElementById(`excerpt-${index}`);
    const btn = el.previousElementSibling;
    el.classList.toggle('expanded');
    btn.textContent = el.classList.contains('expanded') ? '▾ Hide excerpt' : '▸ Show excerpt';
}


// ══════════════════════════════════════════════════════════════
// SEMANTIC SEARCH
// ══════════════════════════════════════════════════════════════
btnSearch.addEventListener('click', doSearch);
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });


async function doSearch() {
    const query = searchInput.value.trim();
    if (!query || !contractId) return;

    btnSearch.disabled = true;
    searchProgress.classList.add('visible');
    searchLoader.classList.add('visible');
    searchResults.innerHTML = '';

    try {
        const res = await fetch(`${API}/api/search?contract_id=${contractId}&q=${encodeURIComponent(query)}&top_k=5`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Search failed');
        }
        const data = await res.json();
        renderSearchResults(data);
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        searchProgress.classList.remove('visible');
        searchLoader.classList.remove('visible');
        btnSearch.disabled = false;
    }
}


function renderSearchResults(data) {
    const riskIcons = { HIGH: '🔴', MEDIUM: '🟡', LOW: '🟢' };

    let html = `
        <div class="search-answer">
            <div class="search-answer__label">AI Answer</div>
            <div class="search-answer__text">${escapeHtml(data.answer)}</div>
            <div class="search-answer__risk">
                <span class="risk-badge risk-badge--${data.risk_level}">
                    ${riskIcons[data.risk_level] || '⚪'} ${data.risk_level} • ${(data.risk_confidence * 100).toFixed(0)}%
                </span>
                <span style="color: var(--text-muted); font-size: 0.85rem;">
                    ${escapeHtml(data.risk_reason)}
                </span>
            </div>
        </div>`;

    if (data.source_chunks && data.source_chunks.length) {
        html += `
            <div class="results-header">
                <span class="results-header__title">📎 Source Chunks</span>
                <span class="results-header__count">${data.source_chunks.length} matches</span>
            </div>
            <div class="source-chunks">`;

        data.source_chunks.forEach((chunk, i) => {
            html += `
                <div class="source-chunk">
                    <div class="source-chunk__header">
                        <span style="font-weight:600; font-size:0.85rem;">#${i + 1}</span>
                        <span class="source-chunk__score">Score: ${chunk.score.toFixed(4)}</span>
                    </div>
                    <div class="source-chunk__text">${escapeHtml(chunk.text)}</div>
                </div>`;
        });

        html += '</div>';
    }

    searchResults.innerHTML = html;
}


// ── Utility ──────────────────────────────────────────────────
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
