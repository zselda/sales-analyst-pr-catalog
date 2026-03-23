/**
 * Financial Intelligence Platform — Frontend Application v3.0
 * =============================================================
 * Enhanced with:
 * - Agent metrics visualization
 * - Evaluation score display
 * - Parallel execution timeline
 * - Pipeline architecture awareness
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const API_BASE = 'http://localhost:8000';

let uploadSessionId = null;
let cy = null;

// ============================================================================
// STATUS MANAGEMENT
// ============================================================================

function setStatus(status, text) {
    const dot = document.querySelector('.status-dot');
    const textEl = document.querySelector('.status-text');
    dot.className = 'status-dot';
    if (status === 'running') dot.classList.add('running');
    if (status === 'error') dot.classList.add('error');
    textEl.textContent = text;
}

function addLog(message, type = 'info') {
    const container = document.getElementById('logEntries');
    const timestamp = new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${timestamp}] ${message}`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

function clearLog() {
    document.getElementById('logEntries').innerHTML = '';
}

// ============================================================================
// FILE UPLOAD HANDLING
// ============================================================================

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadZone').classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadZone').classList.remove('drag-over');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadZone').classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadMizanFile(files[0]);
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) uploadMizanFile(files[0]);
}

async function uploadMizanFile(file) {
    if (!file.name.endsWith('.xlsx')) {
        addLog('❌ Only .xlsx files are accepted', 'error');
        return;
    }

    const uploadContent = document.getElementById('uploadContent');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadSpinner = document.getElementById('uploadSpinner');
    const uploadResult = document.getElementById('uploadResult');

    uploadContent.style.display = 'none';
    uploadStatus.style.display = 'flex';
    uploadSpinner.style.display = 'block';
    uploadResult.innerHTML = `<strong>Uploading ${file.name}...</strong>`;

    addLog(`📤 Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, 'running');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch(`${API_BASE}/api/upload-mizan`, {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();

        uploadSessionId = data.session_id;
        uploadSpinner.style.display = 'none';
        uploadResult.innerHTML = `
            <div class="upload-success">
                <strong>✅ ${data.filename}</strong><br>
                <small>${data.mizan_count} accounts parsed</small>
            </div>
        `;

        // Update entity summary
        document.getElementById('entitySummary').style.display = 'block';
        document.getElementById('accountCount').textContent = data.mizan_count;
        document.getElementById('customerCount').textContent = data.customer_count;
        document.getElementById('supplierCount').textContent = data.supplier_count;
        document.getElementById('txnCount').textContent = data.transaction_count > 0
            ? `${data.transaction_count} (generated)` : 'N/A';

        // Show Upload button if no transactions
        const btnUploadTxn = document.getElementById('btnUploadTxn');
        if (data.transaction_count === 0) {
            btnUploadTxn.style.display = 'inline-block';
        } else {
            btnUploadTxn.style.display = 'none';
        }

        // Update company hint with name and sector
        const companyLabel = `${data.company_name || 'Company'} • ${data.sector || 'General'}`;
        document.getElementById('companyHint').textContent = companyLabel;

        // Show data preview
        if (data.preview && data.preview.length > 0) {
            showDataPreview(data.preview, `Mizan: ${data.mizan_count} accounts`);
        }

        addLog(`✅ Parsed ${data.mizan_count} accounts`, 'success');
        addLog(`  🏢 Company: ${data.company_name || 'Company'} | Sector: ${data.sector || 'General'}`, 'success');
        addLog(`  🟢 ${data.customer_count} customers | 🔴 ${data.supplier_count} suppliers`, 'success');
        if (data.customers_sample && data.customers_sample.length > 0) {
            addLog(`  Sample customers: ${data.customers_sample.join(', ')}`, 'info');
        }

    } catch (err) {
        uploadSpinner.style.display = 'none';
        uploadResult.innerHTML = `<span style="color:#ff4757;">❌ ${err.message}</span>`;
        addLog(`❌ Upload failed: ${err.message}`, 'error');
    }
}

async function uploadTransactions(file) {
    if (!uploadSessionId) {
        addLog('❌ Upload a Mizan file first', 'error');
        return;
    }
    if (!file) return;

    const btn = document.getElementById('btnUploadTxn');
    btn.textContent = '⏳...';
    btn.disabled = true;
    addLog(`📋 Uploading transactions: ${file.name}...`, 'running');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch(`${API_BASE}/api/upload-transactions?session_id=${uploadSessionId}`, {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        document.getElementById('txnCount').textContent = `${data.transaction_count} (${data.incoming} in / ${data.outgoing} out)`;
        btn.style.display = 'none';
        addLog(`✅ Loaded ${data.transaction_count} transactions from ${data.filename}`, 'success');
    } catch (err) {
        btn.textContent = 'Upload';
        btn.disabled = false;
        addLog(`❌ Transaction upload failed: ${err.message}`, 'error');
        // Reset file input so user can try again
        document.getElementById('txnFileInput').value = '';
    }
}

function showDataPreview(data, label) {
    const badge = document.getElementById('previewBadge');
    const wrapper = document.getElementById('previewTableWrapper');
    const thead = document.getElementById('previewHead');
    const tbody = document.getElementById('previewBody');
    badge.textContent = label;
    badge.classList.add('active');
    wrapper.style.display = 'block';
    if (!data || data.length === 0) return;
    const cols = Object.keys(data[0]).slice(0, 5);
    thead.innerHTML = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
    tbody.innerHTML = data.slice(0, 10).map(row => {
        return '<tr>' + cols.map(c => {
            let val = row[c];
            if (typeof val === 'number') val = val.toLocaleString('tr-TR');
            if (typeof val === 'string' && val.length > 20) val = val.substr(0, 20) + '…';
            return `<td>${val ?? ''}</td>`;
        }).join('') + '</tr>';
    }).join('');
}

// ============================================================================
// SWARM EXECUTION
// ============================================================================

async function runSwarm() {
    const btn = document.getElementById('btnRunSwarm');
    const taxId = document.getElementById('taxId').value.trim();
    if (!taxId) { addLog('Please enter a Tax ID', 'error'); return; }

    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Swarm Running...';
    setStatus('running', 'Executing Parallel Swarm...');

    clearLog();
    addLog('🚀 Intelligence Swarm activated (parallel mode)', 'running');
    addLog(`Target: Tax ID ${taxId}`, 'info');

    // Simulate agent progress
    const agents = [
        { name: 'Data Ingestion Agent', delay: 400, icon: '📥' },
        { name: 'Quantitative Analyst Agent', delay: 800, icon: '📊' },
        { name: 'Verifier Agent (checking...)', delay: 1200, icon: '✓' },
        { name: 'Network Mapper Agent', delay: 1600, icon: '🌐' },
        { name: 'Sales Strategist Agent', delay: 2000, icon: '🎯' },
    ];
    agents.forEach(a => {
        setTimeout(() => addLog(`${a.icon} ${a.name} executing...`, 'running'), a.delay);
    });

    try {
        const resp = await fetch(`${API_BASE}/api/run-swarm`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tax_id: taxId, session_id: uploadSessionId }),
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }
        const result = await resp.json();

        setStatus('success', 'Swarm Complete ✅');
        addLog('✅ All agents completed successfully', 'success');
        addLog(`Verification: ${result.verification_status} (${result.retry_count} attempts)`,
            result.verification_status === 'approved' ? 'success' : 'error');

        // Show evaluation score
        if (result.evaluation) {
            const score = result.evaluation.overall_score;
            const passed = result.evaluation.passed;
            addLog(`📊 Quality Score: ${(score * 100).toFixed(1)}% ${passed ? '✅' : '❌'}`, passed ? 'success' : 'error');
            showEvalBadge(score, passed);
        }

        // Show agent metrics
        if (result.agent_metrics) {
            renderAgentMetrics(result.agent_metrics);
        }

        // Render results
        renderNetworkGraph(result.network_data);
        renderStrategyReport(result.strategy_report);
        renderMetricsTab(result);

        // Show PDF download button
        document.getElementById('btnDownloadPdf').style.display = 'inline-flex';

        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🚀</span> Re-Run Intelligence Swarm';

    } catch (err) {
        setStatus('error', 'Swarm Failed');
        addLog(`❌ Swarm failed: ${err.message}`, 'error');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🚀</span> Retry Intelligence Swarm';
    }
}

// ============================================================================
// EVALUATION & METRICS DISPLAY
// ============================================================================

function showEvalBadge(score, passed) {
    const badge = document.getElementById('evalScoreBadge');
    const value = document.getElementById('evalScoreValue');
    badge.style.display = 'flex';
    value.textContent = `${(score * 100).toFixed(0)}%`;
    value.className = `eval-value ${passed ? 'passed' : 'failed'}`;
}

function renderAgentMetrics(metrics) {
    const panel = document.getElementById('metricsPanel');
    const body = document.getElementById('metricsBody');
    panel.style.display = 'block';

    const agents = Object.entries(metrics).sort((a, b) => {
        const order = ['data_ingestion', 'quant_analyst', 'verifier', 'network_mapper', 'strategist'];
        return order.indexOf(a[0]) - order.indexOf(b[0]);
    });

    body.innerHTML = agents.map(([name, m]) => {
        const timeFormatted = m.execution_time_ms > 1000
            ? `${(m.execution_time_ms / 1000).toFixed(1)}s`
            : `${Math.round(m.execution_time_ms)}ms`;
        const statusIcon = m.status === 'success' ? '✅' : '❌';
        const llmBadge = m.llm_calls > 0 ? `<span class="metric-llm">${m.llm_calls} LLM</span>` : '';

        return `
            <div class="metric-row ${m.status}">
                <span class="metric-name">${statusIcon} ${formatAgentName(name)}</span>
                <span class="metric-time">${timeFormatted} ${llmBadge}</span>
            </div>
        `;
    }).join('');
}

function formatAgentName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function renderMetricsTab(result) {
    const container = document.getElementById('metricsContainer');
    const metrics = result.agent_metrics || {};
    const evaluation = result.evaluation || {};

    // Build metrics display
    let html = `
            < h1 >📊 Pipeline Metrics & Evaluation</h1 >
        <h2>🎯 Quality Evaluation</h2>
        <table>
            <thead><tr><th>Dimension</th><th>Score</th><th>Status</th></tr></thead>
            <tbody>
    `;

    if (evaluation.scores) {
        const dimLabels = {
            data_completeness: '📥 Data Completeness',
            ratio_accuracy: '📊 Ratio Accuracy',
            network_coverage: '🌐 Network Coverage',
            strategy_quality: '📈 Strategy Quality',
        };
        Object.entries(evaluation.scores).forEach(([dim, score]) => {
            const pct = (score * 100).toFixed(0);
            const status = score >= 0.7 ? '✅' : score >= 0.4 ? '⚠️' : '❌';
            html += `<tr><td>${dimLabels[dim] || dim}</td><td><strong>${pct}%</strong></td><td>${status}</td></tr>`;
        });
        html += `</tbody></table>`;
        html += `< p > <strong>Overall Score: ${(evaluation.overall_score * 100).toFixed(1)}%</strong> — ${evaluation.passed ? '✅ PASSED' : '❌ FAILED'}</p > `;
    }

    html += `< h2 >⚡ Agent Execution Times</h2 > `;
    html += `< table ><thead><tr><th>Agent</th><th>Time</th><th>LLM Calls</th><th>Status</th></tr></thead><tbody>`;

    Object.entries(metrics).forEach(([name, m]) => {
        const time = m.execution_time_ms > 1000
            ? `${(m.execution_time_ms / 1000).toFixed(2)}s`
            : `${Math.round(m.execution_time_ms)}ms`;
        const badge = '';
        html += `<tr>
            <td>${formatAgentName(name)}${badge}</td>
            <td><strong>${time}</strong></td>
            <td>${m.llm_calls || 0}</td>
            <td>${m.status === 'success' ? '✅' : '❌'}</td>
        </tr>`;
    });
    html += `</tbody></table > `;

    // Pipeline architecture diagram
    html += `
            < h2 >🏗️ Pipeline Architecture</h2 >
        <div class="arch-diagram">
            <div class="arch-node seq">data_ingestion</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node seq">quant_analyst</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node seq">verifier</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node seq">network_mapper</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node seq">strategist</div>
        </div>
        <p style="font-size:11px;color:var(--text-muted);margin-top:8px;">
            Sequential pipeline — verifier loops back to quant_analyst on rejection
        </p>
        `;

    html += `< hr > <p style="font-size:11px;color:var(--text-muted);">Generated by Financial Intelligence Platform v3.0 — Parallel Multi-Agent Architecture</p>`;

    container.innerHTML = html;
}

// ============================================================================
// NETWORK GRAPH (CYTOSCAPE.JS)
// ============================================================================

function renderNetworkGraph(networkData) {
    if (!networkData || !networkData.nodes) return;
    const placeholder = document.getElementById('graphPlaceholder');
    if (placeholder) placeholder.style.display = 'none';

    const elements = [];
    networkData.nodes.forEach(node => {
        elements.push({
            group: 'nodes', data: {
                id: node.id, label: node.label, type: node.type,
                color: node.color, nodeSize: node.size || 30,
                balance: node.balance || 0, account_code: node.account_code || '',
            }
        });
    });
    networkData.edges.forEach(edge => {
        elements.push({
            group: 'edges', data: {
                source: edge.source, target: edge.target, weight: edge.weight,
                label: edge.label, type: edge.type, color: edge.color,
            }
        });
    });

    if (cy) cy.destroy();

    cy = cytoscape({
        container: document.getElementById('cyGraph'),
        elements: elements,
        style: [
            {
                selector: 'node', style: {
                    'label': 'data(label)', 'background-color': 'data(color)',
                    'width': 'data(nodeSize)', 'height': 'data(nodeSize)',
                    'color': '#e8ecf4', 'font-size': '10px', 'font-family': 'Inter, sans-serif',
                    'font-weight': 500, 'text-valign': 'bottom', 'text-halign': 'center',
                    'text-margin-y': 8, 'text-outline-color': '#0a0e1a', 'text-outline-width': 2,
                    'border-width': 2, 'border-color': 'data(color)', 'border-opacity': 0.4,
                    'overlay-padding': '6px',
                }
            },
            {
                selector: 'node[type="target"]', style: {
                    'width': 70, 'height': 70, 'font-size': '12px', 'font-weight': 700,
                    'border-width': 3, 'border-color': '#FFD700', 'background-opacity': 0.9,
                    'text-margin-y': 10,
                }
            },
            {
                selector: 'edge', style: {
                    'width': 2, 'line-color': 'data(color)', 'target-arrow-color': 'data(color)',
                    'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'opacity': 0.7,
                    'label': 'data(label)', 'font-size': '8px', 'font-family': 'JetBrains Mono, monospace',
                    'color': '#8b95b0', 'text-outline-color': '#0a0e1a', 'text-outline-width': 1.5,
                    'text-rotation': 'autorotate', 'arrow-scale': 1.2,
                }
            },
            {
                selector: 'node:active', style: {
                    'overlay-opacity': 0.2, 'overlay-color': '#667eea',
                }
            },
        ],
        layout: {
            name: 'cose', idealEdgeLength: 180, nodeOverlap: 30, refresh: 20,
            fit: true, padding: 50, randomize: false, componentSpacing: 120,
            nodeRepulsion: () => 8000, edgeElasticity: () => 150,
            nestingFactor: 1.2, gravity: 80, numIter: 1000,
            animate: true, animationDuration: 800,
        },
        minZoom: 0.3, maxZoom: 3, wheelSensitivity: 0.3,
    });

    cy.on('tap', 'node', function (evt) {
        const data = evt.target.data();
        const detailEl = document.getElementById('nodeDetail');
        const nameEl = document.getElementById('nodeDetailName');
        const infoEl = document.getElementById('nodeDetailInfo');
        nameEl.textContent = data.label;
        nameEl.style.color = data.color;
        let info = `< strong > Type:</strong > ${data.type.charAt(0).toUpperCase() + data.type.slice(1)} <br>`;
        if (data.account_code) info += `<strong>Account:</strong> ${data.account_code}<br>`;
        if (data.balance) info += `<strong>Balance:</strong> ₺${data.balance.toLocaleString('tr-TR')}<br>`;
        info += `<strong>Connections:</strong> ${evt.target.connectedEdges().length}`;
        infoEl.innerHTML = info;
        detailEl.style.display = 'block';
    });

    cy.on('tap', function (evt) {
        if (evt.target === cy) document.getElementById('nodeDetail').style.display = 'none';
    });

    addLog(`Graph rendered: ${networkData.stats.total_nodes} nodes, ${networkData.stats.total_edges} edges`, 'success');
}

// ============================================================================
// STRATEGY REPORT
// ============================================================================

function renderStrategyReport(markdown) {
    if (!markdown) return;
    document.getElementById('reportContainer').innerHTML = marked.parse(markdown);
    addLog('Strategy report rendered', 'success');
}

// ============================================================================
// CHAT INTERFACE
// ============================================================================

async function sendChat() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    appendChatMessage(message, 'user');

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        appendChatMessage(data.response, 'bot');
    } catch (err) {
        appendChatMessage(`Error: ${err.message}. Make sure the backend is running.`, 'bot');
    }
}

function appendChatMessage(text, sender) {
    const container = document.getElementById('chatMessages');
    const msg = document.createElement('div');
    msg.className = `chat-msg ${sender}`;
    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🤖';
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    bubble.innerHTML = sender === 'bot' ? marked.parse(text) : text;
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

// ============================================================================
// TAB SWITCHING
// ============================================================================

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const tabMap = { report: 'tabReport', metrics: 'tabMetrics', chat: 'tabChat' };
    document.getElementById(tabMap[tab] || 'tabReport').classList.add('active');
}

// ============================================================================
// PDF DOWNLOAD
// ============================================================================

async function downloadPDF() {
    const btn = document.getElementById('btnDownloadPdf');
    const taxId = document.getElementById('taxId').value.trim() || '1234567890';
    const originalText = btn.textContent;

    btn.textContent = '⏳ Generating...';
    btn.disabled = true;
    addLog('📄 Generating ING-branded PDF report...', 'running');

    try {
        const resp = await fetch(`${API_BASE}/api/report/pdf?tax_id=${taxId}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const arrayBuffer = await resp.arrayBuffer();
        // Explicitly create a PDF-typed blob
        const blob = new Blob([arrayBuffer], { type: 'application/pdf' });
        const url = window.URL.createObjectURL(blob);

        // Trigger download via anchor element
        const a = document.createElement('a');
        a.href = url;
        a.download = `financial_report_${taxId}_${new Date().toISOString().slice(0, 10)}.pdf`;
        a.type = 'application/pdf';
        document.body.appendChild(a);
        a.click();

        // Delay cleanup so browser can finish the download
        setTimeout(() => {
            window.URL.revokeObjectURL(url);
            a.remove();
        }, 5000);

        addLog('✅ PDF downloaded successfully', 'success');
    } catch (err) {
        addLog(`❌ PDF failed: ${err.message}`, 'error');
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    addLog('Platform initialized (v4.0 — Excel Upload + Multi-Agent)', 'info');
    addLog('Upload a Mizan Excel file to begin analysis', 'info');
    addLog('Architecture: upload → data_ingestion → quant → verifier → network_mapper → strategist', 'info');

    // ── Resize Handle: drag to adjust center/right pane widths ──
    const handle = document.getElementById('resizeHandle');
    const grid = document.querySelector('.command-center');
    let isResizing = false;

    // Create an overlay to capture mouse events during resize
    // (prevents the Cytoscape canvas from stealing them)
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:9999;cursor:col-resize;display:none;';
    document.body.appendChild(overlay);

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isResizing = true;
        handle.classList.add('active');
        document.body.classList.add('resizing');
        overlay.style.display = 'block';
    });

    overlay.addEventListener('mousemove', onResize);
    document.addEventListener('mousemove', onResize);

    function onResize(e) {
        if (!isResizing) return;
        e.preventDefault();
        const gridRect = grid.getBoundingClientRect();
        const rightPaneWidth = gridRect.right - e.clientX;

        // Clamp: min 250px, max 700px for the right pane
        const clamped = Math.max(250, Math.min(700, rightPaneWidth));
        grid.style.gridTemplateColumns = `300px 1fr 6px ${clamped}px`;

        // Live-resize the Cytoscape graph
        if (cy) cy.resize();
    }

    function stopResize() {
        if (!isResizing) return;
        isResizing = false;
        handle.classList.remove('active');
        document.body.classList.remove('resizing');
        overlay.style.display = 'none';
        // Final refit of the graph
        if (cy) {
            cy.resize();
            cy.fit(undefined, 50);
        }
    }

    document.addEventListener('mouseup', stopResize);
    overlay.addEventListener('mouseup', stopResize);
});

