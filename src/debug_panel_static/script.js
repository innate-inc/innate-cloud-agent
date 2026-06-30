// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Innate Inc

/**
 * Brain Debug Panel - Client-side JavaScript
 * Real-time state monitoring for Brain instances
 */

let currentConnectionId = null;
let debugData = {};

// ============================================================================
// Utility Functions
// ============================================================================

function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString();
}

function formatRelativeTime(isoString) {
    const now = new Date();
    const then = new Date(isoString);
    const diff = Math.floor((now - then) / 1000);
    
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
}

function truncate(str, maxLen = 200) {
    if (!str) return '';
    if (str.length <= maxLen) return str;
    return str.substring(0, maxLen) + '...';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatJson(obj) {
    const str = JSON.stringify(obj, null, 2);
    return str
        .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
        .replace(/: "([^"]+)"/g, ': <span class="json-string">"$1"</span>')
        .replace(/: (\d+)/g, ': <span class="json-number">$1</span>')
        .replace(/: (true|false)/g, ': <span class="json-boolean">$1</span>')
        .replace(/: null/g, ': <span class="json-null">null</span>');
}

// ============================================================================
// Rendering Functions
// ============================================================================

function renderConnectionTabs() {
    const tabsContainer = document.getElementById('connectionTabs');
    const connectionIds = Object.keys(debugData);
    
    if (connectionIds.length === 0) {
        tabsContainer.style.display = 'none';
        return;
    }
    
    tabsContainer.style.display = 'flex';
    tabsContainer.innerHTML = connectionIds.map(cid => `
        <button class="connection-tab ${cid === currentConnectionId ? 'active' : ''}" 
                onclick="selectConnection('${cid}')">
            ${escapeHtml(cid)}
        </button>
    `).join('');
    
    // Auto-select first connection if none selected
    if (!currentConnectionId && connectionIds.length > 0) {
        currentConnectionId = connectionIds[0];
    }
}

function selectConnection(connectionId) {
    currentConnectionId = connectionId;
    renderConnectionTabs();
    renderMainContent();
}

function renderMainContent() {
    const mainContent = document.getElementById('mainContent');
    const connectionIds = Object.keys(debugData);
    
    if (connectionIds.length === 0) {
        mainContent.innerHTML = renderNoConnections();
        return;
    }
    
    const state = debugData[currentConnectionId];
    if (!state) return;
    
    if (state.error) {
        mainContent.innerHTML = renderError(state.error);
        return;
    }

    mainContent.innerHTML = `
        ${renderSidebarLeft(state)}
        ${renderContentMain(state)}
        ${renderSidebarRight(state)}
    `;
}

function renderNoConnections() {
    return `
        <div class="no-connections">
            <div class="no-connections-icon">🔌</div>
            <div>No active Brain connections</div>
            <div style="font-size: 0.85rem; margin-top: 8px; color: var(--text-dim);">
                Connect a client to the WebSocket server to see debug information
            </div>
        </div>
    `;
}

function renderError(error) {
    return `
        <div class="no-connections">
            <div class="no-connections-icon">⚠️</div>
            <div>Error loading state</div>
            <div style="font-size: 0.85rem; margin-top: 8px; color: var(--accent-rose);">
                ${escapeHtml(error)}
            </div>
        </div>
    `;
}

function renderSidebarLeft(state) {
    return `
        <div class="sidebar-left">
            ${renderStatusCard(state)}
            ${renderDirectiveCard(state)}
            ${renderUserMessageCard(state)}
            ${renderPrimitivesCard(state)}
        </div>
    `;
}

function renderStatusCard(state) {
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">⚡</span>
                    Status
                </span>
            </div>
            <div class="card-body">
                <div class="status-row" style="margin-bottom: 12px;">
                    <div class="status-indicator ${state.running ? 'active' : 'idle'}"></div>
                    <span class="status-text">${state.running ? 'Running' : 'Stopped'}</span>
                </div>
                <div class="state-grid">
                    <div class="state-item">
                        <span class="state-label">Connection ID</span>
                        <span class="state-value">${escapeHtml(state.connection_id || 'N/A')}</span>
                    </div>
                    <div class="state-item">
                        <span class="state-label">Model</span>
                        <span class="state-value highlight-teal">${escapeHtml(state.gemini_variant || 'N/A')}</span>
                    </div>
                    <div class="state-item">
                        <span class="state-label">Message Queue</span>
                        <span class="state-value">${state.message_queue_size || 0} pending</span>
                    </div>
                    <div class="state-item">
                        <span class="state-label">History Entries</span>
                        <span class="state-value">${state.history_entry_count || 0}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderDirectiveCard(state) {
    const content = state.directive 
        ? `<div class="directive-box">${escapeHtml(state.directive)}</div>`
        : `<div class="empty-state"><div class="empty-icon">📝</div><div class="empty-text">No directive set</div></div>`;
    
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">📋</span>
                    Directive
                </span>
            </div>
            <div class="card-body">${content}</div>
        </div>
    `;
}

function renderUserMessageCard(state) {
    const content = state.latest_user_message 
        ? `<div class="state-value highlight-amber">${escapeHtml(state.latest_user_message)}</div>`
        : `<div class="empty-state"><div class="empty-icon">💭</div><div class="empty-text">No pending message</div></div>`;
    
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">💬</span>
                    Latest User Message
                </span>
            </div>
            <div class="card-body">${content}</div>
        </div>
    `;
}

function renderPrimitivesCard(state) {
    const localPrims = state.local_primitives?.map(name => 
        `<span class="primitive-tag local">⚙️ ${escapeHtml(name)}</span>`
    ).join('') || '';
    
    const registeredPrims = state.registered_primitives?.map(p => 
        `<span class="primitive-tag">📦 ${escapeHtml(p.name)}</span>`
    ).join('') || '';
    
    const hasAny = state.local_primitives?.length || state.registered_primitives?.length;
    const content = hasAny 
        ? localPrims + registeredPrims
        : '<div class="empty-state"><div class="empty-text">No primitives</div></div>';
    
    const count = (state.registered_primitives?.length || 0) + (state.local_primitives?.length || 0);
    
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">🔧</span>
                    Primitives
                </span>
                <span class="card-badge">${count}</span>
            </div>
            <div class="card-body compact">
                <div class="primitives-scroll">${content}</div>
            </div>
        </div>
    `;
}

function renderContentMain(state) {
    return `
        <div class="content-main">
            ${renderCurrentPrimitiveCard(state)}
            ${renderHistoryCard(state)}
        </div>
    `;
}

function renderCurrentPrimitiveCard(state) {
    const prim = state.primitive_in_execution;
    const badge = prim 
        ? `<span class="card-badge" style="background: var(--accent-emerald-dim); color: var(--accent-emerald);">ACTIVE</span>`
        : `<span class="card-badge">IDLE</span>`;
    
    let content;
    if (prim) {
        const guidelines = prim.guidelines 
            ? `<div style="margin-top: 10px; font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(truncate(prim.guidelines, 150))}</div>`
            : '';
        const primId = prim.primitive_id 
            ? `<span class="primitive-id">${escapeHtml(prim.primitive_id)}</span>` 
            : '';
        
        content = `
            <div class="primitive-card">
                <div class="primitive-name">
                    ${escapeHtml(prim.name)}
                    ${primId}
                </div>
                <div class="primitive-inputs">${formatJson(prim.inputs || {})}</div>
                ${guidelines}
            </div>
        `;
    } else {
        content = `
            <div class="empty-state">
                <div class="empty-icon">⏸️</div>
                <div class="empty-text">No primitive currently executing</div>
            </div>
        `;
    }
    
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">▶️</span>
                    Primitive In Execution
                </span>
                ${badge}
            </div>
            <div class="card-body">${content}</div>
        </div>
    `;
}

function renderHistoryCard(state) {
    const entries = state.history_entries?.slice().reverse().map(entry => `
        <div class="history-item type-${entry.type}">
            <div class="history-meta">
                <span class="history-type">${escapeHtml(entry.type)}</span>
                <span class="history-time">${formatRelativeTime(entry.timestamp)}</span>
            </div>
            <div class="history-desc">${escapeHtml(truncate(entry.description, 300))}</div>
        </div>
    `).join('') || `
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-text">No history entries</div>
        </div>
    `;
    
    return `
        <div class="card" style="flex: 1;">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">📜</span>
                    History
                </span>
                <span class="card-badge">${state.history_entries?.length || 0} entries</span>
            </div>
            <div class="card-body" style="padding: 12px;">
                <div class="history-list">${entries}</div>
            </div>
        </div>
    `;
}

function renderSidebarRight(state) {
    return `
        <div class="sidebar-right">
            ${renderPrimitiveIdsCard(state)}
            ${renderDiscrepanciesCard(state)}
            ${renderSystemInfoCard(state)}
        </div>
    `;
}

function renderPrimitiveIdsCard(state) {
    const idsMap = state.primitive_ids_map || {};
    const count = Object.keys(idsMap).length;
    
    let content;
    if (count > 0) {
        content = Object.entries(idsMap).map(([id, prim]) => `
            <div class="primitive-card" style="margin-bottom: 8px;">
                <div class="primitive-name" style="font-size: 0.75rem;">
                    ${escapeHtml(prim.name)}
                </div>
                <div class="primitive-id" style="margin-bottom: 6px;">${escapeHtml(id)}</div>
                <div class="primitive-inputs" style="max-height: 60px;">${formatJson(prim.inputs || {})}</div>
            </div>
        `).join('');
    } else {
        content = `
            <div class="empty-state">
                <div class="empty-icon">🔗</div>
                <div class="empty-text">No primitive IDs tracked</div>
            </div>
        `;
    }
    
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">🗺️</span>
                    Recent Primitive IDs
                </span>
                <span class="card-badge">${count}</span>
            </div>
            <div class="card-body compact">${content}</div>
        </div>
    `;
}

function renderDiscrepanciesCard(state) {
    const discrepancies = state.discrepancies || [];
    const count = discrepancies.length;
    const badgeStyle = count > 0 ? 'style="background: var(--accent-rose-dim); color: var(--accent-rose);"' : '';
    
    let content;
    if (count > 0) {
        content = discrepancies.slice().reverse().map(d => `
            <div class="discrepancy-item">
                <div class="discrepancy-time">${formatRelativeTime(d.timestamp)}</div>
                <div class="discrepancy-msg">${escapeHtml(d.message)}</div>
            </div>
        `).join('');
    } else {
        content = `
            <div class="empty-state">
                <div class="empty-icon">✅</div>
                <div class="empty-text">No discrepancies</div>
            </div>
        `;
    }
    
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">⚠️</span>
                    Discrepancies
                </span>
                <span class="card-badge" ${badgeStyle}>${count}</span>
            </div>
            <div class="card-body compact" style="max-height: 250px; overflow-y: auto;">
                ${content}
            </div>
        </div>
    `;
}

function renderSystemInfoCard(state) {
    return `
        <div class="card">
            <div class="card-header">
                <span class="card-title">
                    <span class="card-title-icon">ℹ️</span>
                    System Info
                </span>
            </div>
            <div class="card-body">
                <div class="state-grid">
                    <div class="state-item">
                        <span class="state-label">Memory Commands</span>
                        <span class="state-value ${state.enable_memory_commands ? 'highlight-emerald' : ''}">
                            ${state.enable_memory_commands ? 'Enabled' : 'Disabled'}
                        </span>
                    </div>
                    <div class="state-item">
                        <span class="state-label">Summarizing</span>
                        <span class="state-value ${state.is_summarizing ? 'highlight-amber' : ''}">
                            ${state.is_summarizing ? 'In Progress' : 'Idle'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// Data Fetching
// ============================================================================

async function fetchDebugData() {
    try {
        const response = await fetch('/api/debug');
        const data = await response.json();
        debugData = data;
        
        renderConnectionTabs();
        renderMainContent();
        
        document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
    } catch (error) {
        console.error('Failed to fetch debug data:', error);
    }
}

// ============================================================================
// Initialization
// ============================================================================

// Initial fetch and setup auto-refresh
fetchDebugData();
setInterval(fetchDebugData, 1000);

