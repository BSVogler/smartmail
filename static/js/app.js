// Global variables
let currentData = null;
let currentChart = null;
let statusCheckInterval = null;
let selectedClusterId = null;
let systemInfo = null;

// Check system information
async function checkSystemInfo() {
    try {
        const response = await fetch('/api/system-info');
        systemInfo = await response.json();
        console.log('System info:', systemInfo);
        updateSystemDisplay();
    } catch (error) {
        console.error('System info check failed:', error);
        systemInfo = {
            is_macos: false,
            has_apple_mail: false,
            can_open_local_emails: false,
            platform: 'unknown'
        };
    }
}

// Check application status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        updateStatusDisplay(status);
    } catch (error) {
        console.error('Status check failed:', error);
        updateStatusDisplay({
            app_initialized: false,
            infinity_available: false,
            has_data: false
        });
    }
}

// Update status display
function updateStatusDisplay(status) {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = statusIndicator.querySelector('.status-text');
    
    // Remove existing status classes
    statusIndicator.classList.remove('connected', 'partial', 'offline');
    
    if (status.app_initialized) {
        if (status.infinity_available && status.has_data) {
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Connected';
        } else if (status.infinity_available || status.has_data) {
            statusIndicator.classList.add('partial');
            statusText.textContent = 'Partial';
        } else {
            statusIndicator.classList.add('offline');
            statusText.textContent = 'Offline';
        }
    } else {
        statusIndicator.classList.add('offline');
        statusText.textContent = 'Starting...';
    }
    
    // Store current status for detailed view
    window.currentStatus = status;
    
    // Add simple tooltip for quick reference
    statusIndicator.title = getDetailedStatusTooltip(status);
}

// Generate detailed status tooltip
function getDetailedStatusTooltip(status) {
    if (!status.app_initialized) {
        return 'Application is starting up...';
    }
    
    const parts = [];
    parts.push(`App: ${status.app_initialized ? 'Online' : 'Offline'}`);
    parts.push(`Infinity: ${status.infinity_available ? 'Online' : 'Offline'}`);
    parts.push(`Data: ${status.has_data ? `${status.total_chunks} chunks` : 'No data'}`);
    
    return parts.join('\n');
}

// Update system display with macOS/Apple Mail info
function updateSystemDisplay() {
    if (!systemInfo) return;
    
    const statusText = document.querySelector('.status-text');
    const statusIndicator = document.getElementById('status-indicator');
    
    // Update tooltip to include system info
    let systemTooltip = '';
    if (systemInfo.is_macos) {
        systemTooltip += `Platform: macOS\n`;
        if (systemInfo.has_apple_mail) {
            systemTooltip += `Apple Mail: Available\n`;
            systemTooltip += `Local emails: ${systemInfo.can_open_local_emails ? 'Supported' : 'Not supported'}`;
        } else if (systemInfo.permission_issue) {
            systemTooltip += `Apple Mail: Permission Required\n`;
            systemTooltip += `${systemInfo.mail_access_note}`;
        } else {
            systemTooltip += `Apple Mail: Not found`;
        }
    } else {
        systemTooltip += `Platform: ${systemInfo.platform}\nApple Mail: Not available`;
    }
    
    // Add system info to existing tooltip
    const existingTooltip = statusIndicator.title || '';
    statusIndicator.title = existingTooltip + (existingTooltip ? '\n\n' : '') + systemTooltip;
}

// Toggle detailed status view
function toggleStatusDetails() {
    const statusDetails = document.getElementById('status-details');
    const isVisible = statusDetails.style.display !== 'none';
    
    if (isVisible) {
        statusDetails.style.display = 'none';
    } else {
        // Update detailed status content
        updateStatusDetails();
        statusDetails.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (statusDetails.style.display === 'block') {
                statusDetails.style.display = 'none';
            }
        }, 5000);
    }
}

// Update detailed status content
function updateStatusDetails() {
    const statusDetails = document.getElementById('status-details');
    const status = window.currentStatus || {};
    
    if (!status.app_initialized) {
        statusDetails.innerHTML = '<div style="text-align: center; color: #6c757d;">Application starting...</div>';
        return;
    }
    
    const detailsHtml = `
        <div class="status-item">
            <span>🚀 Application:</span>
            <span class="status-badge ${status.app_initialized ? 'online' : 'offline'}">
                ${status.app_initialized ? 'online' : 'offline'}
            </span>
        </div>
        <div class="status-item">
            <span>⚡ Infinity Server:</span>
            <span class="status-badge ${status.infinity_available ? 'online' : 'offline'}">
                ${status.infinity_available ? 'online' : 'offline'}
            </span>
        </div>
        <div class="status-item">
            <span>📊 Email Data:</span>
            <span class="status-badge ${status.has_data ? 'online' : 'offline'}">
                ${status.has_data ? `${status.total_chunks} chunks` : 'no data'}
            </span>
        </div>
    `;
    
    statusDetails.innerHTML = detailsHtml;
}

// Main data loading function
async function loadData() {
    const refreshMailsBtn = document.getElementById('refresh-mails-btn');
    const reindexBtn = document.getElementById('reindex-btn');
    
    try {
        const response = await fetch('/api/emails');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            // Still enable both buttons if there's an error
            refreshMailsBtn.disabled = false;
            reindexBtn.disabled = false;
            return;
        }
        
        // Validate data structure
        if (!data.points || !Array.isArray(data.points)) {
            showError('Invalid data format received from server');
            refreshMailsBtn.disabled = false;
            reindexBtn.disabled = false;
            return;
        }
        
        if (data.points.length === 0) {
            showError('No email data found. Click "Reindex All" to fetch your emails.');
            refreshMailsBtn.disabled = false;
            reindexBtn.disabled = false;
            return;
        }
        
        currentData = data;
        updateClusters(data);
        createVisualization(data);
        updateEmailList(data);
        
        refreshMailsBtn.disabled = false;
        reindexBtn.disabled = false;
        
    } catch (error) {
        console.error('Load data error:', error);
        showError('Failed to load email data: ' + error.message);
        refreshMailsBtn.disabled = false;
        reindexBtn.disabled = false;
    } finally {
        // No load button to re-enable
    }
}

// Reindex emails function
async function reindexEmails() {
    const btn = document.getElementById('reindex-btn');
    btn.textContent = '⏳ Reindexing...';
    btn.disabled = true;
    
    // Show progress bar
    showProgress('🔄 Initializing reindexing...', 5);
    
    // Get clustering parameters from UI
    const params = {
        clustering_method: document.getElementById('clustering-method').value,
        eps: parseFloat(document.getElementById('eps-param').value),
        min_samples: parseInt(document.getElementById('min-samples').value),
        spread: parseFloat(document.getElementById('spread-param').value)
    };
    
    try {
        // Simulate progress through different stages
        updateProgress('📧 Connecting to email server...', 10);
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress('📥 Fetching emails...', 20);
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress('✂️ Processing and chunking...', 40);
        
        const response = await fetch('/api/reindex', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });
        
        updateProgress('🧠 Generating embeddings...', 60);
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        updateProgress('🔗 Clustering emails...', 75);
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress('📊 Creating visualization...', 85);
        
        const result = await response.json();
        
        if (result.success) {
            updateProgress('💾 Saving data...', 95);
            await new Promise(resolve => setTimeout(resolve, 300));
            
            updateProgress('✅ Loading visualization...', 100);
            await loadData();
            await checkStatus(); // Refresh status after successful reindex
            
            // Keep progress at 100% for a moment before hiding
            await new Promise(resolve => setTimeout(resolve, 800));
            hideProgress();
        } else {
            hideProgress();
            showError('Failed to reindex emails: ' + (result.message || 'Unknown error'));
        }
        
    } catch (error) {
        hideProgress();
        showError('Failed to reindex: ' + error.message);
    } finally {
        btn.textContent = '📧 Reindex All';
        btn.disabled = false;
    }
}

// Refresh mails function - only update new/removed emails
async function refreshMails() {
    const btn = document.getElementById('refresh-mails-btn');
    btn.textContent = '⏳ Refreshing...';
    btn.disabled = true;
    
    // Show progress bar for mail refresh
    showProgress('📧 Checking for new emails...', 10);
    
    try {
        // Simulate progress through different stages
        updateProgress('🔍 Comparing with server...', 20);
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress('📥 Fetching new emails...', 40);
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress('✂️ Processing new content...', 60);
        
        const response = await fetch('/api/refresh-mails', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        updateProgress('🧠 Updating embeddings...', 75);
        await new Promise(resolve => setTimeout(resolve, 800));
        
        updateProgress('🔗 Recalculating clusters...', 90);
        
        const result = await response.json();
        
        if (result.success) {
            updateProgress('✅ Loading updated data...', 100);
            await loadData();
            await checkStatus(); // Refresh status after successful refresh
            
            // Keep progress at 100% for a moment before hiding
            await new Promise(resolve => setTimeout(resolve, 800));
            hideProgress();
            
            // Show summary of changes
            if (result.added > 0 || result.removed > 0 || result.changed > 0) {
                console.log(`Mail refresh completed: +${result.added} new emails, -${result.removed} removed emails, ~${result.changed || 0} updated emails`);
            }
        } else {
            hideProgress();
            showError('Failed to refresh mails: ' + (result.message || 'Unknown error'));
        }
        
    } catch (error) {
        hideProgress();
        showError('Failed to refresh mails: ' + error.message);
    } finally {
        btn.textContent = '📨 Refresh Mails';
        btn.disabled = false;
    }
}

// Perform semantic search
async function performSearch(query) {
    if (!query || query.trim().length === 0) {
        clearSearchResults();
        return;
    }
    
    const trimmedQuery = query.trim();
    if (trimmedQuery.length < 2) {
        clearSearchResults();
        return;
    }
    
    try {
        const response = await fetch(`/api/search/${encodeURIComponent(trimmedQuery)}`);
        const results = await response.json();
        
        if (currentData && results.length > 0) {
            highlightSearchResults(results);
            displaySearchResultsInSeparateSection(results);
        } else {
            clearSearchResults();
        }
        
    } catch (error) {
        console.error('Search failed:', error);
        clearSearchResults();
    }
}


// Update clusters display
function updateClusters(data) {
    const clustersContainer = document.getElementById('clusters-container');
    
    if (!data.clusters || Object.keys(data.clusters).length === 0) {
        clustersContainer.innerHTML = '<div class="loading">No clusters found</div>';
        return;
    }
    
    // Calculate totals for header
    const totalEmails = data.total_emails || data.points.length;
    const totalClusters = data.total_clusters;
    
    let clustersHtml = `
        <div style="padding: 12px; background: #f8f9fa; border-bottom: 1px solid #e9ecef; font-size: 12px; color: #6c757d; display: flex; justify-content: space-between; align-items: center;">
            <span>📧 ${totalEmails} emails in ${totalClusters} clusters</span>
            <button class="settings-btn" onclick="toggleClusteringParams()" title="Clustering Settings">
                ⚙️
            </button>
        </div>
    `;
    
    Object.entries(data.clusters).forEach(([clusterId, cluster]) => {
        const size = cluster.size || 0;
        const emailCount = cluster.emails ? cluster.emails.length : 0;
        const keywords = cluster.common_words || [];
        
        const subjects = cluster.subjects || [];
        const subjectPreview = subjects.length > 0 ? subjects.slice(0, 2).join(', ') : '';
        
        const clusterDisplayName = clusterId === '-1' ? 'Miscellaneous' : `Cluster ${clusterId}`;
        
        // Count unread emails in this cluster
        const clusterEmails = currentData.points.filter(p => p.cluster == clusterId);
        const unreadCount = clusterEmails.filter(p => p.is_unread).length;
        const unreadDisplay = unreadCount > 0 ? ` (${unreadCount} unread)` : '';
        
        clustersHtml += `
            <div class="cluster-item" data-cluster-id="${clusterId}" onclick="highlightCluster(${clusterId})">
                <div class="cluster-id">${clusterDisplayName}</div>
                <div class="cluster-size">${emailCount} emails${unreadDisplay}</div>
                ${subjectPreview ? `<div class="cluster-subjects" style="font-size: 11px; color: #6c757d; margin: 4px 0;">${subjectPreview}</div>` : ''}
                <div class="cluster-keywords">
                    ${keywords.map(word => `<span class="keyword-tag">${word}</span>`).join('')}
                </div>
            </div>
        `;
    });
    
    clustersContainer.innerHTML = clustersHtml;
    
    // Update cluster item selection after rendering
    updateClusterItemSelection();
}

// Update visual selection state of cluster items
function updateClusterItemSelection() {
    const clusterItems = document.querySelectorAll('.cluster-item');
    
    clusterItems.forEach(item => {
        const clusterId = parseInt(item.dataset.clusterId);
        
        if (selectedClusterId === clusterId) {
            item.style.backgroundColor = '#e8f4f8';
            item.style.borderLeft = '3px solid #0066cc';
        } else {
            item.style.backgroundColor = '';
            item.style.borderLeft = '';
        }
    });
}

// Create the main visualization
function createVisualization(data) {
    if (!data.points || data.points.length === 0) {
        document.getElementById('chart').innerHTML = '<div class="loading">No data to visualize</div>';
        return;
    }
    
    const points = data.points;
    
    // Group points by cluster (preserve -1 for noise points)
    const clusterGroups = {};
    points.forEach(point => {
        const clusterId = point.cluster; // Don't convert -1 to 0
        if (!clusterGroups[clusterId]) {
            clusterGroups[clusterId] = [];
        }
        clusterGroups[clusterId].push(point);
    });

    // Create traces for each cluster
    const traces = Object.entries(clusterGroups).map(([clusterId, clusterPoints]) => ({
        x: clusterPoints.map(p => p.x || 0),
        y: clusterPoints.map(p => p.y || 0),
        mode: 'markers',
        type: 'scatter',
        name: clusterId === '-1' ? 'Miscellaneous' : `Cluster ${clusterId}`,
        text: clusterPoints.map(p => 
            `<b>Subject:</b> ${p.subject || 'No subject'}${p.is_unread ? ' 📬' : ''}<br>` +
            `<b>From:</b> ${p.from || 'Unknown sender'}<br>` +
            `<b>Date:</b> ${p.date || 'No date'}<br>` +
            `<b>Status:</b> ${p.is_unread ? 'Unread' : 'Read'}<br>` +
            `<b>Chunks:</b> ${p.num_chunks || 1}<br>` +
            `<b>Preview:</b> ${(p.email_preview || p.chunk_preview || 'No preview').substring(0, 150)}...`
        ),
        hovertemplate: '%{text}<extra></extra>',
        marker: {
            size: clusterPoints.map(p => p.is_unread ? 10 : 8),
            opacity: clusterPoints.map(p => p.is_unread ? 1.0 : 0.7),
            line: {
                width: clusterPoints.map(p => p.is_unread ? 2 : 1),
                color: clusterPoints.map(p => p.is_unread ? '#ff6b35' : 'white')
            }
        }
    }));

    const layout = {
        title: 'Email Clusters in 2D Space (UMAP)',
        xaxis: { title: 'UMAP Dimension 1' },
        yaxis: { title: 'UMAP Dimension 2' },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1,
            xanchor: 'left',
            y: 1
        },
        margin: { t: 40, r: 150, b: 40, l: 40 }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d']
    };

    Plotly.newPlot('chart', traces, layout, config);
    currentChart = traces;
    
    // Store cluster ID to trace index mapping for highlighting
    window.clusterToTraceMap = {};
    Object.keys(clusterGroups).forEach((clusterId, index) => {
        window.clusterToTraceMap[clusterId] = index;
    });
}

// Highlight a specific cluster with persistent selection
function highlightCluster(clusterId) {
    if (!currentChart || !window.clusterToTraceMap) return;
    
    // Convert cluster ID to string for consistent comparison
    const clusterIdStr = String(clusterId);
    
    // Toggle selection if clicking the same cluster
    if (selectedClusterId === clusterId) {
        selectedClusterId = null;
        // Reset to default view
        const resetUpdate = currentChart.map((trace, index) => ({
            marker: {
                ...trace.marker,
                opacity: 0.7,
                size: 8
            }
        }));
        Plotly.restyle('chart', { marker: resetUpdate.map(u => u.marker) });
        updateClusterItemSelection();
        return;
    }
    
    // Select new cluster
    selectedClusterId = clusterId;
    
    // Get the correct trace index for this cluster ID
    const targetTraceIndex = window.clusterToTraceMap[clusterIdStr];
    
    const update = currentChart.map((trace, index) => ({
        marker: {
            ...trace.marker,
            opacity: index === targetTraceIndex ? 1.0 : 0.3,
            size: index === targetTraceIndex ? 10 : 6
        }
    }));
    
    Plotly.restyle('chart', { marker: update.map(u => u.marker) });
    updateClusterItemSelection();
    
    // Update email list to show only selected cluster
    updateEmailList(currentData, selectedClusterId);
}

// Highlight search results in the visualization
function highlightSearchResults(results) {
    if (!currentChart || !currentData || results.length === 0) {
        console.log('No chart, data, or results to highlight');
        return;
    }
    
    console.log('Search results:', results);
    
    // Clear any cluster selection
    selectedClusterId = null;
    updateClusterItemSelection();
    
    // Extract email IDs from search results
    const matchingEmailIds = new Set();
    results.forEach(result => {
        if (result.metadata && result.metadata.email_id) {
            matchingEmailIds.add(result.metadata.email_id);
        }
    });
    
    console.log('Matching email IDs:', matchingEmailIds);
    
    if (matchingEmailIds.size === 0) {
        console.log('No matching email IDs found');
        return;
    }
    
    // Find corresponding points in the visualization data
    const matchingPointIndices = new Set();
    currentData.points.forEach((point, index) => {
        if (matchingEmailIds.has(point.email_id)) {
            // Find which cluster (trace) this point belongs to
            const clusterId = point.cluster;
            matchingPointIndices.add({ pointIndex: index, clusterId: clusterId });
        }
    });
    
    console.log('Matching point indices:', matchingPointIndices);
    
    // Create individual point styling updates
    const update = currentChart.map((trace, traceIndex) => {
        // Find the cluster ID for this trace index
        const clusterId = Object.keys(window.clusterToTraceMap || {}).find(
            id => window.clusterToTraceMap[id] === traceIndex
        );
        
        // Get all points in this trace from currentData
        const traceDataPoints = currentData.points.filter(point => String(point.cluster) === String(clusterId));
        
        // Create arrays for individual point styling
        const opacities = [];
        const sizes = [];
        const lineWidths = [];
        const lineColors = [];
        
        traceDataPoints.forEach((dataPoint) => {
            if (matchingEmailIds.has(dataPoint.email_id)) {
                // This point matches search results - highlight it
                opacities.push(1.0);
                sizes.push(12);
                lineWidths.push(2);
                lineColors.push('#ff6b35');
            } else {
                // This point doesn't match - fade it
                opacities.push(0.2);
                sizes.push(6);
                lineWidths.push(1);
                lineColors.push('white');
            }
        });
        
        return {
            marker: {
                ...trace.marker,
                opacity: opacities,
                size: sizes,
                line: {
                    width: lineWidths,
                    color: lineColors
                }
            }
        };
    });
    
    Plotly.restyle('chart', { marker: update.map(u => u.marker) });
    
    // Show search results in sidebar
    displaySearchResultsInSidebar(results, matchingEmailIds);
}

// Display search results in separate section (keeping this for backward compatibility but not using)
function displaySearchResultsInSidebar(results, matchingEmailIds) {
    // This function is now unused - keeping for backward compatibility
    // Results are displayed in displaySearchResultsInSeparateSection instead
}

// Display search results in the dedicated search results section
function displaySearchResultsInSeparateSection(results) {
    const searchContainer = document.getElementById('search-results-container');
    const searchResultsList = document.getElementById('search-results-list');
    
    // Show the search results container
    searchContainer.style.display = 'block';
    
    // Sort results by similarity (highest first)
    const sortedResults = [...results].sort((a, b) => b.similarity - a.similarity);
    
    // Count unique emails
    const matchingEmailIds = new Set();
    sortedResults.forEach(result => {
        if (result.metadata && result.metadata.email_id) {
            matchingEmailIds.add(result.metadata.email_id);
        }
    });
    
    // Update the header text
    const header = searchContainer.querySelector('h4');
    header.textContent = `🔍 Found ${matchingEmailIds.size} matching emails`;
    
    let searchHtml = '';
    
    sortedResults.forEach((result, index) => {
        const excerpt = result.chunk.length > 150 ? result.chunk.substring(0, 150) + '...' : result.chunk;
        
        searchHtml += `
            <div class="search-result-item">
                <div class="cluster-size">From: ${result.metadata.from}</div>
                <div class="cluster-subjects" style="font-size: 11px; color: #2c3e50; margin: 4px 0; font-weight: 600;">
                    ${result.metadata.subject}
                </div>
                <div style="font-size: 11px; color: #6c757d; margin: 4px 0; line-height: 1.3;">
                    ${excerpt}
                </div>
            </div>
        `;
    });
    
    searchResultsList.innerHTML = searchHtml;
}

// Clear search results and restore normal view
function clearSearchResults() {
    // Hide search results container
    const searchContainer = document.getElementById('search-results-container');
    searchContainer.style.display = 'none';
    
    if (!currentChart || !currentData) return;
    
    // Reset visualization to normal state
    const resetUpdate = currentChart.map((trace, index) => ({
        marker: {
            ...trace.marker,
            opacity: 0.7,
            size: 8,
            line: {
                width: 1,
                color: 'white'
            }
        }
    }));
    
    Plotly.restyle('chart', { marker: resetUpdate.map(u => u.marker) });
    
    // Clear search input
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.value = '';
    }
}

// Show error message
function showError(message) {
    const clustersContainer = document.getElementById('clusters-container');
    clustersContainer.innerHTML = `<div style="padding: 12px;"><div class="error">${message}</div></div>`;
}

// Progress bar functions
function showProgress(text = "Processing...", percent = 0) {
    const container = document.getElementById('progress-container');
    const textElement = document.getElementById('progress-text');
    const percentElement = document.getElementById('progress-percent');
    const fillElement = document.getElementById('progress-fill');
    
    container.style.display = 'block';
    textElement.textContent = text;
    percentElement.textContent = `${percent}%`;
    fillElement.style.width = `${percent}%`;
}

function updateProgress(text, percent) {
    const textElement = document.getElementById('progress-text');
    const percentElement = document.getElementById('progress-percent');
    const fillElement = document.getElementById('progress-fill');
    
    textElement.textContent = text;
    percentElement.textContent = `${percent}%`;
    fillElement.style.width = `${percent}%`;
}

function hideProgress() {
    const container = document.getElementById('progress-container');
    container.style.display = 'none';
}

// Setup parameter controls
function setupParameterControls() {
    // Update parameter value displays
    const epsSlider = document.getElementById('eps-param');
    const epsValue = document.getElementById('eps-value');
    const minSamplesSlider = document.getElementById('min-samples');
    const minSamplesValue = document.getElementById('min-samples-value');
    const spreadSlider = document.getElementById('spread-param');
    const spreadValue = document.getElementById('spread-value');
    const methodSelect = document.getElementById('clustering-method');
    
    // Load saved parameters
    loadParameterValues();
    
    let parameterChangeTimeout = null;
    
    function onParameterChange() {
        // Save current parameter values
        saveParameterValues();
        
        // Clear any existing timeout
        if (parameterChangeTimeout) {
            clearTimeout(parameterChangeTimeout);
        }
        
        // Set a new timeout to avoid too frequent updates
        parameterChangeTimeout = setTimeout(() => {
            recalculateClustering();
        }, 800); // Wait 800ms after user stops changing parameters
    }
    
    epsSlider.addEventListener('input', function() {
        epsValue.textContent = this.value;
        onParameterChange();
    });
    
    minSamplesSlider.addEventListener('input', function() {
        minSamplesValue.textContent = this.value;
        onParameterChange();
    });
    
    spreadSlider.addEventListener('input', function() {
        spreadValue.textContent = this.value;
        onParameterChange();
    });
    
    methodSelect.addEventListener('change', function() {
        onParameterChange();
    });
}

// Save parameter values to localStorage
function saveParameterValues() {
    const params = {
        clustering_method: document.getElementById('clustering-method').value,
        eps: document.getElementById('eps-param').value,
        min_samples: document.getElementById('min-samples').value,
        spread: document.getElementById('spread-param').value
    };
    
    localStorage.setItem('smartmail_clustering_params', JSON.stringify(params));
}

// Load parameter values from localStorage
function loadParameterValues() {
    try {
        const savedParams = localStorage.getItem('smartmail_clustering_params');
        if (!savedParams) return;
        
        const params = JSON.parse(savedParams);
        
        // Apply saved values to controls
        const methodSelect = document.getElementById('clustering-method');
        const epsSlider = document.getElementById('eps-param');
        const epsValue = document.getElementById('eps-value');
        const minSamplesSlider = document.getElementById('min-samples');
        const minSamplesValue = document.getElementById('min-samples-value');
        const spreadSlider = document.getElementById('spread-param');
        const spreadValue = document.getElementById('spread-value');
        
        if (params.clustering_method) {
            methodSelect.value = params.clustering_method;
        }
        if (params.eps) {
            epsSlider.value = params.eps;
            epsValue.textContent = params.eps;
        }
        if (params.min_samples) {
            minSamplesSlider.value = params.min_samples;
            minSamplesValue.textContent = params.min_samples;
        }
        if (params.spread) {
            spreadSlider.value = params.spread;
            spreadValue.textContent = params.spread;
        }
        
    } catch (error) {
        console.log('No saved parameters found or error loading them:', error);
    }
}

// Recalculate clustering with new parameters (without full reindexing)
async function recalculateClustering() {
    // Only proceed if we have existing data
    if (!currentData || !currentData.points || currentData.points.length === 0) {
        console.log('No existing data available for reclustering');
        return;
    }
    
    // Get current parameters
    const params = {
        clustering_method: document.getElementById('clustering-method').value,
        eps: parseFloat(document.getElementById('eps-param').value),
        min_samples: parseInt(document.getElementById('min-samples').value),
        spread: parseFloat(document.getElementById('spread-param').value),
        recalculate_only: true  // Flag to indicate we only want clustering recalculation
    };
    
    try {
        console.log('Recalculating clustering with parameters:', params);
        
        // Show spinner in clustering parameters box
        showClusteringSpinner();
        
        const response = await fetch('/api/recalculate-clustering', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Reload the visualization data
            await loadData();
            
            // Hide spinner after a moment
            setTimeout(() => {
                hideClusteringSpinner();
            }, 300);
        } else {
            hideClusteringSpinner();
            console.error('Failed to recalculate clustering:', result.message);
            // Don't show error to user for parameter changes, just log it
        }
        
    } catch (error) {
        hideClusteringSpinner();
        console.error('Error recalculating clustering:', error);
        // Don't show error to user for parameter changes, just log it
    }
}

// Show clustering spinner
function showClusteringSpinner() {
    const spinner = document.getElementById('clustering-spinner');
    if (spinner) {
        spinner.style.display = 'flex';
    }
}

// Hide clustering spinner
function hideClusteringSpinner() {
    const spinner = document.getElementById('clustering-spinner');
    if (spinner) {
        spinner.style.display = 'none';
    }
}

// Toggle clustering parameters section
function toggleClusteringParams() {
    const paramsSection = document.getElementById('clustering-params');
    if (paramsSection) {
        const isVisible = paramsSection.style.display !== 'none';
        paramsSection.style.display = isVisible ? 'none' : 'block';
    }
}

// Update email list display
function updateEmailList(data, filteredClusterId = null) {
    const emailList = document.getElementById('email-list');
    const clearFilterBtn = document.getElementById('clear-filter-btn');
    
    if (!data.points || data.points.length === 0) {
        emailList.innerHTML = '<div class="loading">No emails to display</div>';
        return;
    }
    
    // Filter emails by cluster if specified
    let emailsToShow = data.points;
    if (filteredClusterId !== null) {
        emailsToShow = data.points.filter(point => point.cluster == filteredClusterId);
        clearFilterBtn.style.display = 'block';
    } else {
        clearFilterBtn.style.display = 'none';
    }
    
    // Sort emails by date (newest first) and unread status
    emailsToShow.sort((a, b) => {
        // Unread emails first
        if (a.is_unread !== b.is_unread) {
            return b.is_unread - a.is_unread;
        }
        // Then by date (newest first)
        return new Date(b.date) - new Date(a.date);
    });
    
    let emailsHtml = '';
    emailsToShow.forEach(email => {
        const unreadClass = email.is_unread ? 'unread' : '';
        const unreadIndicator = email.is_unread ? '📬 ' : '';
        const preview = email.email_preview || 'No preview available';
        const shortPreview = preview.length > 120 ? preview.substring(0, 120) + '...' : preview;
        
        // Add Apple Mail button if available
        const appleMailButton = (systemInfo && systemInfo.can_open_local_emails) 
            ? `<button class="apple-mail-btn" onclick="event.stopPropagation(); openEmailInAppleMail('${email.email_id}')" title="Open in Apple Mail">📧</button>`
            : '';
        
        emailsHtml += `
            <div class="email-item ${unreadClass}" data-email-id="${email.email_id}" onclick="selectEmail('${email.email_id}')">
                <div class="email-item-header">
                    <div class="email-subject">${unreadIndicator}${email.subject || 'No Subject'}</div>
                    ${appleMailButton}
                </div>
                <div class="email-from">From: ${email.from || 'Unknown Sender'}</div>
                <div class="email-date">${formatEmailDate(email.date)}</div>
                <div class="email-preview">${shortPreview}</div>
            </div>
        `;
    });
    
    emailList.innerHTML = emailsHtml;
    
    // Add double-click handlers to the new email items
    addEmailDoubleClickHandler();
}

// Format email date for display
function formatEmailDate(dateString) {
    if (!dateString) return 'No Date';
    
    try {
        const date = new Date(dateString);
        const now = new Date();
        const diffHours = (now - date) / (1000 * 60 * 60);
        
        if (diffHours < 24) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (diffHours < 24 * 7) {
            return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
        } else {
            return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
        }
    } catch (e) {
        return dateString.substring(0, 16); // Fallback to raw string
    }
}

// Handle email selection
function selectEmail(emailId) {
    console.log('Selected email:', emailId);
    // TODO: Could implement email detail view or actions here
}

// Clear email filter
function clearEmailFilter() {
    updateEmailList(currentData, null);
    
    // Also clear cluster selection
    selectedClusterId = null;
    updateClusterItemSelection();
    
    // Reset visualization highlighting
    if (currentChart && currentData) {
        const resetUpdate = currentChart.map((trace, index) => ({
            marker: {
                ...trace.marker,
                opacity: 0.7,
                size: 8,
                line: {
                    width: 1,
                    color: 'white'
                }
            }
        }));
        Plotly.restyle('chart', { marker: resetUpdate.map(u => u.marker) });
    }
}

// Splitter functionality
function initializeSplitter() {
    const splitter = document.getElementById('splitter');
    const sidebar = document.getElementById('sidebar');
    const chartContainer = document.getElementById('chart-container');
    let isDragging = false;
    let startX = 0;
    let startWidth = 0;
    
    splitter.addEventListener('mousedown', function(e) {
        isDragging = true;
        startX = e.clientX;
        startWidth = parseInt(document.defaultView.getComputedStyle(sidebar).width, 10);
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', function(e) {
        if (!isDragging) return;
        
        const dx = e.clientX - startX;
        const newWidth = startWidth + dx;
        
        // Enforce min/max width constraints
        const minWidth = 250;
        const maxWidth = Math.min(600, window.innerWidth * 0.7); // Max 70% of window width
        const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
        
        sidebar.style.width = constrainedWidth + 'px';
        
        // Save the width preference
        localStorage.setItem('smartmail_sidebar_width', constrainedWidth);
        
        e.preventDefault();
    });
    
    document.addEventListener('mouseup', function() {
        if (isDragging) {
            isDragging = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            
            // Trigger chart resize after splitter drag
            if (window.Plotly && document.getElementById('chart')) {
                setTimeout(() => {
                    window.Plotly.Plots.resize('chart');
                }, 100);
            }
        }
    });
    
    // Load saved width preference
    const savedWidth = localStorage.getItem('smartmail_sidebar_width');
    if (savedWidth) {
        const width = parseInt(savedWidth, 10);
        if (width >= 250 && width <= 600) {
            sidebar.style.width = width + 'px';
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    let searchTimeout = null;
    
    // Initialize splitter functionality
    initializeSplitter();
    
    // Add auto-search with debouncing
    document.getElementById('search-input').addEventListener('input', function(e) {
        const query = e.target.value;
        
        // Clear any existing timeout
        if (searchTimeout) {
            clearTimeout(searchTimeout);
        }
        
        // Set a new timeout to perform search after user stops typing
        searchTimeout = setTimeout(() => {
            performSearch(query);
        }, 500); // Wait 500ms after user stops typing
    });
    
    // Add enter key support for immediate search
    document.getElementById('search-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            if (searchTimeout) {
                clearTimeout(searchTimeout);
            }
            performSearch(e.target.value);
        }
    });
    
    // Initial checks
    checkSystemInfo();
    checkStatus();
    
    // Auto-load data when page opens
    loadData();
    
    // Set up parameter controls
    setupParameterControls();
    
    // Set up periodic status checks (every 30 seconds)
    statusCheckInterval = setInterval(checkStatus, 30000);
});

// Email viewer functionality
async function openEmailViewer(emailData) {
    const viewer = document.getElementById('email-viewer');
    const subject = document.getElementById('email-viewer-subject');
    const from = document.getElementById('email-viewer-from');
    const date = document.getElementById('email-viewer-date');
    const content = document.getElementById('email-viewer-content');
    
    // Show the viewer first with loading state
    viewer.style.display = 'flex';
    content.innerHTML = '<div class="loading">Loading email content...</div>';
    
    // Adjust chart container height
    const chartContainer = document.getElementById('chart-container');
    chartContainer.style.flex = '1';
    
    try {
        // Fetch full email content from server
        const response = await fetch(`/api/email/${emailData.email_id}`);
        
        if (!response.ok) {
            throw new Error('Failed to load email content');
        }
        
        const fullEmailData = await response.json();
        
        // Populate email data
        subject.textContent = fullEmailData.subject || 'No Subject';
        from.textContent = fullEmailData.from || 'Unknown Sender';
        date.textContent = fullEmailData.date || 'No Date';
        
        // Format and display full content
        let emailContent = fullEmailData.content || 'No content available';
        
        // Try to format the content nicely
        emailContent = emailContent
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
        
        content.innerHTML = emailContent;
        
    } catch (error) {
        console.error('Error loading email content:', error);
        
        // Fallback to preview data
        subject.textContent = emailData.subject || 'No Subject';
        from.textContent = emailData.from || 'Unknown Sender';
        date.textContent = emailData.date || 'No Date';
        
        let emailContent = emailData.email_preview || 'No content available';
        emailContent = emailContent
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
        
        content.innerHTML = `<div class="error">Failed to load full content. Showing preview:</div>${emailContent}`;
    }
}

function closeEmailViewer() {
    const viewer = document.getElementById('email-viewer');
    viewer.style.display = 'none';
    
    // Restore chart container to full height
    const chartContainer = document.getElementById('chart-container');
    chartContainer.style.flex = '2';
}

// Open email in Apple Mail (macOS only)
async function openEmailInAppleMail(emailId) {
    if (!systemInfo) {
        alert('System information not available');
        return;
    }
    
    if (!systemInfo.is_macos) {
        alert('Apple Mail integration only available on macOS');
        return;
    }
    
    if (systemInfo.permission_issue) {
        if (confirm(`Apple Mail integration requires permission.\n\nTo enable:\n1. Open System Preferences → Privacy & Security\n2. Go to 'Full Disk Access'\n3. Add your terminal/IDE to the list\n4. Restart the SmartMail server\n\nWould you like to open Privacy Settings now?`)) {
            openPrivacySettings();
        }
        return;
    }
    
    if (!systemInfo.can_open_local_emails) {
        alert('Apple Mail integration not available on this system');
        return;
    }
    
    try {
        const response = await fetch(`/api/open-email-local/${emailId}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to open email');
        }
        
        console.log('Email opened in Apple Mail:', result.message);
        
    } catch (error) {
        console.error('Error opening email in Apple Mail:', error);
        
        if (error.message.includes('Permission') || error.message.includes('Operation not permitted')) {
            if (confirm(`Permission Error: SmartMail needs access to Apple Mail data.\n\nTo fix this:\n1. Open System Preferences → Privacy & Security\n2. Go to 'Full Disk Access'\n3. Add your terminal/IDE to the list\n4. Restart SmartMail\n\nWould you like to open Privacy Settings now?\n\nError: ${error.message}`)) {
                openPrivacySettings();
            }
        } else {
            alert(`Failed to open email in Apple Mail: ${error.message}`);
        }
    }
}

// Open macOS Privacy Settings
async function openPrivacySettings() {
    try {
        const response = await fetch('/api/open-privacy-settings', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            console.log('Privacy settings opened successfully');
        } else {
            console.error('Failed to open privacy settings:', result.message);
            alert('Could not open Privacy Settings automatically. Please open System Preferences manually.');
        }
        
    } catch (error) {
        console.error('Error opening privacy settings:', error);
        alert('Could not open Privacy Settings automatically. Please open System Preferences manually.');
    }
}

// Add double-click handler to email items
function addEmailDoubleClickHandler() {
    const emailItems = document.querySelectorAll('.email-item');
    emailItems.forEach(item => {
        item.addEventListener('dblclick', function(e) {
            e.preventDefault();
            
            // Get email data from the item
            const emailId = this.dataset.emailId;
            if (!currentData || !currentData.points) return;
            
            // Find the email data in current data
            const emailData = currentData.points.find(point => point.email_id === emailId);
            if (emailData) {
                openEmailViewer(emailData);
            }
        });
    });
}