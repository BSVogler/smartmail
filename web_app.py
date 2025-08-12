#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from email_indexer import EmailIndexer
import os
from typing import Dict, List, Optional, AsyncGenerator
import json
from contextlib import asynccontextmanager

# Global indexer instance
indexer = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown"""
    # Startup
    global indexer
    try:
        print("Starting SmartMail application...")
        indexer = EmailIndexer()
        
        # Try to load existing data first
        if os.path.exists("smartmail_data.pkl"):
            if indexer.load_data("smartmail_data.pkl"):
                print("Loaded existing email data")
            else:
                print("Failed to load data, will need to reindex")
        else:
            print("No existing data found, will need to index emails")
            
    except Exception as e:
        print(f"Error initializing indexer: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down SmartMail application...")
    # Add any cleanup code here if needed

app = FastAPI(
    title="SmartMail Visualization", 
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main visualization page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SmartMail - Email Clustering Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
            }
            .main-content {
                display: flex;
                height: 80vh;
            }
            .sidebar {
                width: 300px;
                padding: 20px;
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
                overflow-y: auto;
            }
            .chart-container {
                flex: 1;
                padding: 20px;
                position: relative;
            }
            .controls {
                margin-bottom: 20px;
            }
            .control-group {
                margin-bottom: 15px;
            }
            .control-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #495057;
            }
            .control-group input, .control-group select {
                width: 100%;
                padding: 8px 12px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                width: 100%;
                margin-top: 10px;
            }
            .btn:hover {
                opacity: 0.9;
            }
            .btn:disabled {
                background: #6c757d;
                cursor: not-allowed;
            }
            .stats {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            .stats h3 {
                margin-top: 0;
                color: #495057;
            }
            .stat-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            .clusters-list {
                background: white;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            .cluster-item {
                padding: 12px;
                border-bottom: 1px solid #dee2e6;
                cursor: pointer;
            }
            .cluster-item:hover {
                background-color: #f8f9fa;
            }
            .cluster-item:last-child {
                border-bottom: none;
            }
            .cluster-id {
                font-weight: 600;
                margin-bottom: 5px;
            }
            .cluster-size {
                color: #6c757d;
                font-size: 12px;
                margin-bottom: 8px;
            }
            .cluster-keywords {
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
            }
            .keyword-tag {
                background: #e3f2fd;
                color: #1976d2;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 11px;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #6c757d;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 12px;
                border-radius: 4px;
                margin-bottom: 15px;
            }
            #chart {
                width: 100%;
                height: calc(100% - 60px);
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📧 SmartMail</h1>
                <p>Interactive Email Clustering & Semantic Search Visualization</p>
            </div>
            
            <div class="main-content">
                <div class="sidebar">
                    <div class="controls">
                        <div class="control-group">
                            <label for="search-input">🔍 Semantic Search</label>
                            <input type="text" id="search-input" placeholder="Search emails by content...">
                        </div>
                        
                        <button class="btn" onclick="performSearch()">Search</button>
                        <button class="btn" onclick="loadData()" id="load-btn">🔄 Load Email Data</button>
                        <button class="btn" onclick="reindexEmails()" id="reindex-btn" disabled>📧 Reindex Emails</button>
                    </div>
                    
                    <div class="stats" id="stats-container">
                        <div class="loading">Load data to see statistics</div>
                    </div>
                    
                    <div class="clusters-list" id="clusters-container">
                        <div class="loading">Clusters will appear here</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div id="chart">
                        <div class="loading">
                            <h3>Welcome to SmartMail Visualization</h3>
                            <p>Click "Load Email Data" to begin exploring your email clusters</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentData = null;
            let currentChart = null;

            async function loadData() {
                const loadBtn = document.getElementById('load-btn');
                const reindexBtn = document.getElementById('reindex-btn');
                
                loadBtn.textContent = '⏳ Loading...';
                loadBtn.disabled = true;
                
                try {
                    const response = await fetch('/api/emails');
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        showError(data.error);
                        // Still enable reindex button if there's an error
                        reindexBtn.disabled = false;
                        return;
                    }
                    
                    // Validate data structure
                    if (!data.points || !Array.isArray(data.points)) {
                        showError('Invalid data format received from server');
                        reindexBtn.disabled = false;
                        return;
                    }
                    
                    if (data.points.length === 0) {
                        showError('No email data found. Click "Reindex Emails" to fetch your emails.');
                        reindexBtn.disabled = false;
                        return;
                    }
                    
                    currentData = data;
                    updateStats(data);
                    updateClusters(data);
                    createVisualization(data);
                    
                    reindexBtn.disabled = false;
                    
                } catch (error) {
                    console.error('Load data error:', error);
                    showError('Failed to load email data: ' + error.message);
                    reindexBtn.disabled = false;
                } finally {
                    loadBtn.textContent = '🔄 Reload Data';
                    loadBtn.disabled = false;
                }
            }

            async function reindexEmails() {
                const btn = document.getElementById('reindex-btn');
                btn.textContent = '⏳ Reindexing...';
                btn.disabled = true;
                
                try {
                    const response = await fetch('/api/reindex', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.success) {
                        await loadData();
                    } else {
                        showError('Failed to reindex emails');
                    }
                    
                } catch (error) {
                    showError('Failed to reindex: ' + error.message);
                } finally {
                    btn.textContent = '📧 Reindex Emails';
                    btn.disabled = false;
                }
            }

            async function performSearch() {
                const query = document.getElementById('search-input').value.trim();
                if (!query) return;
                
                try {
                    const response = await fetch(`/api/search/${encodeURIComponent(query)}`);
                    const results = await response.json();
                    
                    if (currentData && results.length > 0) {
                        highlightSearchResults(results);
                    }
                    
                } catch (error) {
                    showError('Search failed: ' + error.message);
                }
            }

            function updateStats(data) {
                const statsContainer = document.getElementById('stats-container');
                statsContainer.innerHTML = `
                    <h3>📊 Statistics</h3>
                    <div class="stat-item">
                        <span>Total Chunks:</span>
                        <strong>${data.total_chunks}</strong>
                    </div>
                    <div class="stat-item">
                        <span>Clusters:</span>
                        <strong>${data.total_clusters}</strong>
                    </div>
                    <div class="stat-item">
                        <span>Unique Emails:</span>
                        <strong>${new Set(data.points.map(p => p.email_id)).size}</strong>
                    </div>
                `;
            }

            function updateClusters(data) {
                const clustersContainer = document.getElementById('clusters-container');
                
                if (!data.clusters || Object.keys(data.clusters).length === 0) {
                    clustersContainer.innerHTML = '<div class="loading">No clusters found</div>';
                    return;
                }
                
                let clustersHtml = '';
                Object.entries(data.clusters).forEach(([clusterId, cluster]) => {
                    const size = cluster.size || 0;
                    const emailCount = cluster.emails ? cluster.emails.length : 0;
                    const keywords = cluster.common_words || [];
                    
                    clustersHtml += `
                        <div class="cluster-item" onclick="highlightCluster(${clusterId})">
                            <div class="cluster-id">Cluster ${clusterId}</div>
                            <div class="cluster-size">${size} chunks • ${emailCount} emails</div>
                            <div class="cluster-keywords">
                                ${keywords.map(word => `<span class="keyword-tag">${word}</span>`).join('')}
                            </div>
                        </div>
                    `;
                });
                
                clustersContainer.innerHTML = clustersHtml;
            }

            function createVisualization(data) {
                if (!data.points || data.points.length === 0) {
                    document.getElementById('chart').innerHTML = '<div class="loading">No data to visualize</div>';
                    return;
                }
                
                const points = data.points;
                
                // Group points by cluster
                const clusterGroups = {};
                points.forEach(point => {
                    const clusterId = point.cluster || 0; // Handle undefined cluster
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
                    name: `Cluster ${clusterId}`,
                    text: clusterPoints.map(p => 
                        `<b>Subject:</b> ${p.subject || 'No subject'}<br>` +
                        `<b>From:</b> ${p.from || 'Unknown sender'}<br>` +
                        `<b>Date:</b> ${p.date || 'No date'}<br>` +
                        `<b>Preview:</b> ${(p.chunk_preview || 'No preview').substring(0, 100)}...`
                    ),
                    hovertemplate: '%{text}<extra></extra>',
                    marker: {
                        size: 8,
                        opacity: 0.7,
                        line: {
                            width: 1,
                            color: 'white'
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
            }

            function highlightCluster(clusterId) {
                if (!currentChart) return;
                
                const update = currentChart.map((trace, index) => ({
                    marker: {
                        ...trace.marker,
                        opacity: index == clusterId ? 1.0 : 0.3,
                        size: index == clusterId ? 10 : 6
                    }
                }));
                
                Plotly.restyle('chart', { marker: update.map(u => u.marker) });
            }

            function highlightSearchResults(results) {
                // This would highlight search results in the visualization
                console.log('Search results:', results);
            }

            function showError(message) {
                const statsContainer = document.getElementById('stats-container');
                statsContainer.innerHTML = `<div class="error">${message}</div>`;
            }

            // Initialize on page load
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('search-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        performSearch();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/emails")
async def get_emails():
    """Get all emails with clustering and 2D coordinates"""
    if indexer is None:
        return JSONResponse(content={"error": "Indexer not initialized"})
    
    if indexer.coordinates_2d is None or indexer.cluster_labels is None or not indexer.email_chunks:
        return JSONResponse(content={
            "error": "No email data available. Please reindex emails first.",
            "points": [],
            "clusters": {},
            "total_chunks": 0,
            "total_clusters": 0
        })
    
    try:
        data = indexer.get_visualization_data()
        if "error" in data:
            return JSONResponse(content=data)
        return data
    except Exception as e:
        return JSONResponse(content={
            "error": f"Failed to get visualization data: {str(e)}",
            "points": [],
            "clusters": {},
            "total_chunks": 0,
            "total_clusters": 0
        })

@app.get("/api/clusters")
async def get_clusters():
    """Get cluster information"""
    if indexer is None or indexer.clusters is None:
        raise HTTPException(status_code=404, detail="No cluster data available")
    
    return indexer.clusters

@app.get("/api/search/{query}")
async def search_emails(query: str):
    """Search emails and return results"""
    if indexer is None:
        raise HTTPException(status_code=500, detail="Indexer not initialized")
    
    if indexer.index is None:
        raise HTTPException(status_code=404, detail="No indexed data available")
    
    results = indexer.search(query, k=10)
    
    return [
        {
            "chunk": result["chunk"],
            "metadata": result["metadata"],
            "similarity": result["similarity"]
        }
        for result in results
    ]

@app.post("/api/reindex")
async def reindex_emails():
    """Reindex all emails"""
    if indexer is None:
        raise HTTPException(status_code=500, detail="Indexer not initialized")
    
    try:
        success = indexer.index_emails()
        if success:
            # Save the data after successful indexing
            indexer.save_data("smartmail_data.pkl")
            return {"success": True, "message": "Emails reindexed successfully"}
        else:
            return {"success": False, "message": "Failed to reindex emails"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

if __name__ == "__main__":
    print("Starting SmartMail Visualization Server...")
    print("Open http://localhost:8000 in your browser")
    
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )