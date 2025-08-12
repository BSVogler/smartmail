#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from email_indexer import EmailIndexer
import os
import platform
import subprocess
from typing import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import traceback
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global indexer instance
indexer = None

def is_macos():
    """Check if running on macOS"""
    return platform.system() == 'Darwin'

def has_apple_mail():
    """Check if Apple Mail is available by looking for Mail directory"""
    if not is_macos():
        logger.info("Not running on macOS")
        return False
    
    # Check for Mail directory in user's Library
    home_dir = os.path.expanduser('~')
    mail_dir = os.path.join(home_dir, 'Library', 'Mail')
    exists = os.path.exists(mail_dir)
    
    if exists:
        logger.info(f"Apple Mail directory found at: {mail_dir}")
        # Test if we can actually access the directory
        try:
            subdirs = [d for d in os.listdir(mail_dir) if os.path.isdir(os.path.join(mail_dir, d))]
            logger.info(f"Mail subdirectories: {subdirs}")
            return True
        except PermissionError as e:
            logger.error(f"Permission denied accessing Mail directory: {e}")
            logger.error("macOS privacy settings prevent access to Mail data")
            logger.error("The application needs 'Full Disk Access' permission in System Preferences > Privacy & Security")
            return False
        except Exception as e:
            logger.warning(f"Could not list Mail directory contents: {e}")
            return False
    else:
        logger.info(f"Apple Mail directory not found at: {mail_dir}")
        return False

def get_system_info():
    """Get system information for frontend"""
    is_mac = is_macos()
    has_mail = has_apple_mail()
    
    # Detect permission issue
    permission_issue = False
    if is_mac:
        home_dir = os.path.expanduser('~')
        mail_dir = os.path.join(home_dir, 'Library', 'Mail')
        if os.path.exists(mail_dir):
            try:
                os.listdir(mail_dir)
            except PermissionError:
                permission_issue = True
    
    return {
        "is_macos": is_mac,
        "has_apple_mail": has_mail,
        "can_open_local_emails": is_mac and has_mail,
        "permission_issue": permission_issue,
        "platform": platform.system(),
        "mail_access_note": "Requires 'Full Disk Access' permission" if permission_issue else None
    }

def find_email_file_by_id(email_id):
    """Find .emlx file in Apple Mail directories by email ID"""
    if not has_apple_mail():
        return None
    
    home_dir = os.path.expanduser('~')
    mail_dir = os.path.join(home_dir, 'Library', 'Mail')
    
    logger.info(f"Searching for email ID '{email_id}' in {mail_dir}")
    
    # Search through all version directories (V2, V3, V7, V10, etc.)
    files_checked = 0
    for root, dirs, files in os.walk(mail_dir):
        # Skip certain directories that don't contain emails
        if any(skip_dir in root for skip_dir in ['MailData', 'Envelope Index', 'Mailboxes']):
            continue
            
        for file in files:
            if file.endswith('.emlx'):
                file_path = os.path.join(root, file)
                files_checked += 1
                
                try:
                    # Read the .emlx file and check for matching email ID
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(16384)  # Read first 16KB for better matching
                        
                        # Try multiple matching strategies
                        match_patterns = [
                            f'Message-ID: <{email_id}>',
                            f'Message-ID:<{email_id}>',
                            f'Message-Id: <{email_id}>',
                            f'message-id: <{email_id}>',
                            email_id
                        ]
                        
                        for pattern in match_patterns:
                            if pattern in content:
                                logger.info(f"Found email file: {file_path} (matched pattern: {pattern})")
                                return file_path
                            
                except Exception as e:
                    # Skip files that can't be read
                    logger.debug(f"Could not read file {file_path}: {e}")
                    continue
                    
                # Log progress every 100 files
                if files_checked % 100 == 0:
                    logger.info(f"Checked {files_checked} .emlx files so far...")
    
    logger.warning(f"Email file not found for ID: {email_id} after checking {files_checked} files")
    
    # Try alternative approach: search by subject or sender if available
    if indexer and indexer.chunk_metadata:
        for meta in indexer.chunk_metadata:
            if meta['email_id'] == email_id:
                subject = meta.get('subject', '').strip()
                sender = meta.get('from', '').strip()
                logger.info(f"Trying alternative search by subject: '{subject}' and sender: '{sender}'")
                return find_email_file_by_content(subject, sender)
    
    return None

def find_email_file_by_content(subject, sender):
    """Fallback: Find .emlx file by subject and sender"""
    if not has_apple_mail() or not subject:
        return None
    
    home_dir = os.path.expanduser('~')
    mail_dir = os.path.join(home_dir, 'Library', 'Mail')
    
    # Clean subject for matching (remove Re:, Fwd: etc.)
    clean_subject = subject.lower()
    for prefix in ['re:', 'fwd:', 'fw:', 'forward:', 're[']:
        clean_subject = clean_subject.replace(prefix, '').strip()
    
    for root, dirs, files in os.walk(mail_dir):
        if any(skip_dir in root for skip_dir in ['MailData', 'Envelope Index', 'Mailboxes']):
            continue
            
        for file in files:
            if file.endswith('.emlx'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(16384)
                        
                        # Check for subject match
                        if clean_subject in content.lower() and len(clean_subject) > 5:
                            # Also check sender if available
                            if not sender or sender.lower() in content.lower():
                                logger.info(f"Found email file by content: {file_path}")
                                return file_path
                                
                except Exception as e:
                    continue
    
    return None

def open_privacy_settings():
    """Open macOS Privacy & Security settings"""
    try:
        # Open Privacy & Security settings directly
        subprocess.run(['open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles'], check=True)
        return True
    except Exception as e:
        logger.error(f"Failed to open privacy settings: {e}")
        try:
            # Fallback: open general System Preferences
            subprocess.run(['open', '/System/Applications/System Preferences.app'], check=True)
            return True
        except Exception as e2:
            logger.error(f"Failed to open System Preferences: {e2}")
            return False

def open_email_in_apple_mail(email_id):
    """Open email in Apple Mail by finding and opening the .emlx file"""
    logger.info(f"=== Starting Apple Mail integration for email ID: {email_id} ===")
    
    if not has_apple_mail():
        logger.error("Apple Mail not available")
        return {"success": False, "message": "Apple Mail not available"}
    
    logger.info("Apple Mail is available, searching for email file...")
    email_file = find_email_file_by_id(email_id)
    
    if not email_file:
        logger.error(f"Email file not found for ID: {email_id}")
        
        # Add debug info about what email metadata we have
        if indexer and indexer.chunk_metadata:
            matching_metadata = [meta for meta in indexer.chunk_metadata if meta['email_id'] == email_id]
            if matching_metadata:
                meta = matching_metadata[0]
                logger.info(f"Found metadata for email {email_id}:")
                logger.info(f"  Subject: {meta.get('subject')}")
                logger.info(f"  From: {meta.get('from')}")
                logger.info(f"  Date: {meta.get('date')}")
            else:
                logger.error(f"No metadata found for email ID: {email_id}")
                logger.info(f"Available email IDs: {[meta['email_id'] for meta in indexer.chunk_metadata[:10]]}")
        
        return {"success": False, "message": "Email file not found in Apple Mail storage"}
    
    logger.info(f"Found email file: {email_file}")
    
    try:
        # Use subprocess to open the .emlx file (macOS will open it in Apple Mail)
        logger.info(f"Executing: open '{email_file}'")
        result = subprocess.run(['open', email_file], check=True, capture_output=True, text=True)
        logger.info(f"Successfully opened email file in Apple Mail: {email_file}")
        logger.info(f"Subprocess stdout: {result.stdout}")
        logger.info(f"Subprocess stderr: {result.stderr}")
        return {"success": True, "message": "Email opened in Apple Mail"}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to open email file: {e}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return {"success": False, "message": f"Failed to open email: {str(e)}"}

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

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

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global exception handler for better error logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__
        }
    )

@app.get("/api/system-info")
async def get_system_info_endpoint():
    """Get system information and capabilities"""
    return get_system_info()

@app.get("/api/status")
async def get_status():
    """Get application and server status"""
    if indexer is None:
        return {
            "app_initialized": False,
            "infinity_available": False,
            "has_data": False
        }
    
    return {
        "app_initialized": True,
        "infinity_available": indexer.infinity_available,
        "infinity_url": indexer.infinity_url,
        "has_data": bool(indexer.coordinates_2d is not None and len(indexer.email_chunks) > 0),
        "total_chunks": len(indexer.email_chunks) if indexer.email_chunks else 0
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main visualization page"""
    return templates.TemplateResponse("index.html", {"request": request})

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
        
        # Convert any remaining numpy types to native Python types
        data = convert_numpy_types(data)
        return data
        
    except Exception as e:
        logger.error(f"Error in get_emails: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
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
    
    return convert_numpy_types(indexer.clusters)

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
async def reindex_emails(params: dict = Body(...)):
    """Reindex all emails"""
    try:
        logger.info("Starting email reindexing...")
        
        if indexer is None:
            logger.error("Indexer not initialized")
            raise HTTPException(status_code=500, detail="Indexer not initialized")
        
        logger.info("Calling indexer.index_emails()...")
        success = indexer.index_emails()
        
        if success:
            logger.info("Basic indexing successful, now applying clustering parameters...")
            
            # Extract parameters with defaults
            clustering_method = params.get('clustering_method', 'dbscan')
            eps = params.get('eps', 0.25)
            min_samples = params.get('min_samples', 2)
            n_clusters = params.get('n_clusters', None)
            min_dist = params.get('min_dist', 0.0)
            spread = params.get('spread', 0.4)
            
            logger.info(f"Applying clustering: method={clustering_method}, eps={eps}, min_samples={min_samples}, spread={spread}")
            
            # Apply custom clustering and visualization parameters
            indexer.perform_clustering(
                n_clusters=n_clusters,
                clustering_method=clustering_method,
                eps=eps,
                min_samples=min_samples
            )
            indexer.generate_2d_coordinates(
                min_dist=min_dist,
                spread=spread
            )
            
            logger.info("Custom clustering applied, saving data...")
            # Save the data after successful indexing
            indexer.save_data("smartmail_data.pkl")
            logger.info("Data saved successfully")
            return {"success": True, "message": f"Emails reindexed with {clustering_method} clustering"}
        else:
            logger.error("Indexing returned False")
            return {"success": False, "message": "Failed to reindex emails"}
            
    except Exception as e:
        logger.error(f"Exception during reindexing: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@app.post("/api/recalculate-clustering")
async def recalculate_clustering(params: dict = Body(...)):
    """Recalculate clustering and visualization without full reindexing"""
    try:
        logger.info("Starting clustering recalculation...")
        
        if indexer is None:
            logger.error("Indexer not initialized")
            raise HTTPException(status_code=500, detail="Indexer not initialized")
        
        # Check if we have existing email data
        if indexer.email_embeddings is None or indexer.email_metadata is None:
            logger.error("No existing email data for reclustering")
            return {"success": False, "message": "No existing email data available. Please reindex emails first."}
        
        logger.info("Applying new clustering parameters...")
        
        # Extract parameters with defaults
        clustering_method = params.get('clustering_method', 'dbscan')
        eps = params.get('eps', 0.25)
        min_samples = params.get('min_samples', 2)
        n_clusters = params.get('n_clusters', None)
        min_dist = params.get('min_dist', 0.0)
        spread = params.get('spread', 0.4)
        
        logger.info(f"Reclustering: method={clustering_method}, eps={eps}, min_samples={min_samples}, spread={spread}")
        
        # Only run clustering and 2D visualization - skip all the expensive parts
        indexer.perform_clustering(
            n_clusters=n_clusters,
            clustering_method=clustering_method,
            eps=eps,
            min_samples=min_samples
        )
        indexer.generate_2d_coordinates(
            min_dist=min_dist,
            spread=spread
        )
        
        logger.info("Reclustering complete, saving data...")
        # Save the updated data
        indexer.save_data("smartmail_data.pkl")
        logger.info("Data saved successfully")
        
        return {"success": True, "message": f"Clustering recalculated with {clustering_method}"}
        
    except Exception as e:
        logger.error(f"Exception during reclustering: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Reclustering failed: {str(e)}")

@app.post("/api/refresh-mails")
async def refresh_mails():
    """Refresh emails - only update new/removed emails and recalculate clustering"""
    try:
        logger.info("Starting mail refresh...")
        
        if indexer is None:
            logger.error("Indexer not initialized")
            raise HTTPException(status_code=500, detail="Indexer not initialized")
        
        # Check if we have existing data
        if not hasattr(indexer, 'email_chunks') or not indexer.email_chunks:
            logger.info("No existing data, falling back to full reindex")
            success = indexer.index_emails()
            return {"success": success, "added": len(indexer.email_chunks) if success else 0, "removed": 0, "message": "No existing data, performed full reindex"}
        
        logger.info("Refreshing emails (incremental update)...")
        
        # Store original counts
        original_chunk_count = len(indexer.email_chunks)
        original_email_count = len(indexer.email_metadata) if indexer.email_metadata else 0
        
        # Connect to IMAP and fetch current emails
        if not indexer.connect_imap():
            return {"success": False, "message": "Failed to connect to email server"}
        
        current_emails = indexer.fetch_emails(50)
        if not current_emails:
            return {"success": False, "message": "Failed to fetch emails from server"}
        
        # Get current email IDs from server
        current_email_ids = set(email['id'] for email in current_emails)
        
        # Get existing email IDs from our data
        existing_email_ids = set()
        if indexer.chunk_metadata:
            existing_email_ids = set(meta['email_id'] for meta in indexer.chunk_metadata)
        
        # Find new and removed emails
        new_email_ids = current_email_ids - existing_email_ids
        removed_email_ids = existing_email_ids - current_email_ids
        
        # Find emails that may have changed status (read/unread)
        potentially_changed_ids = current_email_ids & existing_email_ids
        
        # Check for status changes in existing emails
        changed_email_ids = set()
        if potentially_changed_ids:
            current_emails_dict = {email['id']: email for email in current_emails}
            existing_unread_status = {}
            
            # Get current unread status from existing data
            for meta in indexer.chunk_metadata:
                if meta['email_id'] not in existing_unread_status:
                    existing_unread_status[meta['email_id']] = meta.get('is_unread', False)
            
            # Compare with server status
            for email_id in potentially_changed_ids:
                current_unread = current_emails_dict[email_id].get('is_unread', False)
                existing_unread = existing_unread_status.get(email_id, False)
                if current_unread != existing_unread:
                    changed_email_ids.add(email_id)
        
        logger.info(f"Found {len(new_email_ids)} new emails, {len(removed_email_ids)} removed emails, {len(changed_email_ids)} changed emails")
        
        # Remove old emails from data structures
        if removed_email_ids:
            # Remove from chunk-level data
            indexer.email_chunks = [chunk for i, chunk in enumerate(indexer.email_chunks) 
                                   if indexer.chunk_metadata[i]['email_id'] not in removed_email_ids]
            indexer.chunk_metadata = [meta for meta in indexer.chunk_metadata 
                                     if meta['email_id'] not in removed_email_ids]
            
            # Remove from embeddings
            if indexer.embeddings_array is not None:
                keep_indices = [i for i, meta in enumerate(indexer.chunk_metadata) 
                               if meta['email_id'] not in removed_email_ids]
                indexer.embeddings_array = indexer.embeddings_array[keep_indices]
        
        # Add new emails
        if new_email_ids:
            new_emails = [email for email in current_emails if email['id'] in new_email_ids]
            new_chunks, new_metadata = indexer.chunk_emails(new_emails)
            
            if new_chunks:
                # Generate embeddings for new chunks
                if indexer.infinity_available:
                    new_embeddings = indexer.get_embeddings_infinity(new_chunks)
                else:
                    new_embeddings = indexer.get_embeddings_local(new_chunks)
                
                # Append to existing data
                indexer.email_chunks.extend(new_chunks)
                indexer.chunk_metadata.extend(new_metadata)
                
                if indexer.embeddings_array is not None and new_embeddings is not None:
                    indexer.embeddings_array = np.vstack([indexer.embeddings_array, new_embeddings])
                else:
                    indexer.embeddings_array = new_embeddings
        
        # Update status for changed emails (no re-processing needed, just metadata update)
        if changed_email_ids:
            current_emails_dict = {email['id']: email for email in current_emails}
            
            # Update chunk metadata
            for i, meta in enumerate(indexer.chunk_metadata):
                if meta['email_id'] in changed_email_ids:
                    indexer.chunk_metadata[i]['is_unread'] = current_emails_dict[meta['email_id']].get('is_unread', False)
            
            logger.info(f"Updated status for {len(changed_email_ids)} emails")
        
        # Rebuild FAISS index with updated data
        if indexer.embeddings_array is not None:
            indexer.build_index(indexer.embeddings_array)
        
        # Recalculate email-level embeddings and clustering
        indexer.aggregate_email_embeddings()
        indexer.perform_clustering()
        indexer.generate_2d_coordinates()
        
        # Save updated data
        indexer.save_data("smartmail_data.pkl")
        
        logger.info("Mail refresh completed successfully")
        return {
            "success": True, 
            "added": len(new_email_ids), 
            "removed": len(removed_email_ids),
            "changed": len(changed_email_ids),
            "message": f"Mail refresh completed: +{len(new_email_ids)} new, -{len(removed_email_ids)} removed, ~{len(changed_email_ids)} updated"
        }
        
    except Exception as e:
        logger.error(f"Exception during mail refresh: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Mail refresh failed: {str(e)}")

@app.get("/api/email/{email_id}")
async def get_email_content(email_id: str):
    """Get full content of a specific email"""
    if indexer is None:
        raise HTTPException(status_code=500, detail="Indexer not initialized")
    
    if not indexer.chunk_metadata:
        raise HTTPException(status_code=404, detail="No email data available")
    
    # Find all chunks for this email
    email_chunks = []
    email_metadata = None
    
    for i, meta in enumerate(indexer.chunk_metadata):
        if meta['email_id'] == email_id:
            email_chunks.append(indexer.email_chunks[i])
            if email_metadata is None:
                email_metadata = meta
    
    if not email_chunks:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Combine all chunks to get full content
    full_content = ' '.join(email_chunks)
    
    return {
        "email_id": email_id,
        "subject": email_metadata.get('subject', 'No Subject'),
        "from": email_metadata.get('from', 'Unknown Sender'),
        "date": email_metadata.get('date', 'No Date'),
        "content": full_content,
        "is_unread": email_metadata.get('is_unread', False),
        "num_chunks": len(email_chunks)
    }

@app.post("/api/open-email-local/{email_id}")
async def open_email_local(email_id: str):
    """Open email in Apple Mail by finding and opening the .emlx file"""
    logger.info(f"=== API REQUEST: open-email-local/{email_id} ===")
    logger.info(f"System platform: {platform.system()}")
    
    if not is_macos():
        logger.error("Request made on non-macOS system")
        raise HTTPException(status_code=400, detail="Local email opening only available on macOS")
    
    if not has_apple_mail():
        logger.error("Apple Mail not found on system")
        raise HTTPException(status_code=400, detail="Apple Mail not found on this system")
    
    logger.info(f"System checks passed, attempting to open email...")
    result = open_email_in_apple_mail(email_id)
    
    logger.info(f"Apple Mail integration result: {result}")
    
    if not result["success"]:
        logger.error(f"Failed to open email: {result['message']}")
        raise HTTPException(status_code=404, detail=result["message"])
    
    logger.info(f"Successfully processed request for email {email_id}")
    return result

@app.post("/api/open-privacy-settings")
async def open_privacy_settings_endpoint():
    """Open macOS Privacy & Security settings to help user grant permissions"""
    if not is_macos():
        raise HTTPException(status_code=400, detail="Privacy settings only available on macOS")
    
    success = open_privacy_settings()
    
    if success:
        return {"success": True, "message": "Privacy settings opened"}
    else:
        return {"success": False, "message": "Failed to open privacy settings"}

if __name__ == "__main__":
    print("Starting SmartMail Visualization Server...")
    print("Open http://localhost:8000 in your browser")
    
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",  # Changed to debug for more verbose logging
        access_log=True,
        use_colors=True
    )