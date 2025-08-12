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
import re
import plistlib
from typing import AsyncGenerator, Optional
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


def apple_mail_dir() -> Optional[str]:
    """Check if Apple Mail is available by looking for Mail directory"""
    if not is_macos():
        logger.info("Not running on macOS")
        return None

    # Check for Mail directory in user's Library
    home_dir: str = os.path.expanduser('~')
    mail_dir = os.path.join(home_dir, 'Library', 'Mail', 'v10')
    exists = os.path.exists(mail_dir)

    if exists:
        logger.info(f"Apple Mail directory found at: {mail_dir}")
        # Test if we can actually access the directory
        try:
            subdirs = [d for d in os.listdir(mail_dir) if os.path.isdir(os.path.join(mail_dir, d))]
            return mail_dir
        except PermissionError as e:
            logger.error(f"Permission denied accessing Mail directory: {e}")
            logger.error("macOS privacy settings prevent access to Mail data")
            logger.error(
                "The application needs 'Full Disk Access' permission in System Preferences > Privacy & Security")
            return None
        except Exception as e:
            logger.warning(f"Could not list Mail directory contents: {e}")
            return None
    else:
        logger.info(f"Apple Mail directory not found at: {mail_dir}")
        return None


def get_system_info():
    """Get system information for frontend"""
    is_mac = is_macos()
    has_mail = apple_mail_dir()

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


def parse_mail_accounts() -> dict:
    """Parse Apple Mail account structure from Accounts.plist"""
    if not apple_mail_dir():
        return {}

    home_dir = os.path.expanduser('~')
    mail_dir = os.path.join(home_dir, 'Library', 'Mail')

    accounts = {}

    # Primary location: V10/MailData/Accounts.plist
    accounts_plist = os.path.join(mail_dir, 'V10', 'MailData', 'Accounts.plist')

    try:
        if os.path.exists(accounts_plist):
            logger.info(f"Reading Apple Mail accounts from: {accounts_plist}")
            with open(accounts_plist, 'rb') as f:
                plist_data = plistlib.load(f)

            logger.debug(f"Accounts.plist structure keys: {list(plist_data.keys())}")

            # Parse account information
            for account_key, account_data in plist_data.items():
                if isinstance(account_data, dict):
                    logger.debug(f"Processing account {account_key}: {list(account_data.keys())}")

                    # Extract email addresses and account info
                    email_addresses = account_data.get('EmailAddresses', [])
                    if not email_addresses and 'EmailAddress' in account_data:
                        email_addresses = [account_data['EmailAddress']]

                    account_name = account_data.get('AccountName', account_data.get('DisplayName', 'Unknown'))
                    account_id = account_data.get('UniqueId', account_key)
                    account_path = account_data.get('Path', '')

                    # Look for account directory in Mail folder
                    if not account_path:
                        # Try to find matching directory by account ID or name
                        for version_dir in ['V10', 'V9', 'V8', 'V7']:
                            version_path = os.path.join(mail_dir, version_dir)
                            if os.path.exists(version_path):
                                for item in os.listdir(version_path):
                                    item_path = os.path.join(version_path, item)
                                    if os.path.isdir(item_path) and (account_id in item or account_key in item):
                                        account_path = item_path
                                        break
                                if account_path:
                                    break

                    for email_addr in email_addresses:
                        if email_addr:  # Skip empty email addresses
                            accounts[email_addr.lower()] = {
                                'account_id': account_id,
                                'account_key': account_key,
                                'path': account_path,
                                'display_name': account_name,
                                'raw_data': account_data  # Keep raw data for debugging
                            }
                            logger.info(f"Found account: {email_addr} -> {account_name} (path: {account_path})")

        else:
            logger.warning(f"Accounts.plist not found at: {accounts_plist}")

    except Exception as e:
        logger.error(f"Could not parse Accounts.plist: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")

    # Fallback: Try other version directories if V10 didn't work
    if not accounts:
        logger.info("Trying fallback: scanning for account directories...")
        for version_dir in ['V9', 'V8', 'V7', 'V6']:
            fallback_accounts_plist = os.path.join(mail_dir, version_dir, 'MailData', 'Accounts.plist')
            if os.path.exists(fallback_accounts_plist):
                logger.info(f"Found fallback Accounts.plist: {fallback_accounts_plist}")
                # Recursively call with different path - simplified for now
                break

    if accounts:
        logger.info(f"Successfully parsed {len(accounts)} Apple Mail accounts: {list(accounts.keys())}")
    else:
        logger.warning("No Apple Mail accounts found in any location")

    return accounts


def find_email_file_by_id(email_id: int, mail_dir: os.PathLike):
    """Find .emlx file in Apple Mail directories by email ID using APPLE_MAILBOX env var"""
    if not apple_mail_dir():
        return None

    logger.info(f"Searching for email ID '{email_id}' in {mail_dir}")

    # Get the configured IMAP email
    imap_email = indexer.email_address if indexer else "unknown"
    logger.info(f"Looking for emails from IMAP account: {imap_email}")

    # Primary: Use APPLE_MAILBOX environment variable
    apple_mailbox = os.getenv('APPLE_MAILBOX')
    target_account_paths = []

    if apple_mailbox:
        logger.info(f"Using APPLE_MAILBOX env var: {apple_mailbox}")
        # Convert to absolute path if it's relative
        if os.path.isabs(apple_mailbox):
            target_account_paths = [apple_mailbox]
        else:
            # Assume it's relative to Mail directory
            target_account_paths = [os.path.join(mail_dir, apple_mailbox)]

        # Verify the path exists
        for path in target_account_paths:
            if os.path.exists(path):
                logger.info(f"Found APPLE_MAILBOX directory: {path}")
            else:
                logger.warning(f"APPLE_MAILBOX directory not found: {path}")
    else:
        # Fallback: Parse Apple Mail account structure
        logger.info("APPLE_MAILBOX not set, falling back to .plist parsing")
        mail_accounts = parse_mail_accounts()
        logger.info(f"Apple Mail accounts: {list(mail_accounts.keys())}")

        if imap_email.lower() in mail_accounts:
            target_account = mail_accounts[imap_email.lower()]
            target_account_paths.append(target_account['path'])
            logger.info(f"Found target account path: {target_account['path']}")
        else:
            logger.warning(f"IMAP account '{imap_email}' not found in Apple Mail accounts")
            logger.info("Searching all accounts...")
            target_account_paths = [acc['path'] for acc in mail_accounts.values() if acc['path']]

    # Search through target account directories first
    files_checked = 0

    # If APPLE_MAILBOX is set, only search in those specific directories
    if apple_mailbox and target_account_paths:
        search_roots = target_account_paths
    else:
        search_roots = [mail_dir]
        logger.info("Searching in all Mail directories (no APPLE_MAILBOX specified)")

    for search_root in search_roots:
        if not os.path.exists(search_root):
            logger.warning(f"Search directory does not exist: {search_root}")
            continue

        for root, dirs, files in os.walk(search_root):
            # Skip certain directories that don't contain emails
            if any(skip_dir in root for skip_dir in ['MailData', 'Envelope Index', 'Mailboxes']):
                continue

            # When using APPLE_MAILBOX, all matches are high priority
            is_target_account = bool(apple_mailbox) or (
                    target_account_paths and any(
                target_path in root for target_path in target_account_paths if target_path)
            )

            for file in files:
                if file.endswith('.emlx'):
                    file_path = os.path.join(root, file)
                    files_checked += 1

                    try:
                        # Read the .emlx file and check for matching email ID
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(16384)  # Read first 16KB for better matching

                            # Try multiple matching strategies
                            #apple mail will match too much when checking just for the email id
                            match_patterns = [
                                f'Message-ID: <{email_id}>',
                                f'Message-ID:<{email_id}>',
                                f'Message-Id: <{email_id}>',
                                f'message-id: <{email_id}>',
                                email_id
                            ]

                            for pattern in match_patterns:
                                if pattern in content:
                                    # Return immediately if from target account
                                    if is_target_account:
                                        # Extract email info for logging HIGH priority matches only
                                        to_line = content.split('To: ')[1].split('\n')[0] if 'To: ' in content else ''
                                        from_line = content.split('From: ')[1].split('\n')[
                                            0] if 'From: ' in content else ''

                                        logger.info(f"Found email file (HIGH priority): {file_path}")
                                        logger.info(f"  Matched pattern: {pattern}")
                                        logger.info(f"  Directory: {root}")
                                        logger.info(f"  From: {from_line[:100]}")
                                        logger.info(f"  To: {to_line[:100]}")
                                        return file_path

                                    # Store as fallback if not from target account (don't log each one)
                                    if not hasattr(find_email_file_by_id, '_fallback_match'):
                                        find_email_file_by_id._fallback_match = file_path
                                        find_email_file_by_id._fallback_count = 1
                                    else:
                                        find_email_file_by_id._fallback_count += 1

                    except Exception as e:
                        # Skip files that can't be read
                        logger.debug(f"Could not read file {file_path}: {e}")
                        continue

                    # Log progress every 100 files
                    if files_checked % 100 == 0:
                        logger.info(f"Checked {files_checked} .emlx files so far...")

    # Check if we have a fallback match from other accounts
    if hasattr(find_email_file_by_id, '_fallback_match'):
        fallback_path = find_email_file_by_id._fallback_match
        fallback_count = getattr(find_email_file_by_id, '_fallback_count', 1)

        # Clean up
        delattr(find_email_file_by_id, '_fallback_match')
        if hasattr(find_email_file_by_id, '_fallback_count'):
            delattr(find_email_file_by_id, '_fallback_count')

        logger.info(f"Using fallback match from different account ({fallback_count} matches found): {fallback_path}")
        return fallback_path

    logger.warning(f"Email file not found for ID: {email_id} after checking {files_checked} files")
    logger.info(f"IMAP account being indexed: {imap_email}")
    logger.info(f"Apple Mail accounts found: {list(mail_accounts.keys())}")

    if imap_email.lower() not in mail_accounts:
        logger.warning(f"ACCOUNT MISMATCH: IMAP account '{imap_email}' not configured in Apple Mail")
        logger.warning("This explains why emails can't be found - different email accounts!")
        logger.info("Possible solutions:")
        logger.info("1. Add the IMAP account to Apple Mail app")
        logger.info("2. Change EMAIL_ADDRESS env var to match one of these Apple Mail accounts:")
        for email_addr, account_info in mail_accounts.items():
            logger.info(f"   - {email_addr} ({account_info['display_name']})")

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
    if not apple_mail_dir() or not subject:
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


def open_email_in_apple_mail(email_id, mail_dir: os.PathLike) -> dict:
    """Open email in Apple Mail by finding and opening the .emlx file"""
    logger.info("Apple Mail is available, searching for email file...")
    email_file = find_email_file_by_id(email_id, mail_dir=mail_dir)

    if not email_file:
        logger.error(f"Email file not found for ID: {email_id}")

        # Add debug info about what email metadata we have
        if indexer and indexer.chunk_metadata:
            matching_metadata = [meta for meta in indexer.chunk_metadata if meta['email_id'] == email_id]
            if matching_metadata:
                meta = matching_metadata[0]
                logger.info(f"Found metadata for email {email_id}:")
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

            logger.info(
                f"Applying clustering: method={clustering_method}, eps={eps}, min_samples={min_samples}, spread={spread}")

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
            return {"success": success, "added": len(indexer.email_chunks) if success else 0, "removed": 0,
                    "message": "No existing data, performed full reindex"}

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

        logger.info(
            f"Found {len(new_email_ids)} new emails, {len(removed_email_ids)} removed emails, {len(changed_email_ids)} changed emails")

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
                    indexer.chunk_metadata[i]['is_unread'] = current_emails_dict[meta['email_id']].get('is_unread',
                                                                                                       False)

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

    if not (dir := apple_mail_dir()):
        logger.error("Apple Mail not found on system")
        raise HTTPException(status_code=400, detail="Apple Mail not found on this system")

    logger.info(f"System checks passed, attempting to open email...")
    result = open_email_in_apple_mail(email_id, mail_dir=dir)

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
