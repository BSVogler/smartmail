#!/usr/bin/env python3

import os
import sys
import imaplib
import email
from email.header import decode_header
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import time
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

load_dotenv()

class EmailIndexer:
    def __init__(self):
        self.imap_server = os.getenv('IMAP_SERVER')
        self.imap_port = int(os.getenv('IMAP_PORT', '993'))
        self.email_address = os.getenv('EMAIL_ADDRESS')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.infinity_url = os.getenv('INFINITY_URL', 'http://localhost:7997')
        self.infinity_model = os.getenv('INFINITY_MODEL', 'BAAI/bge-small-en-v1.5')
        
        if not all([self.imap_server, self.email_address, self.email_password]):
            raise ValueError("Please set IMAP_SERVER, EMAIL_ADDRESS, and EMAIL_PASSWORD environment variables")
        
        self.local_model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        
        print(f"Configuration:")
        print(f"  IMAP Server: {self.imap_server}:{self.imap_port}")
        print(f"  Email: {self.email_address}")
        print(f"  Apple Mailbox: {os.getenv('APPLE_MAILBOX', 'auto-detect')} (for macOS integration)")
        print(f"  Infinity URL: {self.infinity_url} (optional)")
        print(f"  Local fallback model: {self.local_model_name} (loaded only if needed)")
        
        # Don't load local embeddings yet - only if needed
        self.embeddings = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            is_separator_regex=False,
        )
        
        self.email_chunks = []
        self.chunk_metadata = []
        self.index = None
        self.dimension = None
        self.infinity_available = self.check_infinity_server()
        
        # Clustering and visualization
        self.embeddings_array = None  # Chunk-level embeddings
        self.email_embeddings = None  # Email-level embeddings (averaged)
        self.email_metadata = None    # One entry per email
        self.clusters = None
        self.coordinates_2d = None
        self.cluster_labels = None
    
    def check_infinity_server(self) -> bool:
        """Check if Infinity server is available (does not launch server)"""
        try:
            print(f"Checking Infinity server at {self.infinity_url}...")
            response = requests.get(f"{self.infinity_url}/models", timeout=3)
            if response.status_code == 200:
                print(f"✓ Infinity server is online at {self.infinity_url}")
                return True
            else:
                print(f"✗ Infinity server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Infinity server not reachable at {self.infinity_url}")
            return False
        except requests.exceptions.Timeout:
            print(f"✗ Infinity server timeout at {self.infinity_url}")
            return False
        except Exception as e:
            print(f"✗ Error checking Infinity server: {e}")
            return False
        
    def connect_imap(self):
        """Connect to IMAP server and login"""
        print(f"Connecting to {self.imap_server}:{self.imap_port}...")
        try:
            self.mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            self.mail.login(self.email_address, self.email_password)
            print("Successfully connected to IMAP server")
            return True
        except Exception as e:
            print(f"Failed to connect to IMAP server: {e}")
            return False
    
    def fetch_emails(self, num_emails: int = 50) -> List[Dict]:
        """Fetch the last N emails from inbox"""
        emails = []
        
        try:
            self.mail.select('INBOX')
            
            _, search_data = self.mail.search(None, 'ALL')
            email_ids = search_data[0].split()
            
            email_ids = email_ids[-num_emails:] if len(email_ids) > num_emails else email_ids
            
            print(f"Fetching {len(email_ids)} emails...")
            
            for i, email_id in enumerate(email_ids):
                if i % 10 == 0:
                    print(f"Decoding email {i+1}/{len(email_ids)}...")
                
                # Fetch message content without marking as read
                _, msg_data = self.mail.fetch(email_id, '(BODY.PEEK[])')
                
                # Fetch flags separately to be sure we get them
                _, flag_data = self.mail.fetch(email_id, '(FLAGS)')
                
                # Parse message data
                message_data = None
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        message_data = response_part[1]
                        break
                
                # Parse flags data
                flags_str = ""
                for response_part in flag_data:
                    if isinstance(response_part, bytes):
                        flags_str = response_part.decode('utf-8', errors='ignore')
                        break
                
                if message_data:
                    msg = email.message_from_bytes(message_data)
                    
                    subject = self.decode_header_value(msg['Subject'])
                    from_addr = self.decode_header_value(msg['From'])
                    date = msg['Date']
                    
                    body = self.extract_body(msg)
                    
                    # Determine if email is unread (doesn't have SEEN flag)
                    is_unread = True
                    if flags_str and '\\Seen' in flags_str:
                        is_unread = False
                    
                    # Debug logging for flags
                    if i < 3:  # Only log first 3 emails to avoid spam
                        print(f"Email {i+1}: flags = '{flags_str}', is_unread = {is_unread}")
                    
                    email_data = {
                        'id': email_id.decode(),
                        'subject': subject,
                        'from': from_addr,
                        'date': date,
                        'body': body,
                        'full_content': f"From: {from_addr}\nDate: {date}\nSubject: {subject}\n\n{body}",
                        'is_unread': is_unread
                    }
                    
                    emails.append(email_data)
            
            print(f"Successfully fetched {len(emails)} emails")
            return emails
            
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []
    
    def decode_header_value(self, value):
        """Decode email header value"""
        if value is None:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(value):
            if isinstance(part, bytes):
                if encoding:
                    decoded_parts.append(part.decode(encoding, errors='ignore'))
                else:
                    decoded_parts.append(part.decode('utf-8', errors='ignore'))
            else:
                decoded_parts.append(str(part))
        
        return ' '.join(decoded_parts)
    
    def extract_body(self, msg):
        """Extract body from email message"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                
                if content_type == "text/plain":
                    try:
                        body_part = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        body += body_part.decode(charset, errors='ignore')
                    except:
                        pass
                elif content_type == "text/html" and not body:
                    try:
                        body_part = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        html_content = body_part.decode(charset, errors='ignore')
                        import re
                        text = re.sub('<[^<]+?>', '', html_content)
                        body = text
                    except:
                        pass
        else:
            try:
                body_part = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                body = body_part.decode(charset, errors='ignore')
            except:
                body = str(msg.get_payload())
        
        return body.strip()
    
    def chunk_emails(self, emails: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Chunk emails using recursive character text splitting"""
        print("Chunking emails...")
        
        all_chunks = []
        all_metadata = []
        
        for i, email_data in enumerate(emails):
            if i % 10 == 0:
                print(f"Chunking email {i+1}/{len(emails)}...")
            
            content = email_data['full_content']
            
            if len(content.strip()) == 0:
                continue
            
            try:
                chunks = self.text_splitter.split_text(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'email_id': email_data['id'],
                        'subject': email_data['subject'],
                        'from': email_data['from'],
                        'date': email_data['date'],
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'is_unread': email_data.get('is_unread', False)
                    })
            except Exception as e:
                print(f"Error chunking email {email_data['id']}: {e}")
                all_chunks.append(content[:2000])
                all_metadata.append({
                    'email_id': email_data['id'],
                    'subject': email_data['subject'],
                    'from': email_data['from'],
                    'date': email_data['date'],
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'is_unread': email_data.get('is_unread', False)
                })
        
        print(f"Created {len(all_chunks)} chunks from {len(emails)} emails")
        return all_chunks, all_metadata
    
    def get_embeddings_infinity(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Infinity server"""
        if not self.infinity_available:
            print("Infinity server not available, falling back to local embeddings")
            return self.get_embeddings_local(texts)
            
        print(f"✓ Generating embeddings using Infinity server at {self.infinity_url}...")
        
        try:
            url = f"{self.infinity_url}/embeddings"
            
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                if i % 50 == 0:
                    print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
                
                payload = {
                    "model": self.infinity_model,
                    "input": batch
                }
                
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    batch_embeddings = [item['embedding'] for item in result['data']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    print(f"Error from Infinity server: {response.status_code}")
                    print(f"Response: {response.text}")
                    print("Falling back to local embeddings...")
                    return self.get_embeddings_local(texts)
                
                time.sleep(0.1)
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            print(f"Generated {len(embeddings_array)} embeddings with dimension {embeddings_array.shape[1]}")
            return embeddings_array
            
        except Exception as e:
            print(f"Error connecting to Infinity server: {e}")
            print("Falling back to local embeddings...")
            return self.get_embeddings_local(texts)
    
    def get_embeddings_local(self, texts: List[str]) -> np.ndarray:
        """Fallback: Get embeddings using local model"""
        
        # Lazy load the local model only when needed
        if self.embeddings is None:
            print(f"Loading local embedding model: {self.local_model_name}")
            print("Note: This will download the model if not cached locally")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.local_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✓ Local embedding model loaded successfully")
        
        print(f"✓ Generating embeddings using local model: {self.local_model_name}")
        embeddings = []
        
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if i % 100 == 0:
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search"""
        print("Building FAISS index...")
        
        self.dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(self.dimension)
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar email chunks"""
        
        if self.infinity_available:
            query_embedding = self.get_embeddings_infinity([query])
        else:
            query_embedding = self.get_embeddings_local([query])
        
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        seen_emails = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.chunk_metadata[idx]
            
            if metadata['email_id'] not in seen_emails:
                seen_emails.add(metadata['email_id'])
                
                result = {
                    'chunk': self.email_chunks[idx],
                    'metadata': metadata,
                    'similarity': float(distance)
                }
                results.append(result)
        
        return results
    
    def index_emails(self):
        """Main indexing pipeline"""
        print("\n=== Email Indexing Pipeline ===\n")
        
        if not self.connect_imap():
            return False
        
        emails = self.fetch_emails(50)
        if not emails:
            print("No emails fetched")
            return False
        
        self.email_chunks, self.chunk_metadata = self.chunk_emails(emails)
        
        if not self.email_chunks:
            print("No chunks created")
            return False
        
        if self.infinity_available:
            embeddings = self.get_embeddings_infinity(self.email_chunks)
        else:
            embeddings = self.get_embeddings_local(self.email_chunks)
        
        self.embeddings_array = embeddings
        self.build_index(embeddings)
        self.aggregate_email_embeddings()
        self.perform_clustering()
        self.generate_2d_coordinates()
        
        print("\n=== Indexing Complete ===\n")
        print(f"Indexed {len(emails)} emails as {len(self.email_chunks)} searchable chunks")
        
        return True
    
    def interactive_search(self):
        """Interactive search interface"""
        print("\n=== Email Search Interface ===")
        print("Type your search query (or 'quit' to exit)\n")
        
        while True:
            query = input("\nSearch query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = self.search(query, k=5)
            
            if not results:
                print("No results found")
                continue
            
            print(f"\nFound {len(results)} relevant emails:\n")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                chunk = result['chunk']
                similarity = result['similarity']
                
                print(f"\n[{i}] Similarity: {similarity:.3f}")
                print(f"From: {metadata['from']}")
                print(f"Subject: {metadata['subject']}")
                print(f"Date: {metadata['date']}")
                print(f"\nRelevant excerpt:")
                
                excerpt = chunk[:300] + "..." if len(chunk) > 300 else chunk
                print(excerpt.replace('\n', ' '))
                print("-" * 80)
        
        print("\nGoodbye!")
    
    def aggregate_email_embeddings(self):
        """Aggregate chunk embeddings into email-level embeddings by averaging"""
        if self.embeddings_array is None or not self.chunk_metadata:
            print("No embeddings or metadata available for aggregation")
            return
        
        print("Aggregating embeddings by email...")
        
        # Group chunks by email_id
        email_groups = {}
        for i, metadata in enumerate(self.chunk_metadata):
            email_id = metadata['email_id']
            if email_id not in email_groups:
                email_groups[email_id] = {
                    'chunk_indices': [],
                    'metadata': metadata  # Use first chunk's metadata for email
                }
            email_groups[email_id]['chunk_indices'].append(i)
        
        # Create email-level embeddings and metadata
        email_embeddings = []
        email_metadata = []
        
        for email_id, group_data in email_groups.items():
            chunk_indices = group_data['chunk_indices']
            
            # Average the embeddings for all chunks of this email
            email_embedding = np.mean(self.embeddings_array[chunk_indices], axis=0)
            email_embeddings.append(email_embedding)
            
            # Create email-level metadata
            chunks_preview = []
            for idx in chunk_indices:
                chunk_text = self.email_chunks[idx][:100]  # First 100 chars of each chunk
                chunks_preview.append(chunk_text)
            
            combined_preview = " | ".join(chunks_preview)[:500] + "..." if len(" | ".join(chunks_preview)) > 500 else " | ".join(chunks_preview)
            
            email_meta = {
                'email_id': email_id,
                'subject': group_data['metadata']['subject'],
                'from': group_data['metadata']['from'],
                'date': group_data['metadata']['date'],
                'num_chunks': len(chunk_indices),
                'combined_preview': combined_preview,
                'chunk_indices': chunk_indices,  # Keep reference to original chunks
                'is_unread': group_data['metadata'].get('is_unread', False)
            }
            email_metadata.append(email_meta)
        
        self.email_embeddings = np.array(email_embeddings)
        self.email_metadata = email_metadata
        
        print(f"Created {len(email_metadata)} email-level embeddings from {len(self.email_chunks)} chunks")
    
    def perform_clustering(self, n_clusters: Optional[int] = None, clustering_method: str = 'dbscan', 
                         eps: float = 0.25, min_samples: int = 2):
        """Perform clustering on email-level embeddings with tunable parameters"""
        if self.email_embeddings is None:
            print("No email embeddings available for clustering")
            return
        
        print("Performing clustering analysis on emails...")
        
        # Normalize embeddings first for better clustering
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(self.email_embeddings)
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_emails = len(self.email_metadata)
            # Allow more clusters for better granularity
            n_clusters = min(max(3, n_emails // 4), n_emails // 2)  # Up to half the emails as separate clusters
        
        # K-means clustering on normalized embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
        kmeans_labels = kmeans.fit_predict(normalized_embeddings)
        
        # DBSCAN clustering with tighter parameters for more cohesive clusters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        dbscan_labels = dbscan.fit_predict(self.email_embeddings)  # Use original embeddings for cosine
        
        # Count DBSCAN clusters (excluding noise cluster -1)
        dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        print(f"DBSCAN with eps={eps}, min_samples={min_samples} produced {dbscan_n_clusters} clusters")
        
        # If DBSCAN produces too few clusters, try progressively looser parameters
        original_eps = eps
        attempts = 0
        max_attempts = 3
        
        while dbscan_n_clusters < 2 and attempts < max_attempts:
            attempts += 1
            new_eps = eps * (1.5 ** attempts)  # Exponentially increase eps
            new_min_samples = max(1, min_samples - attempts)  # Decrease min_samples
            
            print(f"DBSCAN attempt {attempts}: trying eps={new_eps:.3f}, min_samples={new_min_samples}")
            dbscan_retry = DBSCAN(eps=new_eps, min_samples=new_min_samples, metric='cosine')
            dbscan_labels = dbscan_retry.fit_predict(self.email_embeddings)
            dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            print(f"DBSCAN attempt {attempts} result: {dbscan_n_clusters} clusters")
        
        # Store clustering results
        self.clusters = {
            'kmeans': {
                'labels': kmeans_labels,
                'n_clusters': n_clusters,
                'centers': kmeans.cluster_centers_
            },
            'dbscan': {
                'labels': dbscan_labels,
                'n_clusters': dbscan_n_clusters,
                'eps': eps,
                'min_samples': min_samples
            }
        }
        
        # Choose clustering method based on user preference and results
        if clustering_method == 'dbscan':
            if dbscan_n_clusters >= 1:  # Accept even 1 cluster if user wants DBSCAN
                self.cluster_labels = dbscan_labels
                print(f"✓ Using DBSCAN: {dbscan_n_clusters} clusters + noise points")
            else:
                print(f"⚠ DBSCAN failed completely, falling back to K-means")
                self.cluster_labels = kmeans_labels
                print(f"✓ Using K-means fallback: {n_clusters} clusters")
        else:  # kmeans selected
            self.cluster_labels = kmeans_labels
            print(f"✓ Using K-means: {n_clusters} clusters")
        
        # Show final cluster distribution
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        cluster_distribution = dict(zip(unique_labels, counts))
        print(f"Final cluster distribution: {cluster_distribution}")
    
    def generate_2d_coordinates(self, n_neighbors: int = None, min_dist: float = 0.0, 
                               spread: float = 0.5):
        """Generate 2D coordinates using UMAP for visualization with much tighter clustering"""
        if self.email_embeddings is None:
            print("No email embeddings available for 2D projection")
            return
        
        print("Generating 2D coordinates using UMAP on email embeddings...")
        
        # Use UMAP for dimensionality reduction with very tight visualization parameters
        if n_neighbors is None:
            n_neighbors = min(5, len(self.email_embeddings) - 1)  # Very few neighbors for tight clusters
        
        reducer = umap.UMAP(
            n_neighbors=max(2, n_neighbors),
            min_dist=min_dist,      # 0.0 = points can be touching
            spread=spread,          # 0.5 = much tighter clusters in 2D
            n_components=2,
            random_state=42,
            metric='cosine',
            n_epochs=1000,          # More epochs for better cluster separation
            learning_rate=0.5,      # Slower learning for stable clusters
            negative_sample_rate=10, # More negative sampling for better separation
            repulsion_strength=2.0, # Stronger repulsion between different clusters
            a=None,                 # Let UMAP calculate optimal curve parameters
            b=None
        )
        
        self.coordinates_2d = reducer.fit_transform(self.email_embeddings)
        
        print(f"2D coordinates generated successfully for emails (n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread})")
        
        # Post-process to enhance cluster separation
        self._enhance_cluster_separation()
    
    def _enhance_cluster_separation(self):
        """Post-process 2D coordinates to enhance visual cluster separation"""
        if self.coordinates_2d is None or self.cluster_labels is None:
            return
        
        print("Enhancing cluster separation in 2D visualization...")
        
        # Calculate cluster centers in 2D space
        cluster_centers = {}
        for label in np.unique(self.cluster_labels):
            if label == -1:  # Skip noise points (DBSCAN outliers)
                continue
            cluster_mask = self.cluster_labels == label
            cluster_coords = self.coordinates_2d[cluster_mask]
            cluster_centers[label] = np.mean(cluster_coords, axis=0)
        
        # Move clusters apart while keeping internal structure
        if len(cluster_centers) > 1:
            # Calculate pairwise distances between cluster centers
            center_positions = np.array(list(cluster_centers.values()))
            
            # Find minimum distance between cluster centers
            min_distance = float('inf')
            for i in range(len(center_positions)):
                for j in range(i + 1, len(center_positions)):
                    dist = np.linalg.norm(center_positions[i] - center_positions[j])
                    min_distance = min(min_distance, dist)
            
            # If clusters are too close, spread them out more
            if min_distance < 3.0:  # Threshold for minimum cluster separation
                expansion_factor = 3.0 / min_distance
                
                # Expand from the overall center
                overall_center = np.mean(self.coordinates_2d, axis=0)
                
                for label in cluster_centers.keys():
                    cluster_mask = self.cluster_labels == label
                    
                    # Move cluster center away from overall center
                    center_offset = cluster_centers[label] - overall_center
                    new_center = overall_center + center_offset * expansion_factor
                    
                    # Apply the same transformation to all points in the cluster
                    cluster_coords = self.coordinates_2d[cluster_mask]
                    relative_positions = cluster_coords - cluster_centers[label]
                    self.coordinates_2d[cluster_mask] = new_center + relative_positions
                
                print(f"Enhanced cluster separation with expansion factor {expansion_factor:.2f}")
    
    def get_visualization_data(self) -> Dict:
        """Get data formatted for visualization - one point per email"""
        if not all([self.coordinates_2d is not None, self.cluster_labels is not None, self.email_metadata is not None]):
            return {"error": "Email clustering and 2D coordinates not available"}
        
        data_points = []
        
        for i, (coord, cluster, email_meta) in enumerate(zip(
            self.coordinates_2d, 
            self.cluster_labels, 
            self.email_metadata
        )):
            data_points.append({
                'id': int(i),
                'x': float(coord[0]) if coord[0] is not None else 0.0,
                'y': float(coord[1]) if coord[1] is not None else 0.0,
                'cluster': int(cluster) if cluster is not None else 0,
                'subject': str(email_meta['subject']) if email_meta['subject'] else '',
                'from': str(email_meta['from']) if email_meta['from'] else '',
                'date': str(email_meta['date']) if email_meta['date'] else '',
                'email_preview': str(email_meta['combined_preview']),
                'email_id': str(email_meta['email_id']),
                'num_chunks': int(email_meta['num_chunks']),
                'is_unread': bool(email_meta.get('is_unread', False))
            })
        
        # Calculate cluster statistics
        cluster_stats = {}
        # Convert numpy types to native Python types
        unique_clusters = [int(c) for c in set(self.cluster_labels)]
        
        for cluster_id in unique_clusters:
            cluster_points = [p for p in data_points if p['cluster'] == cluster_id]
            # Extract keywords from email previews
            email_previews = [p['email_preview'] for p in cluster_points]
            cluster_stats[int(cluster_id)] = {
                'size': int(len(cluster_points)),
                'emails': [str(p['email_id']) for p in cluster_points],
                'common_words': self._extract_common_words(email_previews),
                'subjects': [p['subject'] for p in cluster_points[:3]]  # Show first 3 subjects
            }
        
        return {
            'points': data_points,
            'clusters': cluster_stats,
            'total_emails': int(len(data_points)),
            'total_chunks': int(len(self.email_chunks)) if self.email_chunks else 0,
            'total_clusters': int(len(unique_clusters))
        }
    
    def _extract_common_words(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extract most common words from a list of texts"""
        from collections import Counter
        import re
        
        # Simple word extraction and counting
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend(words)
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 3]
        
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(top_n)]
    
    def save_data(self, filepath: str):
        """Save indexed data to file"""
        data = {
            'email_chunks': self.email_chunks,
            'chunk_metadata': self.chunk_metadata,
            'embeddings_array': self.embeddings_array,
            'email_embeddings': self.email_embeddings,
            'email_metadata': self.email_metadata,
            'clusters': self.clusters,
            'coordinates_2d': self.coordinates_2d,
            'cluster_labels': self.cluster_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str):
        """Load indexed data from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.email_chunks = data['email_chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.embeddings_array = data['embeddings_array']
            
            # Load email-level data (backward compatibility)
            self.email_embeddings = data.get('email_embeddings', None)
            self.email_metadata = data.get('email_metadata', None)
            
            self.clusters = data['clusters']
            self.coordinates_2d = data['coordinates_2d']
            self.cluster_labels = data['cluster_labels']
            
            # Rebuild FAISS index for chunk-level search
            if self.embeddings_array is not None:
                self.build_index(self.embeddings_array)
            
            # If email-level data is missing, regenerate it
            if self.email_embeddings is None and self.embeddings_array is not None:
                print("Regenerating email-level embeddings from loaded data...")
                self.aggregate_email_embeddings()
            
            print(f"Data loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False


def main():
    try:
        indexer = EmailIndexer()
        
        if indexer.index_emails():
            indexer.interactive_search()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()