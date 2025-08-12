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
import umap

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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.local_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
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
        self.embeddings_array = None
        self.clusters = None
        self.coordinates_2d = None
        self.cluster_labels = None
    
    def check_infinity_server(self) -> bool:
        """Check if Infinity server is available"""
        try:
            print(f"Checking Infinity server at {self.infinity_url}...")
            response = requests.get(f"{self.infinity_url}/models", timeout=2)
            if response.status_code == 200:
                print(f"✓ Infinity server is online at {self.infinity_url}")
                return True
        except:
            pass
        print(f"✗ Infinity server not available, will use local embeddings")
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
                
                _, msg_data = self.mail.fetch(email_id, '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        subject = self.decode_header_value(msg['Subject'])
                        from_addr = self.decode_header_value(msg['From'])
                        date = msg['Date']
                        
                        body = self.extract_body(msg)
                        
                        email_data = {
                            'id': email_id.decode(),
                            'subject': subject,
                            'from': from_addr,
                            'date': date,
                            'body': body,
                            'full_content': f"From: {from_addr}\nDate: {date}\nSubject: {subject}\n\n{body}"
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
                        'total_chunks': len(chunks)
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
                    'total_chunks': 1
                })
        
        print(f"Created {len(all_chunks)} chunks from {len(emails)} emails")
        return all_chunks, all_metadata
    
    def get_embeddings_infinity(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Infinity server"""
        if not self.infinity_available:
            return self.get_embeddings_local(texts)
            
        print(f"Generating embeddings using Infinity server at {self.infinity_url}...")
        
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
        print("Generating embeddings using local model...")
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
    
    def perform_clustering(self, n_clusters: Optional[int] = None):
        """Perform clustering on the embeddings"""
        if self.embeddings_array is None:
            print("No embeddings available for clustering")
            return
        
        print("Performing clustering analysis...")
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_chunks = len(self.email_chunks)
            n_clusters = min(max(3, n_chunks // 10), 15)  # 3-15 clusters based on data size
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(self.embeddings_array)
        
        # DBSCAN clustering for comparison
        # Normalize embeddings for DBSCAN
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(self.embeddings_array)
        
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(normalized_embeddings)
        
        # Store clustering results
        self.clusters = {
            'kmeans': {
                'labels': kmeans_labels,
                'n_clusters': n_clusters,
                'centers': kmeans.cluster_centers_
            },
            'dbscan': {
                'labels': dbscan_labels,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            }
        }
        
        self.cluster_labels = kmeans_labels  # Use K-means as default
        
        print(f"K-means clustering: {n_clusters} clusters")
        print(f"DBSCAN clustering: {self.clusters['dbscan']['n_clusters']} clusters")
    
    def generate_2d_coordinates(self):
        """Generate 2D coordinates using UMAP for visualization"""
        if self.embeddings_array is None:
            print("No embeddings available for 2D projection")
            return
        
        print("Generating 2D coordinates using UMAP...")
        
        # Use UMAP for dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            metric='cosine'
        )
        
        self.coordinates_2d = reducer.fit_transform(self.embeddings_array)
        
        print("2D coordinates generated successfully")
    
    def get_visualization_data(self) -> Dict:
        """Get data formatted for visualization"""
        if not all([self.coordinates_2d is not None, self.cluster_labels is not None]):
            return {"error": "Clustering and 2D coordinates not available"}
        
        data_points = []
        
        for i, (coord, cluster, metadata, chunk) in enumerate(zip(
            self.coordinates_2d, 
            self.cluster_labels, 
            self.chunk_metadata, 
            self.email_chunks
        )):
            data_points.append({
                'id': i,
                'x': float(coord[0]),
                'y': float(coord[1]),
                'cluster': int(cluster),
                'subject': metadata['subject'],
                'from': metadata['from'],
                'date': metadata['date'],
                'chunk_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'email_id': metadata['email_id']
            })
        
        # Calculate cluster statistics
        cluster_stats = {}
        unique_clusters = set(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_points = [p for p in data_points if p['cluster'] == cluster_id]
            cluster_stats[cluster_id] = {
                'size': len(cluster_points),
                'emails': list(set(p['email_id'] for p in cluster_points)),
                'common_words': self._extract_common_words([p['chunk_preview'] for p in cluster_points])
            }
        
        return {
            'points': data_points,
            'clusters': cluster_stats,
            'total_chunks': len(data_points),
            'total_clusters': len(unique_clusters)
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
            self.clusters = data['clusters']
            self.coordinates_2d = data['coordinates_2d']
            self.cluster_labels = data['cluster_labels']
            
            # Rebuild FAISS index
            if self.embeddings_array is not None:
                self.build_index(self.embeddings_array)
            
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