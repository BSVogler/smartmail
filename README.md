# SmartMail - Email Semantic Search

Index and search your emails using semantic embeddings and natural language queries.

## Features

- Connect to any IMAP email server
- Semantic chunking preserves email context
- Fast embeddings via Infinity server
- In-memory FAISS vector search
- Interactive query interface

## Setup

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your IMAP credentials
```

4. (Optional) Start Infinity server for fast embeddings:
```bash
infinity_emb v2 --model-id BAAI/bge-small-en-v1.5 --port 7997
```

5. Run the indexer:
```bash
uv run python email_indexer.py
```

## Configuration

Set these environment variables in your `.env` file:

- `IMAP_SERVER`: Your email server (e.g., imap.gmail.com)
- `IMAP_PORT`: IMAP port (usually 993)
- `EMAIL_ADDRESS`: Your email address
- `EMAIL_PASSWORD`: Your email password or app password
- `INFINITY_URL`: Infinity server URL (default: http://localhost:7997)
- `INFINITY_MODEL`: Embedding model for Infinity (default: BAAI/bge-small-en-v1.5)
- `LOCAL_EMBEDDING_MODEL`: Local fallback model (default: BAAI/bge-small-en-v1.5)

## Usage

The script will:
1. Connect to your email account
2. Fetch the last 50 emails
3. Perform semantic chunking
4. Generate embeddings (via Infinity or locally)
5. Build a searchable index
6. Launch interactive search interface

Type natural language queries to find relevant emails. The system uses semantic similarity to find related content even if exact words don't match.

# Planned features
Starting for private use cases.

- Auto categorization (moves mails into folders based on semantic similarity)
- Automatic Diary (summaize what is going on in your life)
  - MCP server for other LLMs to access it
- Summarize informative emails
- For urgent mails inform via telegram bot
- Track spendings/bills and reference them in emails
- only keep the latest e-mail for parcel tracing updates
- MCP server to allow other LLMs to access specific emails (like a folder)