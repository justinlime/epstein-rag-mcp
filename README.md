# Epstein Files RAG MCP Server

A Model Context Protocol (MCP) server that provides Retrieval-Augmented Generation (RAG) capabilities over the Epstein Files dataset from HuggingFace.

## Features

- 🔍 Semantic search over 20K+ Epstein Files documents
- 🚀 Runs entirely on CPU and RAM
- 💾 Vector storage on NVME via Qdrant Docker
- 🎯 Uses `all-MiniLM-L6-v2` embedding model
- 📦 Zero local files needed - run directly via `uv`

## Prerequisites

1. **Docker** - for running Qdrant
2. **UV** - Python package manager

Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### 1. Start Qdrant Docker Container

```bash
# Create storage directory on your NVME
mkdir -p /path/to/nvme/qdrant_storage

# Run Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /path/to/nvme/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 2. Configure Your LLM Client

Add this to your MCP servers configuration (e.g., Claude Desktop config):

```json
{
  "mcpServers": {
    "epstein-rag": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/justinlime/epstein-rag-mcp.git",
        "epstein-rag-mcp"
      ],
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

### 3. Restart Your LLM Client

That's it! The server will automatically:
- Download and install dependencies
- Load the embedding model
- Fetch the dataset from HuggingFace (first run only)
- Create embeddings and index them in Qdrant
- Be ready to answer queries

## Usage

Once configured, your LLM can use the `search_epstein_files` tool:

**Example queries:**
- "Find documents mentioning flight logs"
- "Search for references to specific individuals"
- "What documents discuss financial transactions?"

## Configuration

### Environment Variables

- `QDRANT_HOST` - Qdrant server host (default: `localhost`)
- `QDRANT_PORT` - Qdrant server port (default: `6333`)

### First Run

The first time you run the server, it will:
1. Download the `all-MiniLM-L6-v2` model (~80MB)
2. Fetch the Epstein Files dataset from HuggingFace (~100MB)
3. Generate embeddings for all documents (10-30 minutes on CPU)
4. Index them in Qdrant

Subsequent runs will be instant as the data is persisted in Qdrant.

## Development

### Local Installation

```bash
git clone https://github.com/yourusername/epstein-rag-mcp.git
cd epstein-rag-mcp
uv pip install -e .
```

### Running Locally

```bash
uv run epstein-rag-mcp
```

## Docker Management

```bash
# View logs
docker logs qdrant

# Stop Qdrant
docker stop qdrant

# Start Qdrant
docker start qdrant

# Access web UI
# Open http://localhost:6333/dashboard
```

## Troubleshooting

### "Failed to connect to Qdrant"
Make sure the Docker container is running:
```bash
docker ps | grep qdrant
```

### Port Already in Use
Change the port mapping:
```bash
docker run -d --name qdrant -p 6335:6333 ...
```
Then update `QDRANT_PORT` to `6335` in your config.

### Slow Indexing
This is normal on CPU. The first run can take 10-30 minutes depending on your hardware. Reduce `BATCH_SIZE` in the code if you run out of memory.

## Architecture

```
┌─────────────┐
│  LLM Client │
└──────┬──────┘
       │ MCP Protocol
       │
┌──────▼──────────────────┐
│  epstein-rag-mcp        │
│  ┌──────────────────┐   │
│  │ Sentence         │   │
│  │ Transformer      │   │
│  │ (all-MiniLM-L6)  │   │
│  └─────────┬────────┘   │
│            │             │
│  ┌─────────▼────────┐   │
│  │ Dataset Loader   │   │
│  │ (HuggingFace)    │   │
│  └──────────────────┘   │
└────────┬─────────────────┘
         │
    ┌────▼─────┐
    │  Qdrant  │
    │  Docker  │
    └────┬─────┘
         │
    ┌────▼─────┐
    │   NVME   │
    └──────────┘
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Credits

- Dataset: [tensonaut/EPSTEIN_FILES_20K](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K)
- Embedding Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Vector Store: [Qdrant](https://qdrant.tech/)
