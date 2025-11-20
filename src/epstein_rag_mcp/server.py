"""
Epstein Files RAG MCP Server
Main server implementation
"""

import os
import sys
import asyncio
from typing import Any, Sequence, List, Dict

# Core dependencies
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
import tiktoken

# MCP SDK
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server

# Configuration
DATASET_NAME = "tensonaut/EPSTEIN_FILES_20K"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "epstein_files"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MAX_TOKENS_PER_RESULT = int(os.getenv("MAX_TOKENS_PER_RESULT", "150"))
BATCH_SIZE = 100

# Debug: Print environment variables on startup
print(f"DEBUG: QDRANT_HOST env var = {os.getenv('QDRANT_HOST')}", file=sys.stderr)
print(f"DEBUG: QDRANT_PORT env var = {os.getenv('QDRANT_PORT')}", file=sys.stderr)
print(f"DEBUG: MAX_TOKENS_PER_RESULT env var = {os.getenv('MAX_TOKENS_PER_RESULT')}", file=sys.stderr)
print(f"DEBUG: Using QDRANT_HOST = {QDRANT_HOST}", file=sys.stderr)
print(f"DEBUG: Using QDRANT_PORT = {QDRANT_PORT}", file=sys.stderr)
print(f"DEBUG: Using MAX_TOKENS_PER_RESULT = {MAX_TOKENS_PER_RESULT}", file=sys.stderr)


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to a maximum number of tokens."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    except Exception as e:
        print(f"Warning: Token truncation failed, falling back to character truncation: {e}", file=sys.stderr)
        # Fallback to character-based truncation (roughly 4 chars per token)
        return text[:max_tokens * 4] + "..."


class EpsteinRAGServer:
    """RAG server for Epstein Files dataset with Qdrant vector store."""

    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.collection_initialized = False

    def initialize(self):
        """Initialize embedding model and Qdrant client."""
        print("Initializing embedding model...", file=sys.stderr)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...", file=sys.stderr)
        try:
            self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            # Test connection
            self.qdrant_client.get_collections()
            print("Successfully connected to Qdrant!", file=sys.stderr)
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}", file=sys.stderr)
            print("Make sure Qdrant Docker container is running on the specified host:port", file=sys.stderr)
            raise

        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)

        if not collection_exists:
            print("Collection doesn't exist. Loading and indexing dataset...", file=sys.stderr)
            self._load_and_index_dataset()
        else:
            print("Collection already exists. Ready to query.", file=sys.stderr)
            self.collection_initialized = True

    def _load_and_index_dataset(self):
        """Load dataset from HuggingFace and index into Qdrant."""
        print(f"Loading dataset: {DATASET_NAME}...", file=sys.stderr)
        dataset = load_dataset(DATASET_NAME, split="train")

        # Convert to pandas for easier handling
        df = pd.DataFrame(dataset)
        print(f"Dataset loaded with {len(df)} rows", file=sys.stderr)

        # Get embedding dimension
        sample_embedding = self.embedding_model.encode("test")
        embedding_dim = len(sample_embedding)

        # Create collection
        print("Creating Qdrant collection...", file=sys.stderr)
        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )

        # Prepare text for embedding (combine relevant fields)
        print("Preparing documents for indexing...", file=sys.stderr)
        documents = []
        for idx, row in df.iterrows():
            # Combine all text fields into a searchable document
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and isinstance(row[col], str):
                    text_parts.append(f"{col}: {row[col]}")
            doc_text = " | ".join(text_parts)
            documents.append({
                'id': idx,
                'text': doc_text,
                'metadata': row.to_dict()
            })

        # Index in batches
        print(f"Indexing {len(documents)} documents in batches of {BATCH_SIZE}...", file=sys.stderr)
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            texts = [doc['text'] for doc in batch]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

            points = [
                PointStruct(
                    id=doc['id'],
                    vector=embeddings[j].tolist(),
                    payload={
                        'text': doc['text'],
                        **doc['metadata']
                    }
                )
                for j, doc in enumerate(batch)
            ]

            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

            if (i + BATCH_SIZE) % 1000 == 0:
                print(f"Indexed {i + BATCH_SIZE} documents...", file=sys.stderr)

        print("Indexing complete!", file=sys.stderr)
        self.collection_initialized = True

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store for relevant documents."""
        print(f"DEBUG search(): query='{query}', limit={limit}", file=sys.stderr)
        
        if not self.collection_initialized:
            error_msg = "Collection not initialized"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            raise RuntimeError(error_msg)

        try:
            print("DEBUG: Encoding query...", file=sys.stderr)
            # Encode query
            query_vector = self.embedding_model.encode(query).tolist()
            print(f"DEBUG: Query vector length: {len(query_vector)}", file=sys.stderr)

            print("DEBUG: Searching Qdrant...", file=sys.stderr)
            # Search
            search_result = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )
            print(f"DEBUG: Qdrant returned {len(search_result)} results", file=sys.stderr)

            # Format results
            results = []
            for hit in search_result:
                results.append({
                    'score': hit.score,
                    'text': hit.payload.get('text', ''),
                    'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
                })

            print(f"DEBUG: Formatted {len(results)} results", file=sys.stderr)
            return results
        
        except Exception as e:
            print(f"ERROR in search(): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise


# Initialize the RAG server
rag_server = EpsteinRAGServer()

# Create MCP server
mcp_server = Server("epstein-rag-server")


@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_epstein_files",
            description="Search through the Epstein Files dataset using semantic search. Returns the most relevant documents based on the query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 3, max: 5)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": f"Maximum tokens per result excerpt (default: {MAX_TOKENS_PER_RESULT})",
                        "default": MAX_TOKENS_PER_RESULT,
                        "minimum": 50,
                        "maximum": 1000
                    }
                },
                "required": ["query"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    print(f"DEBUG: call_tool invoked with name='{name}'", file=sys.stderr)
    print(f"DEBUG: arguments type: {type(arguments)}", file=sys.stderr)
    print(f"DEBUG: arguments content: {arguments}", file=sys.stderr)
    
    if name != "search_epstein_files":
        error_msg = f"Unknown tool: {name}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise ValueError(error_msg)

    try:
        print("DEBUG: Extracting query from arguments...", file=sys.stderr)
        query = arguments.get("query") if isinstance(arguments, dict) else None
        limit = arguments.get("limit", 3) if isinstance(arguments, dict) else 3
        max_tokens_per_result = arguments.get("max_tokens", MAX_TOKENS_PER_RESULT) if isinstance(arguments, dict) else MAX_TOKENS_PER_RESULT
        
        # Cap limit to prevent context overflow
        limit = min(limit, 5)
        
        print(f"DEBUG: Extracted query='{query}', limit={limit}, max_tokens={max_tokens_per_result}", file=sys.stderr)

        if not query:
            error_msg = "Query parameter is required"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            raise ValueError(error_msg)

        print(f"DEBUG: collection_initialized={rag_server.collection_initialized}", file=sys.stderr)
        print(f"DEBUG: About to call search...", file=sys.stderr)

        # Perform search in thread to avoid blocking
        results = await asyncio.to_thread(rag_server.search, query, limit)

        print(f"DEBUG: Search completed, got {len(results)} results", file=sys.stderr)

        # Format response with token-limited excerpts
        response_text = f"Found {len(results)} relevant documents:\n\n"
        for i, result in enumerate(results, 1):
            # Truncate to token limit
            text_excerpt = truncate_to_tokens(result['text'], max_tokens_per_result)
            response_text += f"{i}. [Score: {result['score']:.3f}]\n{text_excerpt}\n\n"

        print(f"DEBUG: Returning response with {len(response_text)} chars", file=sys.stderr)
        return [TextContent(type="text", text=response_text)]
    
    except Exception as e:
        error_msg = f"Error executing search: {str(e)}\nType: {type(e).__name__}"
        print(f"ERROR in call_tool: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Return error as TextContent so it doesn't fail silently
        return [TextContent(type="text", text=f"Search failed: {error_msg}")]


async def main():
    """Main entry point for the MCP server."""
    try:
        # Initialize RAG system
        print("Starting RAG server initialization...", file=sys.stderr)
        rag_server.initialize()
        print("RAG server initialized successfully!", file=sys.stderr)
    except Exception as e:
        print(f"FATAL: Failed to initialize RAG server: {e}", file=sys.stderr)
        raise

    # Run MCP server
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


def main_sync():
    """Synchronous entry point for console script."""
    import argparse

    parser = argparse.ArgumentParser(description="Epstein Files RAG MCP Server")
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"),
                       help="Qdrant host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")),
                       help="Qdrant port (default: 6333)")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS_PER_RESULT", "150")),
                       help="Maximum tokens per result excerpt (default: 150)")
    args = parser.parse_args()

    # Override global config with CLI args
    global QDRANT_HOST, QDRANT_PORT, MAX_TOKENS_PER_RESULT
    QDRANT_HOST = args.qdrant_host
    QDRANT_PORT = args.qdrant_port
    MAX_TOKENS_PER_RESULT = args.max_tokens

    print(f"Starting with QDRANT_HOST={QDRANT_HOST}, QDRANT_PORT={QDRANT_PORT}, MAX_TOKENS_PER_RESULT={MAX_TOKENS_PER_RESULT}", file=sys.stderr)

    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
