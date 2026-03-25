# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraphRAG-based knowledge graph system for academic document analysis (specifically algorithm design and analysis textbooks). The pipeline converts PDFs to knowledge graphs and stores them in Neo4j for querying.

**Tech Stack**: Microsoft GraphRAG + Neo4j + MinerU API + Alibaba Qwen models

## Core Commands

All GraphRAG commands must be run from the `graphrag/` directory:

```bash
cd graphrag

# Initialize GraphRAG (first time only)
graphrag init --root ./

# Index documents to build knowledge graph
graphrag index --root .

# Tune prompts for domain-specific entity extraction (optional)
graphrag prompt-tune --root . --config ./settings.yaml --language Chinese --output ./prompts --discover-entity-types
```

Run from project root:

```bash
# Convert PDFs to Markdown using MinerU API
python miner_api_parser.py

# Import GraphRAG output into Neo4j database
python graphrag2neo4j.py
```

## Architecture & Data Flow

**Pipeline**: PDF → MinerU → Markdown → GraphRAG → Parquet files → Neo4j

1. **PDF Parsing** (`miner_api_parser.py`):
   - Batch uploads PDFs from `raw_files/` to MinerU API
   - Polls for completion and downloads Markdown to `graphrag/input/`
   - Uses async batch processing with unique data_id tracking

2. **Knowledge Graph Extraction** (`graphrag/`):
   - Reads Markdown files from `input/`
   - Uses Alibaba Qwen models via OpenAI-compatible API
   - Extracts entities, relationships, communities with embeddings
   - Outputs Parquet files to `output/`: entities, relationships, communities, text_units, documents, embeddings

3. **Neo4j Import** (`graphrag2neo4j.py`):
   - Reads Parquet files from `graphrag/output/`
   - Creates nodes: Entity, Community, TextUnit, Document
   - Creates relationships: RELATED, IN_COMMUNITY, HAS_PART, MENTIONS
   - Batch imports with progress tracking (1000 rows/batch)

## Configuration

**Required `.env` variables**:
- `MINERU_API_KEY`: MinerU API authentication
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Database connection
- `GRAPHRAG_API_KEY`, `GRAPHRAG_API_BASE`: Alibaba DashScope credentials
- `GRAPHRAG_CHAT_MODEL`, `GRAPHRAG_EMBEDDING_MODEL`: Model names

**Important `graphrag/settings.yaml` customizations**:
- Uses Alibaba models with `model_provider: openai` (compatibility mode)
- `concurrent_requests: 5` to avoid rate limiting
- `file_pattern: ".*(\\.txt|\\.md|\\.csv|\\.json)$$"` for Markdown input
- Custom `entity_types` for algorithm analysis domain
- `batch_size: 10` for embeddings (Alibaba limit)

## Critical Source Code Modification

GraphRAG's LiteLLM provider requires modification for Alibaba embedding model compatibility:

**File**: `<python_env>/Lib/site-packages/graphrag/language_model/providers/litellm/embedding_model.py`

**Function**: `_base_aembedding`

**Required change**:
```python
new_args = {**self.kwargs, **kwargs}
new_args["encoding_format"] = "float"  # Add this line
return await aembedding(**new_args)
```

Without this modification, embedding generation will fail with Alibaba models.

## Directory Structure

```
graphrag-agent/
├── config.py                    # Centralized config using environment variables
├── miner_api_parser.py          # PDF → Markdown converter (MinerU API)
├── graphrag2neo4j.py            # Parquet → Neo4j importer
├── raw_files/                   # Input: PDF files to process
└── graphrag/                    # GraphRAG workspace
    ├── settings.yaml            # GraphRAG configuration
    ├── .env                     # API keys and credentials
    ├── input/                   # Markdown files for indexing
    ├── output/                  # Generated Parquet files
    ├── prompts/                 # Custom extraction prompts
    ├── cache/                   # LLM response cache
    └── logs/                    # Processing logs
```

## Domain-Specific Notes

This project analyzes algorithm textbooks with custom entity types: `algorithm`, `complexity_class`, `optimization_problem`, `mathematical_model`, `linear_programming`, `constraint`, `objective_function`, `solution_method`, `theoretical_concept`, `computational_problem`.

When modifying entity extraction, ensure the `entity_types` list in `settings.yaml` matches the types defined in `prompts/extract_graph.txt`.
