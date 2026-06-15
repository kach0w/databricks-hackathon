# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Databricks Apps & Agents for Good Hackathon 2026** — an agentic application for social impact built on the Databricks platform. The stack centers on Lakebase (managed Postgres for agent state), AgentBricks (agent orchestration), Databricks Apps (UI), and a knowledge graph layer (Grafo/Neo4j) for relationship-aware reasoning.

## Stack

- **Lakebase** — PostgreSQL-compatible operational database (agent memory, state, structured domain data). Connect via standard Postgres clients (psycopg2, SQLAlchemy, asyncpg).
- **AgentBricks** — Databricks agent orchestration layer with tool-calling and multi-agent support.
- **Databricks Apps** — hosted UI layer; no separate frontend server needed.
- **Knowledge Graph (Grafo)** — entity/relationship extraction and graph traversal for multi-hop reasoning over domain data.
- **Delta Lake** — lakehouse storage for analytics, ML, and raw ingested data.
- **Anthropic SDK** — Claude API for LLM reasoning inside agents; use prompt caching on system prompts.

## Architecture

```
External Data (APIs, CSVs, real-time feeds)
        ↓
  Delta Lake (raw + processed lakehouse data)
        ↓
  Knowledge Graph (entities + relationships extracted from Delta)
        ↓
  AgentBricks (agent orchestration, tool use, multi-agent routing)
        ↓
  Lakebase (agent memory, session state, operational records)
        ↓
  Databricks Apps (UI / dashboard)
```

Agents read from both the knowledge graph (for relational reasoning) and Lakebase (for fast operational lookups). Analytics and ML run against Delta Lake. The knowledge graph is populated from Delta Tables — no separate ingestion pipeline.

## Development Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set required env vars (never hardcode)
export LAKEBASE_CONNECTION_STRING="..."
export ANTHROPIC_API_KEY="..."
export DATABRICKS_HOST="..."
export DATABRICKS_TOKEN="..."
```

## Commands

Add build/test/lint commands here as the project takes shape:

```bash
# Run tests
pytest

# Lint
ruff check . && ruff format --check .

# Type check
mypy .
```

## Key Conventions

- Agent system prompts must use **prompt caching** (`cache_control: {"type": "ephemeral"}`) — system prompts repeat on every agent turn and are expensive without caching.
- All Lakebase schema changes go through migration files (not ad-hoc ALTER TABLE).
- Knowledge graph writes are append-only during the hackathon; no destructive graph mutations.
- Social impact framing: every feature should map to a concrete harm being reduced and a measurable population affected.
