# nlporygon

**Natural Language to SQL for Large Schemas**

Transform natural language questions into SQL queries using LLMs, with intelligent schema compression that makes even massive databases queryable without blowing your token budget.

---

## What is nlporygon?

nlporygon is a Python library that enables natural language querying of databases through LLMs with minimal setup. 
Unlike naive approaches that dump entire schemas into prompts, nlporygon uses a compression system that replaces verbose table, column, and type names with short aliases—dramatically reducing token usage while preserving the structural information LLMs need to generate accurate queries.

**Key capabilities:**

- **Schema Compression** — Reduces prompt size by 60-80% through intelligent aliasing
- **Automatic Relationship Detection** — Discovers foreign keys from DB metadata *and* infers implicit relationships using bloom filters
- **Self-Correcting Queries** — Retries failed queries with error feedback so the LLM can fix its mistakes
- **Context Queries** — Lets the LLM request intermediate data (like valid enum values) before generating the final query
- **Partition Support** — Split large schemas into logical partitions with automatic query routing

---

## Why nlporygon?

### The Problem

LLMs are great at writing SQL, but they need schema context. For small databases, you can paste the schema directly. For real-world databases with hundreds of tables, you quickly hit token limits—and costs spiral.

### The Solution

nlporygon compresses your schema into a token-efficient format:

```
-- Before: ~150 tokens
CREATE TABLE customer_transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    amount DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP
);

-- After: ~30 tokens
t<A>(c<A> d<A>, c<B> d<A> REF[t<B>(c<A>)], c<C> d<B>, c<D> d<C>, c<E> d<D>)
```

The LLM receives the compressed schema plus legend files that map aliases back to real names. It learns to read this format quickly and generates accurate SQL.

---

## Features

- **Schema Introspection** — Automatically discovers tables, columns, types, and relationships
- **Sub-type Detection** — Identifies VARCHAR columns that actually store JSON or arrays
- **Bloom Filter FK Detection** — Finds implicit foreign key relationships even when not declared in the schema
- **Compressed Prompts** — Excel-style aliases (t\<A\>, c\<B\>, d\<C\>) minimize token usage
- **Legend Files** — Bidirectional mappings for encoding prompts and decoding LLM output
- **Context Queries** — LLM can request sample data before writing the final query
- **Retry with Feedback** — Failed queries are retried with error messages for self-correction
- **Schema Partitioning** — Split large schemas and route queries to the right partition
- **Structured Logging** — JSON logging via structlog for observability

---

## Installation

```bash
uv add nlporygon
```

**Requirements:**
- Python 3.13+
- An Anthropic API key (for Claude)

---

## Quick Start

```python
import asyncio
import duckdb
from anthropic import AsyncAnthropic
from pathlib import Path

from nlporygon.database import DuckDb
from nlporygon.models import Config
from nlporygon.schema_builder import generate_table_definitions
from nlporygon.generate_prompt import generate_prompts
from nlporygon.llm_agent import MainAgent

async def main():
    # 1. Connect to your database
    db = DuckDb(
        name="analytics",
        database_version="1.4.3",
        connection=duckdb.connect("path/to/database.db", read_only=True)
    )

    # 2. Configure nlporygon
    config = Config(output_path=Path("./nlporygon_output"))

    # 3. Generate schema definitions (run once, or when schema changes)
    await generate_table_definitions(config, db)

    # 4. Generate compressed prompts
    await generate_prompts(config, db)

    # 5. Query with natural language
    agent = MainAgent(AsyncAnthropic(), config, db)
    results = await agent.query("What are the top 10 customers by total order value?")
    print(results)

asyncio.run(main())
```

---

## Usage Guide

### Step 1: Generate Table Definitions

Table definitions capture your schema structure—tables, columns, data types, and relationships. Run this once initially, then again whenever your schema changes.

```python
from pathlib import Path
from nlporygon.database import DuckDb
from nlporygon.models import Config, ColumnConfig, TableConfig
from nlporygon.schema_builder import generate_table_definitions
import duckdb

# Connect to your database
db = DuckDb(
    name="analytics",
    database_version="1.4.3",
    connection=duckdb.connect("/path/to/analytics.db", read_only=True)
)

# Configure output and filtering
config = Config(
    output_path=Path("./output"),
    column_relationships=ColumnConfig(
        # Ignore columns that shouldn't be used for relationship detection
        global_ignore_column_rules=["pii_column", "internal_column"]
    ),
    table_config=TableConfig(
        # Only include tables matching these patterns
        include_table_rules=["analytics.main.*"],
        # Exclude internal/system tables
        ignore_table_rules=["analytics.main.migrations.*"]
    )
)

# Generate definitions
await generate_table_definitions(config, db)
```

**Output structure:**

```
output/
└── schema/
    ├── customers.yaml
    ├── orders.yaml
    ├── products.yaml
    └── ...
```

Each YAML file contains the table definition:

```yaml
name: analytics.main.customers
columns:
  - name: id
    data_type: INTEGER
  - name: email
    data_type: VARCHAR
  - name: created_at
    data_type: TIMESTAMP
  - name: metadata
    data_type: VARCHAR
    sub_data_type: JSON  # Detected from actual data
    nested_columns:
      - name: metadata->plan
        data_type: VARCHAR
      - name: metadata->seats
        data_type: INTEGER
default_order:
  - created_at
```

### Step 2: Generate Prompts

Convert the schema definitions into compressed prompts for the LLM.

```python
from nlporygon.generate_prompt import generate_prompts

await generate_prompts(config, db)
```

**Output structure:**

```
output/
└── prompt/
    └── default/
        ├── sys_prompt.txt      # Compressed schema + instructions
        ├── table_legend.txt    # t<A> -> actual_table_name
        ├── column_legend.txt   # c<A> -> actual_column_name
        └── data_type_legend.txt # d<A> -> actual_data_type
```

**Compressed schema format:**

```
t<A>(c<A> d<A>, c<B> d<B>, c<C> d<C> REF[t<B>(c<A>)], c<D> d<D> --> JSON)
```

Where:
- `t<A>` = table alias
- `c<A>` = column alias
- `d<A>` = data type alias
- `REF[t<B>(c<A>)]` = foreign key to table B, column A
- `--> JSON` = underlying type for serialized columns

### Step 3: Query with the Agent

Use the `MainAgent` to convert natural language to SQL and execute queries.

```python
from anthropic import AsyncAnthropic
from nlporygon.llm_agent import MainAgent

agent = MainAgent(
    AsyncAnthropic(),  # Uses ANTHROPIC_API_KEY env var
    config,
    db,
)

# Ask questions in natural language
results = await agent.query("How many orders were placed last month?")
print(results)  # [{"count": 1523}]

results = await agent.query("What are the top 5 products by revenue?")
print(results)  # [{"product_name": "...", "revenue": ...}, ...]

results = await agent.query("Show me all customers who signed up this week")
print(results)
```

---

## Configuration Reference

### Config

The main configuration object.

```python
from nlporygon.models import Config, ColumnConfig, TableConfig

config = Config(
    # Required: where to store schema and prompt files
    output_path=Path("./output"),

    # Optional: column relationship detection settings
    column_relationships=ColumnConfig(...),

    # Optional: table filtering and partitioning
    table_config=TableConfig(...),
)
```

### ColumnConfig

Controls relationship detection behavior.

```python
from nlporygon.models import ColumnConfig

column_config = ColumnConfig(
    # Columns to ignore when detecting relationships (e.g., tenant IDs)
    global_ignore_column_rules=["org_id", "tenant_id"],

    # Sample size for relationship detection (default: 20,000)
    sample_size=20_000,

    # Bloom filter size for relationship detection (default: 1,000,000)
    bloom_size=1_000_000,

    # Bloom filter error rate (default: 0.01 = 1%)
    error_rate=0.01,
)
```

### TableConfig

Controls which tables to include and how to partition them.

```python
from nlporygon.models import TableConfig, TablePartitionConfig, CommonTableRule

table_config = TableConfig(
    # Regex patterns for tables to include (empty = include all)
    include_table_rules=["analytics.main.*"],

    # Regex patterns for tables to exclude (takes precedence over include)
    ignore_table_rules=[".*_backup", ".*_tmp"],

    # Tables shared across all partitions
    common_table=CommonTableRule(
        include_table_rules=["analytics.main.customers.*"]
    ),

    # Schema partitions for large databases
    partitions=[
        TablePartitionConfig(
            name="tickets",
            description="Support tickets and issue tracking data",
            include_table_rules=[".*ticket.*", ".*issue.*"]
        ),
        TablePartitionConfig(
            name="activity",
            description="User activity logs and events",
            include_table_rules=["analytics.main.activity.*"]
        ),
    ]
)
```

#### Common Table
`common_table` is a way to specify which tables should be included in all partitions. 
These are your core tables in your Database like the User table.

#### Partitions
Partitions allow you to chunk your schema into tables that fulfil a shared responsibility like ticketing or customer orders. 

Once your schema yaml files have been generated, define the partitions in your Config. 

When partitions are configured, `MainAgent` automatically routes queries to the appropriate partition based on the question content.
It does this by passing the user prompt to the LLM along with the descriptions of each partition. 
The LLM then makes a decision which partition is the best fit based on this info. 

---

## Supported Databases

| Database | Status | Import |
|----------|--------|--------|
| DuckDB   | ✅ Supported | `from nlporygon.database import DuckDb` |
| PostgreSQL | 🔜 Coming soon | — |
| Snowflake | 🔜 Coming soon | — |


## Supported LLMs

| Provider | Model | Status |
|----------|-------|--------|
| Anthropic | Claude Sonnet | ✅ Supported |
| OpenAI | GPT-4 | 🔜 Coming soon |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Yes |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No (default: INFO) |

---

## How It Works

### 1. Schema Introspection

When you call `generate_table_definitions()`, nlporygon:

1. **Queries database metadata** to discover all tables and columns
2. **Samples column data** to detect sub-types (e.g., VARCHAR columns that store JSON)
3. **Builds nested column maps** for JSON columns to expose their internal structure
4. **Sets default ordering** for each table (date columns → primary keys → first column)
5. **Writes YAML definitions** that can be version-controlled and manually edited

### 2. Relationship Detection

nlporygon uses a two-phase approach to find relationships:

**Phase 1: Explicit Foreign Keys**
- Queries `INFORMATION_SCHEMA` (or equivalent) for declared FK constraints
- These relationships are guaranteed accurate

**Phase 2: Bloom Filter Heuristics**

For columns without declared FKs, nlporygon infers relationships:

1. **Identifies potential primary keys** — columns where every value is unique
2. **Creates bloom filters** for each potential PK column (space-efficient probabilistic data structure)
3. **Tests subset relationships** — for each other column, checks if its values appear in the PK's bloom filter
4. **Verifies with actual queries** — bloom filters can have false positives, so matches are verified against the database
5. **Adds discovered relationships** — when confirmed, the relationship is added to the schema

This catches common patterns like `user_id` columns that reference `users.id` even when the FK isn't declared in the schema.

### 3. Prompt Compression

The `generate_prompts()` function:

1. **Assigns aliases** using Excel-style naming:
   - Tables: `t<A>`, `t<B>`, ... `t<Z>`, `t<AA>`, ...
   - Columns: `c<A>`, `c<B>`, ... (globally unique across all tables)
   - Data types: `d<A>`, `d<B>`, ...

2. **Generates compressed schema** in a dense notation:
   ```
   t<A>(c<A> d<A>, c<B> d<B> REF[t<B>(c<C>)], c<D> d<C> --> INTEGER[])
   ```

3. **Creates legend files** for bidirectional mapping (encode prompts, decode LLM output)

4. **Applies prompt caching** via Anthropic's cache control to reduce costs on repeated queries

The compression typically reduces prompt size by **60-80%**, which:
- Fits larger schemas in the context window
- Reduces API costs proportionally
- Speeds up response times (fewer tokens to process)

### 4. Query Execution

When you call `agent.query()`:

```
┌─────────────────┐
│ Natural Language│
│    Question     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Route to        │────▶│ Select Partition│ (if partitions configured)
│ Partition       │     │ via LLM         │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Send to LLM     │◀─── Compressed schema + legends
│ with Schema     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Context Query?  │────▶│ Execute & Feed  │ (INTERNAL_CONTEXT_QUERY)
│                 │     │ Results Back    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │◀──────────────────────┘
         ▼
┌─────────────────┐
│ Execute Final   │
│ SQL Query       │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Success?│
    └────┬────┘
         │
    ┌────┴────┐
    │  Yes    │───────────────────────────▶ Return Results
    └────┬────┘
         │
    ┌────┴────┐
    │   No    │───▶ Retry with error feedback (up to max_attempts)
    └─────────┘
```

The retry loop (default: 2 attempts) lets the LLM self-correct common mistakes like:
- Typos in table/column names
- Incorrect JOIN conditions
- Wrong data type handling
- Missing WHERE clauses

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
