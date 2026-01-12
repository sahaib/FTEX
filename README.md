# 🎫 FTEX - Freshdesk Ticket Extraction & Analysis

> **Production-grade pipeline for extracting, analyzing, and generating actionable insights from Freshdesk support tickets using Self-Validating GenAI.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-qwen3:14b-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-6.0-orange.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [What's New in v6.0](#whats-new-in-v60)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [test - Check API Connection](#test---check-api-connection)
  - [extract - Download Tickets](#extract---download-tickets)
  - [analyze - Run AI Analysis](#analyze---run-ai-analysis)
  - [full - Complete Pipeline](#full---complete-pipeline)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Domain Customization (UserConfig)](#domain-customization-userconfig)
  - [SLA Configuration](#sla-configuration)
- [Analysis Pipeline](#analysis-pipeline)
- [Customization Examples](#customization-examples)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

FTEX is a comprehensive toolkit for support operations teams to:
- 🚀 **Extract** tickets from Freshdesk with checkpointing (survives interruptions)
- 🧠 **Analyze** using self-validating GenAI that discovers patterns from YOUR data
- 📊 **Generate** evidence-based reports with specific ticket IDs for immediate action

**Key Innovations in v6.0:**
- 🎯 Every finding backed by evidence (ticket IDs)
- 🔍 AI self-validation (challenges its own conclusions)
- 💡 Solution quality analysis (evaluates how well issues were resolved)
- 🔧 Fully configurable for any product/domain (maritime, retail, SaaS, IoT)

---

## What's New in v6.0

| Before (v5) | Now (v6) |
|-------------|----------|
| 4 separate analysis scripts | 1 unified `analyze.py` |
| Hardcoded categories | AI discovers categories from YOUR data |
| Trust AI output | Self-validating with confidence scores |
| Generic reports | Evidence-based findings with ticket IDs |
| Domain-specific code | Configurable via `UserConfig` class |
| Separate report generation | Single command generates all outputs |

---

## Features

### Extraction (`freshdesk_extractor_v2.py`)
- ✅ Incremental disk saves (each ticket saved immediately)
- ✅ Checkpoint/resume support (crash-safe)
- ✅ Rich terminal UI with live progress dashboard
- ✅ Weekly date chunking for optimal API usage
- ✅ Rate limit monitoring and auto-throttling
- ✅ Optional attachment downloads

### Smart Detection Engine (`smart_detection.py`) 🆕
- ✅ Pure GenAI analysis (AI reads actual ticket content)
- ✅ Dynamic category discovery (not predefined)
- ✅ Evidence-based findings (every insight has ticket IDs)
- ✅ Confidence scoring (High/Medium/Low)
- ✅ Self-validation (AI challenges its own findings)
- ✅ Solution quality analysis (evaluates resolutions)
- ✅ Anomaly detection (duplicates, recurring issues, spikes)
- ✅ Fully configurable via `UserConfig` class
- ✅ Knowledge base ready (future RAG integration)
- ✅ Fallback to statistical analysis when AI unavailable

### Unified Analyzer (`analyze.py`) 🆕
- ✅ Single command for complete analysis
- ✅ Beautiful Rich terminal UI with progress
- ✅ Multi-sheet Excel report (professionally formatted)
- ✅ Markdown executive summary
- ✅ PDF report generation
- ✅ Raw JSON data for integrations

### Analysis Capabilities
- ✅ True zombie detection (filters false positives)
- ✅ Entity analysis (vessels, stores, devices, accounts)
- ✅ Temporal pattern detection (emerging/declining issues)
- ✅ SLA compliance tracking (FRT + Resolution)
- ✅ Root cause analysis per category
- ✅ Customer/entity health scoring

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Freshdesk     │────▶│   Extractor      │────▶│   tickets.json  │
│   API           │     │   (v2.py)        │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Ollama         │              │
                        │   (qwen3:14b)    │◀─────────────┤
                        └────────┬─────────┘              │
                                 │                        │
                        ┌────────┴────────────────────────┴───────┐
                        │         Smart Detection Engine          │
                        │         (smart_detection.py)            │
                        ├─────────────────────────────────────────┤
                        │  • Category Discovery (AI-powered)      │
                        │  • Evidence Collection                  │
                        │  • Anomaly Detection                    │
                        │  • Solution Quality Analysis            │
                        │  • Self-Validation                      │
                        │  • Confidence Scoring                   │
                        └────────────────┬────────────────────────┘
                                         │
                        ┌────────────────┴────────────────────────┐
                        │           Unified Analyzer              │
                        │           (analyze.py)                  │
                        └────────────────┬────────────────────────┘
                                         │
                                         ▼
                ┌──────────────────────────────────────────────────┐
                │              Generated Reports                    │
                │  • analysis_report.xlsx (7+ sheets, formatted)   │
                │  • analysis_summary.md (executive summary)       │
                │  • analysis_summary.pdf (PDF version)            │
                │  • analysis_data.json (raw data)                 │
                └──────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required
- **Python 3.9+**
- **Freshdesk API Key** (with ticket read permissions)
- **8GB+ RAM** (for processing)

### Optional (for GenAI features)
- **Ollama** (local LLM runtime)
- **16GB+ RAM** recommended for 14B model (24GB ideal)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ftex.git
cd ftex
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
# Core
pip install requests pandas rich python-dotenv

# Reports
pip install openpyxl markdown

# PDF (optional - choose one)
pip install weasyprint    # Option 1: Pure Python
# pip install pdfkit      # Option 2: Requires wkhtmltopdf
```

### 4. Install Ollama (Optional - for GenAI)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull the recommended model (9.3GB)
ollama pull qwen3:14b
```

---

## Quick Start

### 1. Configure Environment

```bash
# Copy example and edit with your credentials
cp .env.example .env

# Edit .env file:
FRESHDESK_API_KEY=your_api_key_here
FRESHDESK_DOMAIN=yourcompany
FRESHDESK_GROUP_ID=48000615489
```

### 2. Test Connection

```bash
python3 run.py test
```

### 3. Run Full Pipeline

```bash
# Extract → Analyze → Report (all in one)
python3 run.py full --days 180
```

---

## CLI Reference

### All Commands Overview

```bash
python3 run.py test      # Test API connection
python3 run.py extract   # Download tickets from Freshdesk
python3 run.py analyze   # Run AI analysis + generate reports
python3 run.py full      # Run entire pipeline
python3 run.py --help    # Show help
```

---

### `test` - Check API Connection

```bash
python3 run.py test
python3 run.py test --api-key YOUR_KEY --domain yourcompany
```

---

### `extract` - Download Tickets

```bash
# Basic (uses .env settings)
python3 run.py extract

# With options
python3 run.py extract --days 90                    # Last 90 days
python3 run.py extract --days 365                   # Last year
python3 run.py extract --days 30 --no-attachments   # Skip attachments (faster)
python3 run.py extract --resume                     # Resume interrupted extraction
python3 run.py extract --group-id 48000615489       # Specific group
```

| Flag | Description | Default |
|------|-------------|---------|
| `--days`, `-d` | Days of history | 180 |
| `--group-id`, `-g` | Filter by group ID | From .env |
| `--no-attachments` | Skip downloading attachments | False |
| `--resume` | Resume from checkpoint | False |
| `--api-key`, `-k` | Override API key | From .env |

---

### `analyze` - Run AI Analysis

```bash
# Full AI analysis (requires Ollama running)
python3 run.py analyze

# Statistical only (no AI required)
python3 run.py analyze --no-ai

# Force re-discovery of categories
python3 run.py analyze --clear-cache

# Custom input/output
python3 run.py analyze --input data/tickets.json --output my_reports/
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input JSON file | output/tickets.json |
| `--output`, `-o` | Output directory | reports/ |
| `--no-ai` | Disable AI (statistical fallback) | False |
| `--clear-cache` | Clear cached categories | False |

**Analysis Outputs (Single Command):**
- `analysis_report.xlsx` - Multi-sheet Excel with all insights
- `analysis_summary.md` - Markdown executive summary
- `analysis_summary.pdf` - PDF version
- `analysis_data.json` - Raw data for integrations

---

### `full` - Complete Pipeline

```bash
# Full pipeline: Extract → Analyze → Report
python3 run.py full --days 180

# Skip extraction (use existing tickets.json)
python3 run.py full --skip-extract

# Without AI
python3 run.py full --days 90 --no-ai
```

| Flag | Description | Default |
|------|-------------|---------|
| `--days`, `-d` | Days of history | 180 |
| `--api-key`, `-k` | Freshdesk API key | From .env |
| `--group-id`, `-g` | Filter by group ID | From .env |
| `--skip-extract` | Use existing data | False |
| `--no-attachments` | Skip attachments | False |
| `--no-ai` | Disable AI analysis | False |

---

## Output Files

After running the full pipeline:

```
FTEX/
├── output/
│   ├── tickets.json              # All tickets (combined)
│   ├── tickets.csv               # Flattened for Excel
│   ├── tickets/                  # Individual ticket JSONs
│   └── checkpoints/              # Resume state
│
└── reports/
    ├── analysis_report.xlsx      # Multi-sheet Excel (7+ sheets)
    ├── analysis_summary.md       # Executive summary
    ├── analysis_summary.pdf      # PDF version
    ├── analysis_data.json        # Raw data
    └── analysis_cache.json       # Cached categories
```

### Excel Report Sheets

| Sheet | Description | Key Metrics |
|-------|-------------|-------------|
| **Overview** | Summary metrics | Total tickets, zombie rate, date range |
| **Issue Categories** | AI-discovered categories | Count, zombies, resolution time, root causes |
| **Entities** | Per-entity analysis | Tickets, zombie rate, top issues |
| **Anomalies** | Detected anomalies | Type, severity, ticket IDs |
| **Zombie Tickets** | No-response tickets | ID, subject, reason |
| **SLA Performance** | Compliance metrics | FRT, resolution by priority |
| **Findings** | Evidence-based insights | Confidence, recommendations |

---

## Project Structure

```
FTEX/
├── src/
│   ├── extraction/
│   │   ├── freshdesk_extractor_v2.py    # Ticket extraction with checkpointing
│   │   └── test_freshdesk_api.py        # API connection tester
│   │
│   ├── shared/
│   │   └── smart_detection.py           # Core analysis engine + UserConfig
│   │
│   └── analysis/
│       └── analyze.py                   # Unified analyzer + report generator
│
├── output/                    # Extracted data (gitignored)
│   ├── tickets.json
│   ├── tickets/
│   └── checkpoints/
│
├── reports/                   # Generated reports (gitignored)
│
├── run.py                     # CLI entry point
├── config.py                  # Centralized configuration
├── requirements.txt
├── .env                       # Your secrets (gitignored)
├── .env.example               # Template for new users
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## Configuration

### Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# Required
FRESHDESK_API_KEY=your_api_key_here
FRESHDESK_DOMAIN=yourcompany

# Optional
FRESHDESK_GROUP_ID=48000615489
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b

# Output (Optional)
FTEX_OUTPUT_DIR=.
FTEX_LOG_LEVEL=INFO
```

| Variable | Description | Required |
|----------|-------------|----------|
| `FRESHDESK_API_KEY` | Your Freshdesk API key | ✅ Yes |
| `FRESHDESK_DOMAIN` | Freshdesk subdomain (e.g., `navtor`) | ✅ Yes |
| `FRESHDESK_GROUP_ID` | Default group ID to filter | No |
| `OLLAMA_URL` | Ollama server URL | No |
| `OLLAMA_MODEL` | Preferred LLM model | No |

### `.env` vs `config.py`

| File | Purpose | Commit to Git? |
|------|---------|----------------|
| `.env` | Your secrets (API keys, domain) | ❌ Never |
| `config.py` | Code that reads `.env` + defaults | ✅ Yes |
| `.env.example` | Template for other users | ✅ Yes |

---

### Domain Customization (UserConfig)

Edit `src/shared/smart_detection.py` → `UserConfig` class to configure for YOUR product:

```python
class UserConfig:
    # =========================================================================
    # ENTITY CONFIGURATION
    # What primary entity do you track tickets by?
    # =========================================================================
    ENTITY_NAME = "vessel"              # or "store", "device", "account"
    ENTITY_NAME_PLURAL = "vessels"
    
    # Regex patterns to extract entity from ticket text
    ENTITY_PATTERNS = [
        r'(?:vessel|ship|mv|m/v)[:\s]+([A-Z][A-Za-z0-9\s\-]{2,25})',
        r'imo[:\s]*(\d{7})',
    ]
    
    # =========================================================================
    # PRODUCT CONTEXT
    # =========================================================================
    PRODUCT_NAME = "Digital Logbook System"
    PRODUCT_DESCRIPTION = """
    Maritime compliance software for electronic record-keeping.
    """
    PRODUCT_MODULES = ["Signature", "Sync", "ORB", "Deck Log"]
    
    # =========================================================================
    # KNOWLEDGE BASE (RAG-Ready)
    # =========================================================================
    GLOSSARY = {
        "ORB": "Oil Record Book - maritime compliance document",
        "IMO": "International Maritime Organization",
    }
    
    KNOWN_SOLUTIONS = {
        "sync_failure": {
            "steps": ["Clear local cache", "Force sync from server"],
            "root_cause": "Cache corruption or network timeout",
            "prevention": "Implement automatic cache validation"
        },
    }
    
    ESCALATION_TRIGGERS = [
        "data loss", "compliance", "audit", "legal", "security breach"
    ]
    
    # =========================================================================
    # THRESHOLDS
    # =========================================================================
    DUPLICATE_REQUEST_DAYS = 365
    DUPLICATE_REQUEST_KEYWORDS = ["activation", "license", "renewal"]
    RECURRING_ISSUE_THRESHOLD = 3
    HIGH_FREQUENCY_MULTIPLIER = 3.0
    SPIKE_MULTIPLIER = 2.0
    
    # Confidence scoring
    HIGH_CONFIDENCE_MIN_EVIDENCE = 10
    MEDIUM_CONFIDENCE_MIN_EVIDENCE = 3
    
    # AI settings
    AI_BATCH_SIZE = 30
    AI_VALIDATION_ENABLED = True
    CACHE_CATEGORIES = True
```

---

### SLA Configuration

Edit `config.py` to set SLA thresholds:

```python
@dataclass  
class SLAConfig:
    first_response: Dict[str, int] = field(default_factory=lambda: {
        'Urgent': 1,   # 1 hour
        'High': 4,     # 4 hours
        'Medium': 8,   # 8 hours
        'Low': 24,     # 24 hours
    })
    resolution: Dict[str, int] = field(default_factory=lambda: {
        'Urgent': 4,    # 4 hours
        'High': 24,     # 1 day
        'Medium': 72,   # 3 days
        'Low': 168,     # 7 days
    })
```

---

## Analysis Pipeline

The analysis engine follows a 6-stage evidence-based approach:

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Data Foundation                                    │
│ └── Extract facts: counts, dates, statuses (undisputable)   │
├─────────────────────────────────────────────────────────────┤
│ STAGE 2: AI Category Discovery                              │
│ └── AI reads tickets, proposes categories + keywords        │
│ └── Categories cached for consistency across runs           │
├─────────────────────────────────────────────────────────────┤
│ STAGE 3: Evidence Collection                                │
│ └── Map ALL tickets to categories                           │
│ └── Collect ticket IDs as evidence                          │
├─────────────────────────────────────────────────────────────┤
│ STAGE 4: Anomaly Detection                                  │
│ └── Duplicate requests (same entity, same issue)            │
│ └── Recurring issues (entity has 3+ of same type)           │
│ └── High-frequency entities (>3x average tickets)           │
│ └── Monthly spikes (>2x average)                            │
├─────────────────────────────────────────────────────────────┤
│ STAGE 5: Solution Quality Analysis                          │
│ └── Evaluate resolved ticket solutions                      │
│ └── Compare against known solutions (knowledge base)        │
│ └── Score: Excellent, Good, Acceptable, Poor                │
├─────────────────────────────────────────────────────────────┤
│ STAGE 6: Finding Generation + Validation                    │
│ └── Generate evidence-based findings                        │
│ └── Calculate confidence (High/Medium/Low)                  │
│ └── AI self-validates findings                              │
└─────────────────────────────────────────────────────────────┘
```

### Confidence Scoring

| Confidence | Criteria |
|------------|----------|
| **High** 🟢 | 10+ supporting tickets, no contradictions |
| **Medium** 🟡 | 3-9 supporting tickets |
| **Low** 🔴 | <3 tickets or unvalidated hypothesis |

### True Zombie Detection

FTEX filters out false positives:

| Detected | Actual Status | FTEX Classification |
|----------|---------------|---------------------|
| No conversations | No response | ✅ True Zombie |
| Customer said "Thanks!" | Acknowledgment | ❌ False Positive |
| Customer said "Got it, closing" | Confirmation | ❌ False Positive |
| Customer asked follow-up | Needs response | ✅ True Zombie |

---

## Customization Examples

### Maritime Industry
```python
ENTITY_NAME = "vessel"
ENTITY_PATTERNS = [r'(?:vessel|ship|mv)[:\s]+([A-Z][A-Za-z\-]+)']
PRODUCT_MODULES = ["Signature", "Logbook", "Sync", "Compliance"]
GLOSSARY = {"ORB": "Oil Record Book", "IMO": "International Maritime Organization"}
```

### Retail / POS
```python
ENTITY_NAME = "store"
ENTITY_PATTERNS = [r'(?:store|location|branch)[:\s#]+(\w+)']
PRODUCT_MODULES = ["POS", "Inventory", "Payments", "Reports"]
DUPLICATE_REQUEST_KEYWORDS = ["terminal", "license", "activation"]
```

### SaaS Platform
```python
ENTITY_NAME = "account"
ENTITY_PATTERNS = [r'(?:account|customer|company)[:\s]+([A-Za-z0-9\s]+)']
PRODUCT_MODULES = ["Auth", "API", "Dashboard", "Billing", "Integrations"]
ESCALATION_TRIGGERS = ["data loss", "security", "sso", "downtime"]
```

### IoT / Hardware
```python
ENTITY_NAME = "device"
ENTITY_PATTERNS = [r'(?:device|serial|unit)[:\s]+([A-Z0-9\-]+)']
PRODUCT_MODULES = ["Firmware", "Connectivity", "Sensors", "Gateway"]
RECURRING_ISSUE_THRESHOLD = 2  # Stricter for hardware
```

---

## Troubleshooting

### Extraction Issues

**API Rate Limit Hit**
```
The script auto-throttles at 40 req/min. If you see rate limit errors:
1. Wait 1 hour for quota reset
2. Use --resume to continue
```

**Mac Sleep Interruption**
```bash
caffeinate -i python3 run.py extract --days 180
```

**Resume Not Working**
```bash
ls output/checkpoints/
# Should see: ticket_ids.json, extraction_state.json
```

### Analysis Issues

**Ollama Not Found**
```bash
curl http://localhost:11434/api/tags
# If not running:
ollama serve
```

**Out of Memory**
```bash
# Use smaller model
ollama pull qwen3:8b

# Or run without GenAI
python3 run.py analyze --no-ai
```

**Categories Not Matching**
```bash
# Clear cache and re-discover
python3 run.py analyze --clear-cache
```

### Report Issues

**PDF Not Generated**
```bash
# Install weasyprint
pip install weasyprint

# Or use pdfkit (requires wkhtmltopdf)
pip install pdfkit
# macOS: brew install wkhtmltopdf
# Linux: apt-get install wkhtmltopdf
```

**Excel Formatting Issues**
```bash
pip install --upgrade openpyxl
```

---

## Roadmap

### Phase 1 ✅ (Complete)
- [x] Ticket extraction with checkpointing
- [x] Smart zombie detection (filters false positives)
- [x] Self-validating AI analysis
- [x] Evidence-based findings
- [x] Solution quality analysis
- [x] Multi-sheet Excel reports
- [x] Unified CLI entry point
- [x] Configurable for any domain

### Phase 2 🚧 (In Progress)
- [ ] Web dashboard for real-time monitoring
- [ ] Scheduled extraction (cron/Airflow)
- [ ] Slack/Teams integration for alerts
- [ ] Customer health scoring dashboard

### Phase 3 📋 (Planned)
- [ ] RAG integration for knowledge base
- [ ] Historical trend analysis
- [ ] Predictive ticket routing
- [ ] Agent performance coaching

### Phase 4 🔮 (Vision)
- [ ] SaaS product offering
- [ ] Integration marketplace (Zendesk, Intercom, etc.)
- [ ] AI-powered auto-responses
- [ ] Customer success automation

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Freshdesk API](https://developers.freshdesk.com/) for ticket data
- [Ollama](https://ollama.ai/) for local LLM inference
- [Rich](https://rich.readthedocs.io/) for beautiful terminal UI
- [Pandas](https://pandas.pydata.org/) for data processing
- [OpenPyXL](https://openpyxl.readthedocs.io/) for Excel generation

---

<p align="center">
  <b>Built with ❤️ for Support Operations Teams Everywhere</b>
</p>