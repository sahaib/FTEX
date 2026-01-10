# 🎫 FTEX - Freshdesk Ticket Extraction & Analysis

> **Production-grade pipeline for extracting, analyzing, and generating actionable insights from Freshdesk support tickets using GenAI.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-qwen3:14b-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI Entry Point](#cli-entry-point)
  - [1. Test API Connection](#1-test-api-connection)
  - [2. Extract Tickets](#2-extract-tickets)
  - [3. Run GenAI Analysis](#3-run-genai-analysis)
  - [4. Run Deep AI Analysis](#4-run-deep-ai-analysis)
  - [5. Generate Reports](#5-generate-reports)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Analysis Parameters](#analysis-parameters)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

FTEX is a comprehensive toolkit for support operations teams to extract tickets from Freshdesk, perform AI-powered analysis to identify patterns and bottlenecks, and generate actionable reports with specific ticket IDs for immediate action.

**Key Capabilities:**
- 🚀 High-speed ticket extraction with checkpointing (survives interruptions)
- 🤖 GenAI-powered clustering and root cause analysis
- 🧠 Deep AI content analysis (studies actual ticket conversations)
- 📊 SLA compliance tracking with stakeholder-specific dashboards
- 📋 Actionable Excel reports with clickable Freshdesk URLs
- 📄 Professional DOCX reports for leadership

---

## Features

### Extraction (`freshdesk_extractor_v2.py`)
- ✅ Incremental disk saves (each ticket saved immediately)
- ✅ Checkpoint/resume support (crash-safe)
- ✅ Rich terminal UI with live progress dashboard
- ✅ Weekly date chunking for optimal API usage
- ✅ Rate limit monitoring and auto-throttling
- ✅ Optional attachment downloads

### Analysis (`analyze_tickets.py`)
- ✅ Sentence embeddings (all-MiniLM-L6-v2)
- ✅ Automatic clustering (HDBSCAN)
- ✅ GenAI-powered cluster labeling
- ✅ Root cause analysis per category
- ✅ Strategic pattern detection
- ✅ Executive summary generation

### Deep AI Analysis (`deep_ai_analysis.py`) 🆕
- ✅ Studies actual ticket content (not just metadata)
- ✅ Identifies the 5 worst systemic issues
- ✅ Analyzes slowest tickets for blockers
- ✅ Investigates ignored (no-response) tickets
- ✅ Customer pain point analysis
- ✅ Quick wins + strategic recommendations

### SLA & Analytics (`generate_sla_report.py`) 🆕
- ✅ First Response Time (FRT) compliance
- ✅ Resolution Time compliance by priority
- ✅ Agent performance scorecards
- ✅ Customer health scores (0-100)
- ✅ Ticket aging analysis
- ✅ Monthly trend tracking

### Actionable Reports (`generate_actionable_report.py`)
- ✅ 12-sheet Excel workbook with ticket IDs
- ✅ Clickable Freshdesk URLs
- ✅ Priority-based categorization
- ✅ Customer deep-dives
- ✅ Duplicate company detection

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Freshdesk     │────▶│   Extractor      │────▶│   tickets.json  │
│   API           │     │   (v2.py)        │     │   tickets.csv   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Ollama         │              │
                        │   (qwen3:14b)    │◀─────────────┤
                        └────────┬─────────┘              │
                                 │                        │
                        ┌────────┴────────┐               │
                        ▼                 ▼               ▼
                ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                │   Cluster    │  │   Deep AI    │  │   SLA        │
                │   Analysis   │  │   Analysis   │  │   Analytics  │
                └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                       │                 │                 │
                       ▼                 ▼                 ▼
                ┌──────────────────────────────────────────────────┐
                │              Report Generation                    │
                │  • actionable_report.xlsx (ticket IDs)           │
                │  • sla_analytics_report.xlsx (SLA metrics)       │
                │  • deep_ai_analysis.md (AI insights)             │
                │  • FTEX_Deep_Analysis_Report.docx (executive)    │
                └──────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required
- **Python 3.9+**
- **Freshdesk API Key** (with ticket read permissions)
- **8GB+ RAM** (for embeddings)

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
pip install requests pandas rich numpy

# Analysis
pip install sentence-transformers scikit-learn hdbscan

# Reports
pip install openpyxl
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
FRESHDESK_DOMAIN=navtor
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
python3 run.py analyze   # Run AI analysis
python3 run.py report    # Generate reports
python3 run.py full      # Run entire pipeline
python3 run.py --help    # Show help
```

---

### `test` - Check API Connection

```bash
python3 run.py test
python3 run.py test --api-key YOUR_KEY    # Override .env
```

---

### `extract` - Download Tickets from Freshdesk

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

### `analyze` - Run Ticket Analysis

```bash
# Basic analysis (clustering only)
python3 run.py analyze

# With AI (requires Ollama running)
python3 run.py analyze --use-ollama

# Deep AI analysis (generates slides + action items)
python3 run.py analyze --deep

# Both AI features
python3 run.py analyze --use-ollama --deep

# Custom input file
python3 run.py analyze --input output/tickets.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input JSON file | output/tickets.json |
| `--use-ollama` | Enable GenAI cluster labeling | False |
| `--deep` | Run deep AI content analysis | False |

**Deep Analysis Outputs:**
- `deep_ai_analysis.md` - Full AI analysis report
- `presentation_slides.md` - Sli.dev format slides
- `action_items.md` - Prioritized action checklist

---

### `report` - Generate Reports

```bash
# Generate all reports
python3 run.py report

# Specific report types
python3 run.py report --type actionable    # Excel with ticket IDs
python3 run.py report --type sla           # SLA analytics
python3 run.py report --type all           # Both (default)

# Custom output directory
python3 run.py report --output-dir my_reports/
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input JSON file | output/tickets.json |
| `--type`, `-t` | Report type: `actionable`, `sla`, `all` | all |
| `--output-dir`, `-o` | Output directory | reports/ |

---

### `full` - Complete Pipeline

Runs: extract → analyze → report in sequence.

```bash
# Basic full run
python3 run.py full

# With options
python3 run.py full --days 90                     # Last 90 days
python3 run.py full --days 180 --no-attachments   # Skip attachments
python3 run.py full --skip-extract                # Use existing data
```

| Flag | Description | Default |
|------|-------------|---------|
| `--days`, `-d` | Days of history | 180 |
| `--group-id`, `-g` | Filter by group ID | From .env |
| `--no-attachments` | Skip downloading attachments | False |
| `--skip-extract` | Skip extraction, use existing data | False |
| `--api-key`, `-k` | Override API key | From .env |

---

### Common Workflows

```bash
# 🔧 First time setup
python3 run.py test

# 📅 Daily analysis (existing data)
python3 run.py analyze --deep

# 📆 Weekly full refresh
python3 run.py full --days 30

# 📊 Quick report regeneration
python3 run.py report --type sla

# 🔄 Resume interrupted extraction
python3 run.py extract --resume

# ⚡ Fast extraction (no attachments)
python3 run.py extract --days 180 --no-attachments
```

---

### Output Files by Command

| Command | Output Files |
|---------|--------------|
| `extract` | `output/tickets.json`, `output/tickets/` |
| `analyze` | `reports/analysis_report.md`, `reports/analysis_report_data.json` |
| `analyze --deep` | `reports/deep_ai_analysis.md`, `reports/presentation_slides.md`, `reports/action_items.md` |
| `report --type actionable` | `reports/actionable_report.xlsx` |
| `report --type sla` | `reports/sla_analytics_report.xlsx` |

---

## Usage (Direct Script Execution)

Cluster-based analysis with optional LLM labeling:

```bash
# Without Ollama (keyword-based labels)
python3 analyze_tickets.py --input output/tickets.json

# With Ollama (GenAI labels + insights)
python3 analyze_tickets.py --input output/tickets.json --use-ollama
```

**Output:** `analysis_report.md` + `analysis_report_data.json`

---

### 4. Run Deep AI Analysis

Let the AI study actual ticket content to find systemic issues:

```bash
python3 deep_ai_analysis.py --input output/tickets.json
```

**What the AI does:**
1. 📚 Studies 100 sampled tickets across all categories
2. ⏰ Analyzes tickets with longest resolution times
3. 🔇 Investigates ignored (no-response) tickets
4. 🏢 Deep-dives into top customer pain points
5. 🧠 Synthesizes findings into "5 Worst Issues"
6. 📝 Generates executive summary

**Output:** `deep_ai_analysis.md` with:
- Executive Summary (MD-ready)
- The 5 Worst Issues (ranked by severity)
- Root Cause Analysis
- Customer Impact Assessment
- Quick Wins (2-week fixes)
- Strategic Recommendations

**Runtime:** ~10-15 minutes (multiple LLM calls)

---

### 5. Generate Reports

#### Actionable Report (Ticket IDs)

```bash
python3 generate_actionable_report.py --input output/tickets.json
```

**Output:** `actionable_report.xlsx` with 12 sheets:

| Sheet | Contents |
|-------|----------|
| 0_SUMMARY | Overview of all categories |
| 1_No_Response_Zombies | Tickets with 0 conversations |
| 2_Long_Resolution_500h+ | Tickets taking >20 days |
| 3_Open_Pending_Tickets | Currently open/pending |
| 4_License_Update_Automate | Automation candidates |
| 5_Onboarding_GoLive | Onboarding tickets |
| 6_Top_Companies | Company breakdown |
| 7_Stolt_Tankers_DeepDive | Top customer deep-dive |
| 8_NYK_DeepDive | Second customer deep-dive |
| 9_Duplicate_Companies | Data cleanup needed |
| 10_Overdue_Tagged | Already flagged overdue |
| 11_Weekly_Planner | Recurring status tickets |

#### SLA Analytics Report

```bash
python3 generate_sla_report.py --input output/tickets.json
```

**Output:** `sla_analytics_report.xlsx` with stakeholder-specific sheets:

| Audience | Sheets | Key Metrics |
|----------|--------|-------------|
| **MD** | Executive Dashboard, Monthly Trends, Customer Health | SLA compliance %, health scores, trends |
| **Manager** | SLA by Priority, Time Patterns, Source Analysis | Priority breakdown, peak hours, channels |
| **TL** | Agent Performance, Ticket Aging, SLA Breaches | Per-agent stats, backlog aging, breach details |

**SLA Targets (Customizable in script):**
```python
# First Response Time (hours)
Urgent: 1, High: 4, Medium: 8, Low: 24

# Resolution Time (hours)
Urgent: 4, High: 24, Medium: 72, Low: 168
```

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
├── reports/
│   ├── actionable_report.xlsx    # Ticket IDs by category
│   ├── sla_analytics_report.xlsx # SLA metrics
│   ├── analysis_report.md        # Cluster analysis
│   ├── deep_ai_analysis.md       # AI insights
│   └── FTEX_Deep_Analysis_Report.docx # Executive report
│
└── docs/
    ├── ACTION_ITEMS.md           # Execution checklist
    └── presentation.md           # Sli.dev slides
```

---

## Project Structure

```
FTEX/
├── src/
│   ├── extraction/
│   │   ├── freshdesk_extractor_v2.py    # Main extractor
│   │   └── test_freshdesk_api.py        # API tester
│   │
│   ├── analysis/
│   │   ├── analyze_tickets.py           # Cluster analysis
│   │   └── deep_ai_analysis.py          # Deep AI analysis
│   │
│   └── reports/
│       ├── generate_actionable_report.py
│       └── generate_sla_report.py
│
├── output/                    # Extracted data (gitignored)
│   ├── tickets.json
│   ├── tickets/
│   └── checkpoints/
│
├── reports/                   # Generated reports (gitignored)
├── docs/                      # Documentation
│
├── run.py                     # CLI entry point
├── config.py                  # Shared configuration module
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

Create a `.env` file in the project root (copy from `.env.example`):

```env
# Required
FRESHDESK_API_KEY=your_api_key_here
FRESHDESK_DOMAIN=yourcompany

# Optional
FRESHDESK_GROUP_ID=48000615489
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b

# Custom stop words for analysis (comma-separated)
FTEX_STOP_WORDS=yourproduct,yourcompany
```

| Variable | Description | Required |
|----------|-------------|----------|
| `FRESHDESK_API_KEY` | Your Freshdesk API key | ✅ Yes |
| `FRESHDESK_DOMAIN` | Freshdesk subdomain (e.g., `navtor`) | ✅ Yes |
| `FRESHDESK_GROUP_ID` | Default group ID to filter | No |
| `OLLAMA_URL` | Ollama server URL | No |
| `OLLAMA_MODEL` | Preferred LLM model | No |
| `FTEX_STOP_WORDS` | Words to ignore in analysis | No |

### `.env` vs `config.py`

| File | Purpose | Commit to Git? |
|------|---------|----------------|
| `.env` | Your secrets (API keys, domain) | ❌ Never |
| `config.py` | Code that reads `.env` + defaults | ✅ Yes |
| `.env.example` | Template for other users | ✅ Yes |

### SLA Configuration

Edit `generate_sla_report.py` or create `config/sla_config.json`:

```json
{
  "first_response": {
    "Urgent": 1,
    "High": 4,
    "Medium": 8,
    "Low": 24
  },
  "resolution": {
    "Urgent": 4,
    "High": 24,
    "Medium": 72,
    "Low": 168
  }
}
```

### .gitignore

```gitignore
# Environment
.env
*.env
venv/

# Output data
output/
reports/*.xlsx
reports/*.docx

# Python
__pycache__/
*.pyc

# Temp
~$*
.DS_Store
```

---

## Analysis Parameters

### Extraction Parameters

| Parameter | Value | Modifiable |
|-----------|-------|------------|
| Freshdesk Domain | yourcompany.freshdesk.com | .env |
| Group ID | 48000615489 | --group-id |
| Date Range | 180 days | --days |
| Attachments | Excluded | --no-attachments |

### Analysis Parameters

| Parameter | Value | Modifiable |
|-----------|-------|------------|
| Embedding Model | all-MiniLM-L6-v2 | Code |
| Clustering | HDBSCAN (auto clusters) | Code |
| LLM Model | qwen3:14b | OLLAMA_MODEL |
| LLM Temperature | 0.2-0.4 | Code |

### Report Filters

| Filter | Criteria | Modifiable |
|--------|----------|------------|
| Zombie Tickets | conversations == 0 | Code |
| Long Resolution | >500 hours | Code |
| License/Update | Subject keywords | Code |
| Onboarding | Subject keywords | Code |

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
caffeinate -i python3 freshdesk_extractor_v2.py ...
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
python3 analyze_tickets.py --input output/tickets.json
```

**HDBSCAN Not Installed**
```bash
pip install hdbscan
# Mac M1/M2/M3/M4:
pip install hdbscan --no-cache-dir
```

---

## Roadmap

### Phase 1 ✅ (Complete)
- [x] Ticket extraction with checkpointing
- [x] GenAI-powered cluster analysis
- [x] Deep AI content analysis
- [x] SLA compliance tracking
- [x] Actionable Excel reports
- [x] Professional DOCX reports
- [x] CLI entry point

### Phase 2 🚧 (In Progress)
- [ ] Web dashboard for real-time monitoring
- [ ] Scheduled extraction (cron/Airflow)
- [ ] Slack/Teams integration for alerts
- [ ] Customer health scoring dashboard

### Phase 3 📋 (Planned)
- [ ] Multi-tenant support
- [ ] Historical trend analysis
- [ ] Predictive ticket routing
- [ ] Self-service portal integration
- [ ] RAG chatbot for support agents

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
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM inference
- [Rich](https://rich.readthedocs.io/) for beautiful terminal UI
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering

---

<p align="center">
  <b>Built with ❤️ for Support Operations Teams Everywhere</b>
</p>