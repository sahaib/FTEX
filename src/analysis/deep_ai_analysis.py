#!/usr/bin/env python3
"""
FTEX Deep AI Analysis v2.0
===========================
AI-powered comprehensive ticket analysis that studies actual ticket content
to identify systemic issues, worst problems, and actionable insights.

Features:
- 🧠 Deep content analysis (not just metadata)
- 📊 Identifies top 5 worst systemic issues
- 🎯 Generates actionable recommendations with ticket IDs
- 📽️ Auto-generates sli.dev presentation slides
- 📋 Creates prioritized action items

Outputs:
- deep_ai_analysis.md      - Full analysis report
- presentation_slides.md   - Sli.dev format slides
- action_items.md          - Prioritized action checklist
- analysis_data.json       - Raw data for further processing

Usage:
    python3 deep_ai_analysis.py --input output/tickets.json
    python3 deep_ai_analysis.py --input output/tickets.json --skip-slides
    python3 deep_ai_analysis.py --input output/tickets.json --output-dir reports/

License: MIT
"""

import json
import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import re
import html

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

try:
    import requests
    import numpy as np
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.markdown import Markdown
    from rich import box
    console = Console()
except ImportError:
    print("❌ Install required packages: pip3 install requests numpy rich")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """Configurable settings for the analysis"""
    
    # Ollama settings
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    PREFERRED_MODELS = ['qwen3:14b', 'qwen3:8b', 'qwen2.5:14b', 'llama3.1:8b', 'mistral:7b']
    
    # Analysis settings
    SAMPLE_SIZE = 100           # Tickets to sample for batch analysis
    BATCH_SIZE = 20             # Tickets per batch
    TOP_CUSTOMERS = 3           # Number of top customers to analyze
    WORST_TICKETS = 15          # Number of worst tickets to analyze
    NO_RESPONSE_SAMPLE = 20     # Number of no-response tickets to analyze
    
    # Thresholds
    ZOMBIE_THRESHOLD_DAYS = 180     # Days after which zombie tickets should be closed
    LONG_RESOLUTION_HOURS = 500     # Hours threshold for "long resolution"
    HIGH_PRIORITY_BREACH_HOURS = 48 # Hours after which high priority is breached
    
    # LLM settings
    TEMPERATURE_ANALYSIS = 0.3
    TEMPERATURE_SYNTHESIS = 0.4
    TEMPERATURE_CREATIVE = 0.5      # For slide generation
    MAX_TOKENS_STANDARD = 4000
    MAX_TOKENS_SYNTHESIS = 6000
    MAX_TOKENS_SLIDES = 8000


# =============================================================================
# OLLAMA CLIENT
# =============================================================================
class OllamaClient:
    """Production-ready Ollama client with auto model detection"""
    
    def __init__(self, base_url: str = Config.OLLAMA_URL):
        self.base_url = base_url
        self.model = None
        self.available = False
        
    def check_availability(self) -> bool:
        """Check if Ollama is running and find best available model"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                # Find preferred model
                for preferred in Config.PREFERRED_MODELS:
                    for available in models:
                        if preferred in available:
                            self.model = available
                            self.available = True
                            return True
                # Fallback to any available model
                if models:
                    self.model = models[0]
                    self.available = True
                    return True
        except requests.exceptions.ConnectionError:
            console.print("[yellow]⚠️ Ollama not running. Start with: ollama serve[/yellow]")
        except Exception as e:
            console.print(f"[red]Ollama error: {e}[/red]")
        return False
    
    def generate(self, prompt: str, temperature: float = 0.4, max_tokens: int = 4000) -> Optional[str]:
        """Generate completion from Ollama"""
        if not self.available:
            return None
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': temperature,
                        'num_predict': max_tokens,
                        'top_p': 0.9,
                    }
                },
                timeout=300
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except requests.exceptions.Timeout:
            console.print("[yellow]⚠️ LLM request timed out[/yellow]")
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
        return None


# =============================================================================
# ANALYSIS PROMPTS
# =============================================================================
class Prompts:
    """All prompts used for AI analysis"""
    
    @staticmethod
    def study_ticket_batch(tickets_text: str, batch_num: int, total_batches: int) -> str:
        return f"""You are a senior support operations analyst studying support tickets for a software company.

BATCH {batch_num} of {total_batches}

TICKET DATA:
{tickets_text}

Study these tickets carefully. Identify:

1. **RECURRING PROBLEMS** - Issues that appear multiple times
2. **FRUSTRATED CUSTOMERS** - Signs of customer frustration or escalation
3. **PROCESS FAILURES** - Where the support process itself failed
4. **PRODUCT GAPS** - Features missing or not working properly
5. **AUTOMATION OPPORTUNITIES** - Repetitive tasks that could be automated

Be specific. Include ticket IDs as evidence. Format as:

RECURRING_PROBLEMS:
- [Problem]: [Count] occurrences - [Ticket IDs]

FRUSTRATED_CUSTOMERS:
- [Company/Ticket ID]: [What frustrated them]

PROCESS_FAILURES:
- [Failure]: [Evidence]

PRODUCT_GAPS:
- [Gap]: [Evidence]

AUTOMATION_OPPORTUNITIES:
- [Task]: [Why automatable] - [Sample ticket IDs]"""

    @staticmethod
    def analyze_worst_tickets(tickets_text: str) -> str:
        return f"""Analyze these tickets with the LONGEST resolution times:

{tickets_text}

Identify:
1. **COMMON PATTERNS** - What do slow tickets have in common?
2. **BLOCKERS** - What caused the delays? (external dependencies, unclear requirements, etc.)
3. **PROCESS GAPS** - Where did the process break down?
4. **PREVENTABLE** - Which delays could have been avoided?

Include specific ticket IDs. Be actionable."""

    @staticmethod
    def analyze_no_response_tickets(tickets_text: str) -> str:
        return f"""These tickets have ZERO responses (completely ignored):

{tickets_text}

Analyze:
1. **WHY ignored?** (routing issue? unclear request? low priority?)
2. **PATTERNS** - Common characteristics?
3. **CUSTOMER IMPACT** - What happened to these customers?
4. **PREVENTION** - How to prevent this?

Include ticket IDs."""

    @staticmethod
    def analyze_customer(company_name: str, tickets_text: str, ticket_count: int) -> str:
        return f"""Analyze support tickets from {company_name} ({ticket_count} total tickets):

{tickets_text}

In 4-5 sentences:
1. What is their PRIMARY pain point?
2. Is this a training issue, product issue, or process issue?
3. What specific action would help them most?
4. Risk level: High/Medium/Low and why?

Be concise and actionable."""

    @staticmethod
    def synthesize_findings(all_findings: str, stats: Dict) -> str:
        return f"""You are preparing a report for the Managing Director based on {stats['total_tickets']:,} support tickets.

FINDINGS FROM ANALYSIS:
{all_findings}

STATISTICS:
- Total Tickets: {stats['total_tickets']:,}
- Average Resolution: {stats['avg_resolution_days']:.1f} days
- Median Resolution: {stats['median_resolution_days']:.1f} days
- No Response (Zombies): {stats['no_response']} ({stats['no_response_pct']:.1f}%)
- Open Backlog: {stats['backlog']}
- Top Customer: {stats['top_customer']} ({stats['top_customer_tickets']} tickets)

Synthesize into a comprehensive report:

## THE 5 WORST ISSUES

For each issue:
### [Rank]. [Issue Name]
**Severity:** Critical/High/Medium
**Evidence:** [Specific ticket IDs and numbers]
**Customer Impact:** [How this hurts customers]
**Root Cause:** [Why this happens]
**Recommendation:** [Specific fix with timeline]

## ROOT CAUSE ANALYSIS
[Deep analysis: Is it people, process, product, or technology?]

## QUICK WINS (This Week)
1. [Action]: [Expected impact] - [Ticket IDs to action]
2. ...

## THIS MONTH PRIORITIES
1. [Priority]: [Justification]
2. ...

## THIS QUARTER GOALS
1. [Goal]: [Success metric]
2. ...

## METRICS TO TRACK
[What should leadership monitor?]

Be direct, specific, and actionable. Include ticket IDs where relevant."""

    @staticmethod
    def generate_executive_summary(synthesis: str, stats: Dict) -> str:
        return f"""Based on this analysis of {stats['total_tickets']:,} tickets:

{synthesis[:3000]}

Write a 3-paragraph EXECUTIVE SUMMARY (under 300 words):

Paragraph 1: Current state of support operations (good and bad)
Paragraph 2: Biggest risks to customer satisfaction and business
Paragraph 3: Top 3 priorities for the next quarter

Be direct. No fluff. This goes to the MD."""

    @staticmethod
    def generate_slidev_presentation(synthesis: str, stats: Dict, exec_summary: str, 
                                      top_customers: List[Tuple[str, int]], 
                                      ticket_samples: Dict[str, List[int]]) -> str:
        return f"""Create a sli.dev presentation (Markdown format) based on this analysis.

EXECUTIVE SUMMARY:
{exec_summary}

KEY STATISTICS:
- Total Tickets: {stats['total_tickets']:,}
- Avg Resolution: {stats['avg_resolution_days']:.1f} days
- Median Resolution: {stats['median_resolution_days']:.1f} days
- No Response: {stats['no_response']} ({stats['no_response_pct']:.1f}%)
- Open Backlog: {stats['backlog']}

TOP CUSTOMERS:
{chr(10).join([f"- {name}: {count} tickets" for name, count in top_customers])}

SAMPLE TICKET IDs:
- Zombies: {', '.join([f'#{id}' for id in ticket_samples.get('zombies', [])[:5]])}
- Long Resolution: {', '.join([f'#{id}' for id in ticket_samples.get('long_resolution', [])[:5]])}
- High Priority: {', '.join([f'#{id}' for id in ticket_samples.get('high_priority', [])[:5]])}

FULL ANALYSIS:
{synthesis[:4000]}

Create a sli.dev presentation with these slides:
1. Title slide with key stat
2. Executive Summary (3 bullets max)
3. Key Statistics dashboard
4. The 5 Worst Issues (one slide each with severity, evidence, fix)
5. Customer Concentration Risk
6. Root Cause Analysis
7. Quick Wins (This Week)
8. This Month Priorities
9. This Quarter Goals
10. Metrics to Track
11. Appendix: Ticket ID References

FORMAT RULES:
- Use --- to separate slides
- Use proper sli.dev frontmatter for first slide
- Use grid layouts: <div class="grid grid-cols-2 gap-8">
- Use color classes: text-red-500, text-green-500, bg-gray-800
- Keep bullet points short (max 10 words each)
- Include actual ticket IDs from the data
- Use tables for metrics
- Use emojis sparingly for visual hierarchy

Start with:
---
theme: default
class: text-center
---

Generate the complete presentation now:"""

    @staticmethod
    def generate_action_items(synthesis: str, stats: Dict, ticket_samples: Dict[str, List[int]]) -> str:
        return f"""Based on this analysis, create a prioritized ACTION ITEMS checklist.

ANALYSIS:
{synthesis[:3000]}

TICKET SAMPLES:
- Zombies ({stats['no_response']} total): {ticket_samples.get('zombies', [])[:10]}
- Long Resolution: {ticket_samples.get('long_resolution', [])[:10]}
- Open Backlog ({stats['backlog']} total): {ticket_samples.get('open_pending', [])[:10]}

Create action items in this EXACT format:

## 🔴 PRIORITY 0: This Week

### Action 1: [Name]
- **Owner:** [Role]
- **Tickets:** [Count] - IDs: #XXX, #XXX, #XXX
- **Steps:**
  1. [Step 1]
  2. [Step 2]
- **Success Metric:** [How to measure]

### Action 2: ...

## 🟠 PRIORITY 1: This Month

### Action 3: ...

## 🟡 PRIORITY 2: This Quarter

### Action 4: ...

Include 6-8 total actions. Be specific with ticket IDs."""


# =============================================================================
# TICKET DATA EXTRACTOR
# =============================================================================
class TicketDataExtractor:
    """Extract structured data from tickets for analysis"""
    
    def __init__(self, tickets: List[Dict]):
        self.tickets = tickets
        self._cache = {}
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text"""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = html.unescape(text)
        return text.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate key statistics"""
        if 'stats' in self._cache:
            return self._cache['stats']
        
        resolution_times = [t.get('resolution_time_hours', 0) 
                          for t in self.tickets if t.get('resolution_time_hours')]
        no_response = len([t for t in self.tickets if len(t.get('conversations', [])) == 0])
        backlog = len([t for t in self.tickets if t.get('status_name') in ['Open', 'Pending']])
        
        company_counts = Counter()
        for t in self.tickets:
            company = t.get('company', {})
            if company and company.get('name'):
                company_counts[company['name']] += 1
        
        top_customer = company_counts.most_common(1)[0] if company_counts else ('Unknown', 0)
        
        stats = {
            'total_tickets': len(self.tickets),
            'avg_resolution_days': np.mean(resolution_times) / 24 if resolution_times else 0,
            'median_resolution_days': np.median(resolution_times) / 24 if resolution_times else 0,
            'p90_resolution_days': np.percentile(resolution_times, 90) / 24 if resolution_times else 0,
            'no_response': no_response,
            'no_response_pct': no_response / len(self.tickets) * 100 if self.tickets else 0,
            'backlog': backlog,
            'top_customer': top_customer[0],
            'top_customer_tickets': top_customer[1],
            'company_counts': company_counts,
        }
        self._cache['stats'] = stats
        return stats
    
    def get_ticket_samples(self) -> Dict[str, List[int]]:
        """Get sample ticket IDs for each category"""
        if 'samples' in self._cache:
            return self._cache['samples']
        
        samples = {
            'zombies': [],
            'long_resolution': [],
            'open_pending': [],
            'high_priority': [],
            'license': [],
            'onboarding': [],
        }
        
        for t in self.tickets:
            tid = t.get('id')
            subject = (t.get('subject') or '').lower()
            status = t.get('status_name', '')
            priority = t.get('priority_name', '')
            resolution = t.get('resolution_time_hours', 0)
            convos = t.get('conversations', [])
            
            if len(convos) == 0:
                samples['zombies'].append(tid)
            if resolution and resolution > Config.LONG_RESOLUTION_HOURS:
                samples['long_resolution'].append(tid)
            if status in ['Open', 'Pending']:
                samples['open_pending'].append(tid)
            if priority in ['Urgent', 'High']:
                samples['high_priority'].append(tid)
            if any(kw in subject for kw in ['license', 'activation', 'key', 'renewal']):
                samples['license'].append(tid)
            if any(kw in subject for kw in ['go-live', 'onboarding', 'golive', 'setup']):
                samples['onboarding'].append(tid)
        
        # Sort long_resolution by actual time
        long_res_tickets = [(t.get('id'), t.get('resolution_time_hours', 0)) 
                           for t in self.tickets if t.get('resolution_time_hours', 0) > Config.LONG_RESOLUTION_HOURS]
        long_res_tickets.sort(key=lambda x: x[1], reverse=True)
        samples['long_resolution'] = [t[0] for t in long_res_tickets]
        
        self._cache['samples'] = samples
        return samples
    
    def get_top_customers(self, n: int = 5) -> List[Tuple[str, int, List[Dict]]]:
        """Get top N customers by ticket count with sample tickets"""
        stats = self.get_stats()
        company_counts = stats['company_counts']
        
        company_tickets = defaultdict(list)
        for t in self.tickets:
            company = t.get('company', {})
            if company and company.get('name'):
                company_tickets[company['name']].append(t)
        
        result = []
        for name, count in company_counts.most_common(n):
            result.append((name, count, company_tickets[name][:10]))
        return result
    
    def format_ticket(self, ticket: Dict, include_conversations: bool = True) -> str:
        """Format a ticket for LLM analysis"""
        parts = []
        parts.append(f"[Ticket #{ticket.get('id')}]")
        parts.append(f"Subject: {ticket.get('subject', 'N/A')}")
        parts.append(f"Status: {ticket.get('status_name', 'Unknown')} | Priority: {ticket.get('priority_name', 'Unknown')}")
        
        company = ticket.get('company', {})
        if company and company.get('name'):
            parts.append(f"Company: {company['name']}")
        
        resolution = ticket.get('resolution_time_hours')
        if resolution:
            parts.append(f"Resolution: {resolution:.0f}h ({resolution/24:.1f} days)")
        
        desc = self.clean_html(ticket.get('description', ''))
        if desc:
            parts.append(f"Description: {desc[:400]}...")
        
        if include_conversations:
            convos = ticket.get('conversations', [])
            if convos:
                parts.append(f"Conversations: {len(convos)}")
                for conv in convos[:2]:
                    direction = "Customer" if conv.get('incoming') else "Agent"
                    body = self.clean_html(conv.get('body_text') or conv.get('body', ''))[:250]
                    if body:
                        parts.append(f"  [{direction}]: {body}...")
        
        parts.append("")
        return "\n".join(parts)
    
    def sample_diverse_tickets(self, n: int = 100) -> List[Dict]:
        """Sample diverse tickets across categories"""
        sampled = []
        seen = set()
        
        by_status = defaultdict(list)
        by_priority = defaultdict(list)
        
        for t in self.tickets:
            by_status[t.get('status_name', 'Unknown')].append(t)
            by_priority[t.get('priority_name', 'Unknown')].append(t)
        
        # Sample from each status
        for status, tix in by_status.items():
            for t in tix[:n // 8]:
                if t.get('id') not in seen:
                    seen.add(t.get('id'))
                    sampled.append(t)
        
        # Add high priority
        for t in by_priority.get('Urgent', [])[:10]:
            if t.get('id') not in seen:
                seen.add(t.get('id'))
                sampled.append(t)
        
        for t in by_priority.get('High', [])[:10]:
            if t.get('id') not in seen:
                seen.add(t.get('id'))
                sampled.append(t)
        
        # Add longest resolution
        by_resolution = sorted(
            [t for t in self.tickets if t.get('resolution_time_hours')],
            key=lambda x: x.get('resolution_time_hours', 0),
            reverse=True
        )
        for t in by_resolution[:15]:
            if t.get('id') not in seen:
                seen.add(t.get('id'))
                sampled.append(t)
        
        return sampled[:n]


# =============================================================================
# DEEP ANALYZER
# =============================================================================
class DeepAnalyzer:
    """Main analysis engine"""
    
    def __init__(self, tickets: List[Dict]):
        self.extractor = TicketDataExtractor(tickets)
        self.ollama = OllamaClient()
        self.stats = {}
        self.findings = []
        self.synthesis = None
        self.exec_summary = None
        self.slides = None
        self.action_items = None
    
    def run_analysis(self) -> bool:
        """Run the complete analysis pipeline"""
        
        # Check Ollama
        console.print("\n[cyan]🔍 Checking AI availability...[/cyan]")
        if not self.ollama.check_availability():
            console.print("[red]❌ Ollama not available. Start with: ollama serve[/red]")
            return False
        console.print(f"[green]✅ Connected to {self.ollama.model}[/green]\n")
        
        # Get stats
        self.stats = self.extractor.get_stats()
        ticket_samples = self.extractor.get_ticket_samples()
        top_customers = self.extractor.get_top_customers(Config.TOP_CUSTOMERS)
        
        all_findings = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            # Phase 1: Batch Analysis
            task1 = progress.add_task("📚 Studying ticket batches...", total=5)
            sampled = self.extractor.sample_diverse_tickets(Config.SAMPLE_SIZE)
            batches = [sampled[i:i+Config.BATCH_SIZE] for i in range(0, len(sampled), Config.BATCH_SIZE)]
            
            for i, batch in enumerate(batches[:5]):
                tickets_text = "\n".join([self.extractor.format_ticket(t) for t in batch])
                prompt = Prompts.study_ticket_batch(tickets_text, i+1, 5)
                result = self.ollama.generate(prompt, Config.TEMPERATURE_ANALYSIS)
                if result:
                    all_findings.append(f"### Batch {i+1} Findings:\n{result}\n")
                progress.update(task1, advance=1)
            
            # Phase 2: Worst Tickets
            task2 = progress.add_task("⏰ Analyzing slowest tickets...", total=1)
            worst_ids = ticket_samples.get('long_resolution', [])[:Config.WORST_TICKETS]
            worst_tickets = [t for t in self.extractor.tickets if t.get('id') in worst_ids]
            if worst_tickets:
                worst_text = "\n".join([self.extractor.format_ticket(t) for t in worst_tickets])
                prompt = Prompts.analyze_worst_tickets(worst_text)
                result = self.ollama.generate(prompt, Config.TEMPERATURE_ANALYSIS)
                if result:
                    all_findings.append(f"### Slowest Tickets Analysis:\n{result}\n")
            progress.update(task2, advance=1)
            
            # Phase 3: No-Response Tickets
            task3 = progress.add_task("🔇 Analyzing ignored tickets...", total=1)
            zombie_ids = ticket_samples.get('zombies', [])[:Config.NO_RESPONSE_SAMPLE]
            zombies = [t for t in self.extractor.tickets if t.get('id') in zombie_ids]
            if zombies:
                zombie_text = "\n".join([self.extractor.format_ticket(t, False) for t in zombies])
                prompt = Prompts.analyze_no_response_tickets(zombie_text)
                result = self.ollama.generate(prompt, Config.TEMPERATURE_ANALYSIS)
                if result:
                    all_findings.append(f"### No-Response Tickets Analysis:\n{result}\n")
            progress.update(task3, advance=1)
            
            # Phase 4: Top Customers
            task4 = progress.add_task("🏢 Analyzing top customers...", total=Config.TOP_CUSTOMERS)
            for company_name, count, sample_tix in top_customers:
                tix_text = "\n".join([self.extractor.format_ticket(t) for t in sample_tix])
                prompt = Prompts.analyze_customer(company_name, tix_text, count)
                result = self.ollama.generate(prompt, Config.TEMPERATURE_ANALYSIS)
                if result:
                    all_findings.append(f"### Customer: {company_name} ({count} tickets):\n{result}\n")
                progress.update(task4, advance=1)
            
            # Phase 5: Synthesize
            task5 = progress.add_task("🧠 Synthesizing insights...", total=1)
            combined = "\n\n".join(all_findings)
            prompt = Prompts.synthesize_findings(combined, self.stats)
            self.synthesis = self.ollama.generate(prompt, Config.TEMPERATURE_SYNTHESIS, Config.MAX_TOKENS_SYNTHESIS)
            progress.update(task5, advance=1)
            
            # Phase 6: Executive Summary
            task6 = progress.add_task("📝 Writing executive summary...", total=1)
            if self.synthesis:
                prompt = Prompts.generate_executive_summary(self.synthesis, self.stats)
                self.exec_summary = self.ollama.generate(prompt, Config.TEMPERATURE_ANALYSIS, 1000)
            progress.update(task6, advance=1)
            
            # Phase 7: Generate Slides
            task7 = progress.add_task("📽️ Generating presentation...", total=1)
            if self.synthesis and self.exec_summary:
                prompt = Prompts.generate_slidev_presentation(
                    self.synthesis, 
                    self.stats, 
                    self.exec_summary,
                    [(name, count) for name, count, _ in top_customers],
                    ticket_samples
                )
                self.slides = self.ollama.generate(prompt, Config.TEMPERATURE_CREATIVE, Config.MAX_TOKENS_SLIDES)
            progress.update(task7, advance=1)
            
            # Phase 8: Generate Action Items
            task8 = progress.add_task("🎯 Creating action items...", total=1)
            if self.synthesis:
                prompt = Prompts.generate_action_items(self.synthesis, self.stats, ticket_samples)
                self.action_items = self.ollama.generate(prompt, Config.TEMPERATURE_ANALYSIS, 3000)
            progress.update(task8, advance=1)
        
        self.findings = all_findings
        return True
    
    def save_reports(self, output_dir: str = ".") -> Dict[str, str]:
        """Save all generated reports"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # 1. Main Analysis Report
        report = self._build_analysis_report(timestamp)
        analysis_path = output_path / "deep_ai_analysis.md"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(report)
        outputs['analysis'] = str(analysis_path)
        console.print(f"[green]✅ Analysis report: {analysis_path}[/green]")
        
        # 2. Sli.dev Slides
        if self.slides:
            slides_path = output_path / "presentation_slides.md"
            with open(slides_path, 'w', encoding='utf-8') as f:
                f.write(self.slides)
            outputs['slides'] = str(slides_path)
            console.print(f"[green]✅ Presentation slides: {slides_path}[/green]")
        
        # 3. Action Items
        if self.action_items:
            actions_path = output_path / "action_items.md"
            action_report = self._build_action_items_report(timestamp)
            with open(actions_path, 'w', encoding='utf-8') as f:
                f.write(action_report)
            outputs['actions'] = str(actions_path)
            console.print(f"[green]✅ Action items: {actions_path}[/green]")
        
        # 4. JSON Data
        data = {
            'generated': timestamp,
            'model': self.ollama.model,
            'stats': {k: v for k, v in self.stats.items() if k != 'company_counts'},
            'ticket_samples': self.extractor.get_ticket_samples(),
            'top_customers': [(name, count) for name, count, _ in self.extractor.get_top_customers(10)],
        }
        data_path = output_path / "analysis_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        outputs['data'] = str(data_path)
        console.print(f"[green]✅ Analysis data: {data_path}[/green]")
        
        return outputs
    
    def _build_analysis_report(self, timestamp: str) -> str:
        """Build the main analysis markdown report"""
        lines = [
            "# 🧠 FTEX Deep AI Analysis Report",
            "",
            f"**Generated:** {timestamp}",
            f"**Tickets Analyzed:** {self.stats['total_tickets']:,}",
            f"**AI Model:** {self.ollama.model}",
            "",
            "---",
            "## 📋 Executive Summary",
            "",
            self.exec_summary or "*Executive summary generation failed*",
            "",
            "---",
            "## 📊 Key Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tickets | {self.stats['total_tickets']:,} |",
            f"| Avg Resolution | {self.stats['avg_resolution_days']:.1f} days |",
            f"| Median Resolution | {self.stats['median_resolution_days']:.1f} days |",
            f"| 90th Percentile | {self.stats['p90_resolution_days']:.1f} days |",
            f"| No Response | {self.stats['no_response']} ({self.stats['no_response_pct']:.1f}%) |",
            f"| Open Backlog | {self.stats['backlog']} |",
            f"| Top Customer | {self.stats['top_customer']} ({self.stats['top_customer_tickets']} tickets) |",
            "",
            "---",
            "",
            self.synthesis or "*Synthesis generation failed*",
            "",
            "---",
            "## 📎 Appendix: Raw AI Findings",
            "",
            "<details>",
            "<summary>Click to expand raw analysis findings</summary>",
            "",
        ]
        
        for finding in self.findings:
            lines.append(finding)
        
        lines.extend([
            "</details>",
            "",
            "---",
            f"*Report generated by FTEX Deep AI Analysis v2.0*",
        ])
        
        return "\n".join(lines)
    
    def _build_action_items_report(self, timestamp: str) -> str:
        """Build the action items report"""
        samples = self.extractor.get_ticket_samples()
        
        header = f"""# 🎯 FTEX Action Items

**Generated:** {timestamp}
**Source:** Deep AI Analysis of {self.stats['total_tickets']:,} tickets

---

## 📋 Quick Reference

| Category | Count | Sample IDs |
|----------|-------|------------|
| Zombies (No Response) | {self.stats['no_response']} | {', '.join([f'#{id}' for id in samples['zombies'][:5]])} |
| Long Resolution | {len(samples['long_resolution'])} | {', '.join([f'#{id}' for id in samples['long_resolution'][:5]])} |
| Open Backlog | {self.stats['backlog']} | {', '.join([f'#{id}' for id in samples['open_pending'][:5]])} |
| License Related | {len(samples['license'])} | {', '.join([f'#{id}' for id in samples['license'][:5]])} |
| Onboarding | {len(samples['onboarding'])} | {', '.join([f'#{id}' for id in samples['onboarding'][:5]])} |

---

"""
        return header + (self.action_items or "*Action items generation failed*")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='FTEX Deep AI Analysis - Comprehensive ticket analysis with AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 deep_ai_analysis.py --input output/tickets.json
  python3 deep_ai_analysis.py --input output/tickets.json --output-dir reports/
  python3 deep_ai_analysis.py --input output/tickets.json --skip-slides
        """
    )
    parser.add_argument('--input', '-i', default='output/tickets.json',
                       help='Path to tickets.json (default: output/tickets.json)')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='Output directory for reports (default: current directory)')
    parser.add_argument('--skip-slides', action='store_true',
                       help='Skip slide generation to save time')
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.input).exists():
        console.print(f"[red]❌ Not found: {args.input}[/red]")
        sys.exit(1)
    
    # Load tickets
    console.print(f"\n[cyan]📂 Loading tickets from {args.input}...[/cyan]")
    with open(args.input, 'r', encoding='utf-8') as f:
        tickets = json.load(f)
    console.print(f"[green]✅ Loaded {len(tickets):,} tickets[/green]")
    
    # Run analysis
    console.print(Panel(
        "[bold white]🧠 FTEX Deep AI Analysis v2.0[/bold white]\n"
        "[dim]Studying ticket content to find systemic issues[/dim]",
        border_style="cyan"
    ))
    
    analyzer = DeepAnalyzer(tickets)
    
    if not analyzer.run_analysis():
        console.print("[red]❌ Analysis failed[/red]")
        sys.exit(1)
    
    # Save reports
    console.print("\n[cyan]💾 Saving reports...[/cyan]")
    outputs = analyzer.save_reports(args.output_dir)
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]✅ ANALYSIS COMPLETE[/bold green]")
    console.print("=" * 60)
    
    table = Table(box=box.SIMPLE)
    table.add_column("Output", style="cyan")
    table.add_column("File", style="white")
    
    for name, path in outputs.items():
        table.add_row(name.capitalize(), path)
    
    console.print(table)
    
    # Show synthesis preview
    if analyzer.synthesis:
        console.print(Panel(
            Markdown(analyzer.synthesis[:2000] + "..." if len(analyzer.synthesis) > 2000 else analyzer.synthesis),
            title="🧠 Analysis Preview",
            border_style="green"
        ))


if __name__ == '__main__':
    main()