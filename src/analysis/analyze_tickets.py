#!/usr/bin/env python3
"""
Freshdesk Ticket Analyzer v3.0 - Production Grade
==================================================
Advanced ticket analysis with comprehensive GenAI integration.

Features:
- Multi-pass LLM analysis for deep insights
- Production-grade prompts with few-shot examples
- Automatic cluster labeling with context
- Root cause analysis per category
- Actionable recommendations generation
- Pattern and anomaly detection

Requirements:
    pip3 install sentence-transformers scikit-learn hdbscan pandas numpy rich python-dotenv

For GenAI features (recommended):
    brew install ollama
    ollama pull qwen3:14b

Usage:
    python3 analyze_tickets.py --input output/tickets.json
    python3 analyze_tickets.py --input output/tickets/ --use-ollama
"""

import argparse
import json
import re
import sys
import os
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
FRESHDESK_DOMAIN = os.getenv('FRESHDESK_DOMAIN', 'your-domain')

# Custom stop words - add your company/product names here
CUSTOM_STOP_WORDS = os.getenv('FTEX_STOP_WORDS', '').split(',') if os.getenv('FTEX_STOP_WORDS') else []

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Rich imports
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
    from rich.align import Align
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None
    print("⚠️  Install 'rich' for beautiful terminal UI: pip3 install rich")


def print_status(msg: str, style: str = None):
    if RICH:
        console.print(msg, style=style)
    else:
        clean = re.sub(r'\[/?[^\]]+\]', '', msg)
        print(clean)


# =============================================================================
# GENAI PROMPTS - Production Grade
# =============================================================================

class GenAIPrompts:
    """Production-grade prompts for ticket analysis"""
    
    @staticmethod
    def cluster_label_prompt(subjects: List[str], descriptions: List[str] = None) -> str:
        """Generate comprehensive prompt for cluster labeling"""
        
        samples_text = "\n".join([f"  - {s}" for s in subjects[:12]])
        
        desc_context = ""
        if descriptions:
            desc_samples = [d[:200] for d in descriptions[:3] if d]
            if desc_samples:
                desc_context = f"""
Sample ticket descriptions (for additional context):
{chr(10).join([f'  "{d}..."' for d in desc_samples])}
"""
        
        return f"""You are a senior support operations analyst categorizing support tickets for a maritime/shipping software company (Digital Logbooks for vessels).

TASK: Analyze these support ticket subjects and provide a precise category label.

TICKET SUBJECTS:
{samples_text}
{desc_context}
INSTRUCTIONS:
1. Identify the PRIMARY issue type these tickets share
2. Consider: Is this about setup/onboarding, technical bugs, user errors, data issues, training needs, or feature requests?
3. Be specific but concise (3-6 words max)
4. Use domain terminology where appropriate (vessel, crew, logbook, compliance, etc.)

EXAMPLES OF GOOD LABELS:
- "Vessel Onboarding & Go-Live"
- "Data Sync Failures"
- "User Permission Issues"  
- "PDF Export Problems"
- "Crew Training Support"
- "Compliance Documentation"
- "Login & Authentication"
- "Offline Mode Issues"

EXAMPLES OF BAD LABELS (too vague):
- "General Issues"
- "Support Tickets"
- "Problems"
- "Requests"

Respond with ONLY the category label (3-6 words), nothing else."""

    @staticmethod
    def root_cause_prompt(label: str, subjects: List[str], descriptions: List[str], 
                          resolution_hours: float, ticket_count: int) -> str:
        """Generate prompt for root cause analysis"""
        
        subjects_text = "\n".join([f"  - {s}" for s in subjects[:8]])
        desc_text = "\n".join([f"  - {d[:300]}..." for d in descriptions[:3] if d])
        
        return f"""You are a senior support operations analyst performing root cause analysis for a maritime software company.

CATEGORY: {label}
TICKET COUNT: {ticket_count}
AVERAGE RESOLUTION TIME: {resolution_hours:.1f} hours ({resolution_hours/24:.1f} days)

SAMPLE TICKET SUBJECTS:
{subjects_text}

SAMPLE DESCRIPTIONS:
{desc_text}

ANALYSIS REQUIRED:
Provide a structured analysis in this exact format:

ROOT_CAUSE: [One sentence identifying the fundamental reason these tickets occur]

CONTRIBUTING_FACTORS:
- [Factor 1]
- [Factor 2]
- [Factor 3]

IMPACT_ASSESSMENT: [Low/Medium/High] - [Brief justification]

RECOMMENDED_ACTIONS:
1. [Immediate action - can be done this week]
2. [Short-term action - can be done this month]
3. [Long-term action - requires planning]

AUTOMATION_POTENTIAL: [Low/Medium/High] - [What could be automated]

Be specific to maritime/shipping software context. Focus on actionable insights."""

    @staticmethod
    def pattern_analysis_prompt(top_clusters: List[Dict], stats: Dict) -> str:
        """Generate prompt for overall pattern analysis"""
        
        cluster_summary = "\n".join([
            f"  {i+1}. {c['label']}: {c['ticket_count']} tickets ({c['percentage']}%), "
            f"Avg resolution: {c.get('avg_resolution_hours', 'N/A')} hours"
            for i, c in enumerate(top_clusters[:10])
        ])
        
        monthly = stats.get('monthly_volume', {})
        monthly_trend = ""
        if monthly:
            sorted_months = sorted(monthly.items())
            if len(sorted_months) >= 2:
                first_half = sum(v for k, v in sorted_months[:len(sorted_months)//2])
                second_half = sum(v for k, v in sorted_months[len(sorted_months)//2:])
                trend = "increasing" if second_half > first_half else "decreasing" if second_half < first_half else "stable"
                monthly_trend = f"Volume trend: {trend} ({first_half} → {second_half})"
        
        return f"""You are a senior support operations analyst providing strategic insights for a maritime software company (Digital Logbooks).

SUPPORT TICKET DATA SUMMARY:
- Total tickets analyzed: {stats.get('total_tickets', 0):,}
- Total conversations: {stats.get('total_conversations', 0):,}
- Unique companies: {len(stats.get('company_ticket_counts', {})):,}
- {monthly_trend}

TOP ISSUE CATEGORIES:
{cluster_summary}

RESOLUTION TIME STATS:
- Average: {stats.get('resolution_time_stats', {}).get('mean_hours', 0):.1f} hours
- Median: {stats.get('resolution_time_stats', {}).get('median_hours', 0):.1f} hours
- 90th percentile: {stats.get('resolution_time_stats', {}).get('p90_hours', 0):.1f} hours

Provide strategic analysis in this format:

KEY_PATTERNS:
1. [Pattern 1 - what you observe]
2. [Pattern 2]
3. [Pattern 3]

SYSTEMIC_ISSUES:
- [Issue that cuts across multiple categories]
- [Another systemic issue]

RESOURCE_ALLOCATION:
[Recommendation on how support resources should be distributed based on volume and complexity]

PRODUCT_FEEDBACK:
- [Feature gap or product issue #1 suggested by ticket patterns]
- [Feature gap #2]
- [Feature gap #3]

QUICK_WINS (changes that could reduce ticket volume by 20%+):
1. [Quick win 1 with estimated impact]
2. [Quick win 2]
3. [Quick win 3]

CUSTOMER_SUCCESS_ALERTS:
[Which customer segments or specific patterns suggest need for proactive outreach]

Be specific, data-driven, and actionable."""

    @staticmethod  
    def company_analysis_prompt(company: str, ticket_count: int, subjects: List[str], 
                                 themes: List[str]) -> str:
        """Generate prompt for analyzing repeat issues at a company"""
        
        subjects_text = "\n".join([f"  - {s}" for s in subjects[:10]])
        
        return f"""You are a customer success analyst for a maritime software company.

COMPANY: {company}
TOTAL TICKETS: {ticket_count}
RECURRING THEMES: {', '.join(themes) if themes else 'None identified'}

SAMPLE TICKET SUBJECTS:
{subjects_text}

Provide a brief analysis (3-4 sentences):
1. What is the primary challenge this company faces?
2. Is this likely a training issue, technical issue, or process issue?
3. What specific intervention would help this customer?

Be concise and actionable."""

    @staticmethod
    def executive_summary_prompt(stats: Dict, top_clusters: List[Dict], 
                                  repeat_issues: List[Dict]) -> str:
        """Generate prompt for executive summary"""
        
        top_3 = "\n".join([
            f"  - {c['label']}: {c['percentage']}% of tickets"
            for c in top_clusters[:3]
        ])
        
        top_companies = list(stats.get('company_ticket_counts', {}).items())[:5]
        companies_text = "\n".join([f"  - {c}: {n} tickets" for c, n in top_companies])
        
        return f"""You are preparing an executive summary of support operations for leadership.

KEY METRICS:
- Total tickets (6 months): {stats.get('total_tickets', 0):,}
- Average resolution time: {stats.get('resolution_time_stats', {}).get('mean_hours', 0):.1f} hours
- Tickets with no response: {stats.get('tickets_with_no_response', 0)}

TOP 3 ISSUE CATEGORIES:
{top_3}

TOP 5 CUSTOMERS BY VOLUME:
{companies_text}

Write a 3-paragraph executive summary:
1. Current state assessment (volume, trends, key issues)
2. Primary challenges and their business impact
3. Recommended priorities for next quarter

Keep it concise, data-driven, and focused on business impact. No bullet points - prose only."""


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """Production-grade Ollama client - Locked to qwen3:14b"""
    
    MODEL = "qwen3:14b"  # LOCKED MODEL - Best for support ticket analysis
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = self.MODEL
        self.available = False
        
    def check_availability(self) -> bool:
        """Check if Ollama is running with qwen3:14b"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                
                # Check if our locked model is available
                for m in models:
                    if 'qwen3:14b' in m or m == 'qwen3:14b':
                        self.model = m
                        self.available = True
                        return True
                
                # Model not found
                print_status(f"[red]Error: qwen3:14b not found. Run: ollama pull qwen3:14b[/red]")
                return False
                    
        except Exception as e:
            print_status(f"[red]Error: Ollama not running. Start with: ollama serve[/red]")
        
        return False
    
    def generate(self, prompt: str, temperature: float = 0.3, max_retries: int = 2) -> Optional[str]:
        """Generate response with retry logic"""
        if not self.available or not self.model:
            return None
        
        try:
            import requests
        except ImportError:
            return None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        'model': self.model,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': temperature,
                            'num_predict': 1500,
                            'top_p': 0.9,
                        }
                    },
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json().get('response', '').strip()
                    return result if result else None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
        
        return None


# =============================================================================
# ANALYSIS DASHBOARD
# =============================================================================

class AnalysisDashboard:
    """Live dashboard for analysis progress"""
    
    def __init__(self):
        self.phase = "Initializing"
        self.phase_icon = "🚀"
        self.current_step = ""
        self.tickets_loaded = 0
        self.embeddings_done = 0
        self.embeddings_total = 0
        self.clusters_found = 0
        self.llm_calls = 0
        self.progress_pct = 0
        self.start_time = datetime.now()
        self.steps_completed = []
        self.ollama_model = None
        
    def add_step(self, step: str):
        self.steps_completed.append((datetime.now(), step))
        
    def generate(self) -> Panel:
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        
        filled = int(self.progress_pct / 2)
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (50 - filled)}[/dim]"
        
        stats = Table(box=None, show_header=False, padding=(0, 2))
        stats.add_column()
        stats.add_column()
        
        emb_text = f"{self.embeddings_done:,}/{self.embeddings_total:,}" if self.embeddings_total else f"{self.embeddings_done:,}"
        
        stats.add_row(
            f"[dim]📋 Tickets:[/dim] [bold cyan]{self.tickets_loaded:,}[/bold cyan]",
            f"[dim]🧮 Embeddings:[/dim] [bold cyan]{emb_text}[/bold cyan]",
        )
        stats.add_row(
            f"[dim]🔬 Clusters:[/dim] [bold cyan]{self.clusters_found}[/bold cyan]",
            f"[dim]🤖 LLM Calls:[/dim] [bold cyan]{self.llm_calls}[/bold cyan]",
        )
        
        # Model info
        model_info = ""
        if self.ollama_model:
            model_info = f"\n[dim]Model:[/dim] [green]{self.ollama_model}[/green]"
        
        steps_text = ""
        for ts, step in self.steps_completed[-6:]:
            time_str = ts.strftime("%H:%M:%S")
            steps_text += f"[dim]{time_str}[/dim] [green]✓[/green] {step}\n"
        if self.current_step:
            steps_text += f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] [yellow]⟳[/yellow] [italic]{self.current_step}...[/italic]"
        
        content = Table.grid(padding=(0, 2))
        content.add_column()
        
        phase_text = Text()
        phase_text.append(f" {self.phase_icon} ", style="bold")
        phase_text.append(self.phase, style="bold cyan")
        
        content.add_row(phase_text)
        content.add_row("")
        content.add_row(f"{bar} [bold]{self.progress_pct:.0f}%[/bold]")
        content.add_row("")
        content.add_row(stats)
        if model_info:
            content.add_row(model_info)
        content.add_row("")
        content.add_row(Rule(style="dim"))
        content.add_row(f"[dim]⏱️  Elapsed:[/dim] {elapsed_str}")
        content.add_row("")
        content.add_row(Rule("Activity Log", style="dim"))
        content.add_row(steps_text if steps_text else "[dim]Starting...[/dim]")
        
        return Panel(
            content,
            title="[bold white]📊 Ticket Analyzer v3.0[/bold white]",
            subtitle="[dim]Production-Grade Analysis with GenAI[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )


# =============================================================================
# MAIN ANALYZER
# =============================================================================

class TicketAnalyzer:
    """Production-grade ticket analyzer with GenAI integration"""
    
    def __init__(self, tickets: List[Dict], use_ollama: bool = False):
        self.tickets = tickets
        self.use_ollama = use_ollama
        self.clusters = {}
        self.embeddings = None
        self.cluster_labels = []
        self.dashboard = AnalysisDashboard() if RICH else None
        self.ollama = OllamaClient() if use_ollama else None
        self.genai_insights = {}
        
    def clean_html(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        import html
        text = html.unescape(text)
        return text.strip()[:2000]
    
    def get_ticket_text(self, ticket: Dict) -> str:
        parts = []
        if ticket.get('subject'):
            parts.append(ticket['subject'])
        if ticket.get('description'):
            parts.append(self.clean_html(ticket['description']))
        for conv in ticket.get('conversations', [])[:2]:
            body = conv.get('body_text') or conv.get('body', '')
            if body:
                parts.append(self.clean_html(body)[:500])
        return ' '.join(parts)[:3000]
    
    def generate_embeddings(self, live=None):
        if self.dashboard:
            self.dashboard.current_step = "Loading embedding model"
            self.dashboard.embeddings_total = len(self.tickets)
            if live:
                live.update(self.dashboard.generate())
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print_status("[red]Error: pip3 install sentence-transformers[/red]")
            sys.exit(1)
        
        texts = [self.get_ticket_text(t) or "empty" for t in self.tickets]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if self.dashboard:
            self.dashboard.add_step("Loaded embedding model (all-MiniLM-L6-v2)")
            self.dashboard.current_step = "Generating embeddings"
            if live:
                live.update(self.dashboard.generate())
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
            if self.dashboard:
                self.dashboard.embeddings_done = len(all_embeddings)
                self.dashboard.progress_pct = (len(all_embeddings) / len(texts)) * 40
                if live:
                    live.update(self.dashboard.generate())
        
        self.embeddings = all_embeddings
        
        if self.dashboard:
            self.dashboard.add_step(f"Generated {len(self.embeddings):,} embeddings")
    
    def cluster_tickets(self, live=None):
        if self.dashboard:
            self.dashboard.current_step = "Clustering tickets"
            self.dashboard.progress_pct = 42
            if live:
                live.update(self.dashboard.generate())
        
        import numpy as np
        
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(10, len(self.tickets) // 100),
                min_samples=5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            self.cluster_labels = clusterer.fit_predict(np.array(self.embeddings))
            method = "HDBSCAN"
        except ImportError:
            from sklearn.cluster import KMeans
            n = min(25, max(5, len(self.tickets) // 100))
            clusterer = KMeans(n_clusters=n, random_state=42, n_init=10)
            self.cluster_labels = clusterer.fit_predict(np.array(self.embeddings))
            method = "KMeans"
        
        self.clusters = defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            self.clusters[label].append(idx)
        
        n_clusters = len([k for k in self.clusters.keys() if k != -1])
        
        if self.dashboard:
            self.dashboard.clusters_found = n_clusters
            self.dashboard.progress_pct = 45
            self.dashboard.add_step(f"Found {n_clusters} clusters ({method})")
            if live:
                live.update(self.dashboard.generate())
    
    def label_clusters_with_genai(self, live=None) -> Dict[int, Dict]:
        """Label clusters and perform root cause analysis using GenAI"""
        
        if self.dashboard:
            self.dashboard.current_step = "Checking Ollama availability"
            if live:
                live.update(self.dashboard.generate())
        
        # Check Ollama
        if self.ollama and self.ollama.check_availability():
            if self.dashboard:
                self.dashboard.ollama_model = self.ollama.model
                self.dashboard.add_step(f"Connected to Ollama ({self.ollama.model})")
                if live:
                    live.update(self.dashboard.generate())
        else:
            if self.dashboard:
                self.dashboard.add_step("Ollama not available, using keyword extraction")
            return self._label_clusters_keywords(live)
        
        cluster_data = {}
        sorted_clusters = sorted(
            [(k, v) for k, v in self.clusters.items() if k != -1],
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        total_to_process = min(20, len(sorted_clusters))
        
        for i, (cluster_id, ticket_indices) in enumerate(sorted_clusters[:20]):
            if self.dashboard:
                self.dashboard.current_step = f"Analyzing cluster {i+1}/{total_to_process}"
                self.dashboard.progress_pct = 45 + (i / total_to_process) * 35
                if live:
                    live.update(self.dashboard.generate())
            
            # Gather cluster data
            subjects = [self.tickets[idx].get('subject', '') for idx in ticket_indices[:15]]
            subjects = [s for s in subjects if s]
            
            descriptions = [self.clean_html(self.tickets[idx].get('description', '')) 
                           for idx in ticket_indices[:5]]
            descriptions = [d for d in descriptions if d]
            
            res_times = [self.tickets[idx].get('resolution_time_hours') 
                        for idx in ticket_indices if self.tickets[idx].get('resolution_time_hours')]
            avg_res = sum(res_times) / len(res_times) if res_times else 0
            
            # Get label
            label_prompt = GenAIPrompts.cluster_label_prompt(subjects, descriptions)
            label = self.ollama.generate(label_prompt, temperature=0.2)
            
            if self.dashboard:
                self.dashboard.llm_calls += 1
                if live:
                    live.update(self.dashboard.generate())
            
            if not label or len(label) > 60 or '\n' in label:
                label = self._extract_label_keywords(ticket_indices)
            
            # Clean up label
            label = label.strip('"\'').strip()
            
            # Get root cause analysis for top 10 clusters
            root_cause = None
            if i < 10 and len(ticket_indices) >= 20:
                rc_prompt = GenAIPrompts.root_cause_prompt(
                    label, subjects, descriptions, avg_res, len(ticket_indices)
                )
                root_cause = self.ollama.generate(rc_prompt, temperature=0.3)
                
                if self.dashboard:
                    self.dashboard.llm_calls += 1
                    if live:
                        live.update(self.dashboard.generate())
            
            cluster_data[cluster_id] = {
                'label': label,
                'root_cause_analysis': root_cause,
                'ticket_count': len(ticket_indices),
                'avg_resolution_hours': avg_res
            }
        
        # Handle uncategorized
        if -1 in self.clusters:
            cluster_data[-1] = {
                'label': 'Uncategorized / Misc',
                'root_cause_analysis': None,
                'ticket_count': len(self.clusters[-1]),
                'avg_resolution_hours': 0
            }
        
        if self.dashboard:
            self.dashboard.add_step(f"Labeled {len(cluster_data)} clusters with GenAI")
        
        return cluster_data
    
    def _label_clusters_keywords(self, live=None) -> Dict[int, Dict]:
        """Fallback: label using keywords"""
        if self.dashboard:
            self.dashboard.current_step = "Generating labels from keywords"
            if live:
                live.update(self.dashboard.generate())
        
        cluster_data = {}
        # Base stop words - common words that don't add meaning
        stop_words = {'re:', 'fw:', 'fwd:', 
                      'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'has',
                      'are', 'was', 'were', 'will', 'can', 'could', 'would', 'should',
                      'ticket', 'support', 'issue', 'problem', 'help', 'need', 'please'}
        # Add custom stop words from environment
        stop_words.update(CUSTOM_STOP_WORDS)
        
        for cluster_id, ticket_indices in self.clusters.items():
            if cluster_id == -1:
                cluster_data[cluster_id] = {
                    'label': 'Uncategorized / Misc',
                    'root_cause_analysis': None,
                    'ticket_count': len(ticket_indices),
                    'avg_resolution_hours': 0
                }
                continue
            
            label = self._extract_label_keywords(ticket_indices)
            
            res_times = [self.tickets[idx].get('resolution_time_hours') 
                        for idx in ticket_indices if self.tickets[idx].get('resolution_time_hours')]
            avg_res = sum(res_times) / len(res_times) if res_times else 0
            
            cluster_data[cluster_id] = {
                'label': label,
                'root_cause_analysis': None,
                'ticket_count': len(ticket_indices),
                'avg_resolution_hours': avg_res
            }
        
        if self.dashboard:
            self.dashboard.add_step(f"Generated {len(cluster_data)} labels (keyword-based)")
            self.dashboard.progress_pct = 80
            if live:
                live.update(self.dashboard.generate())
        
        return cluster_data
    
    def _extract_label_keywords(self, ticket_indices: List[int]) -> str:
        """Extract label from common keywords"""
        # Base stop words - common words that don't add meaning
        stop_words = {'re', 'fw', 'fwd', 
                      'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'has',
                      'are', 'was', 'were', 'will', 'can', 'could', 'would', 'should',
                      'ticket', 'support', 'issue', 'problem', 'help', 'need', 'please',
                      'hello', 'dear', 'regards', 'thanks', 'thank'}
        # Add custom stop words from environment
        stop_words.update(CUSTOM_STOP_WORDS)
        
        words = []
        for idx in ticket_indices[:50]:
            subject = self.tickets[idx].get('subject', '').lower()
            tokens = re.findall(r'\b[a-z]{3,}\b', subject)
            words.extend([w for w in tokens if w not in stop_words])
        
        if words:
            common = Counter(words).most_common(3)
            return ' / '.join([w[0].title() for w in common])
        return f"Cluster {ticket_indices[0] if ticket_indices else 'Unknown'}"
    
    def generate_strategic_insights(self, cluster_data: Dict, stats: Dict, live=None):
        """Generate high-level strategic insights using GenAI"""
        
        if not self.ollama or not self.ollama.available:
            return
        
        if self.dashboard:
            self.dashboard.current_step = "Generating strategic insights"
            self.dashboard.progress_pct = 82
            if live:
                live.update(self.dashboard.generate())
        
        # Prepare cluster summary for prompt
        top_clusters = sorted(
            [{'label': v['label'], 'ticket_count': v['ticket_count'], 
              'percentage': round(v['ticket_count'] / len(self.tickets) * 100, 1),
              'avg_resolution_hours': v['avg_resolution_hours']}
             for k, v in cluster_data.items() if k != -1],
            key=lambda x: x['ticket_count'],
            reverse=True
        )[:10]
        
        # Pattern analysis
        pattern_prompt = GenAIPrompts.pattern_analysis_prompt(top_clusters, stats)
        pattern_analysis = self.ollama.generate(pattern_prompt, temperature=0.4)
        
        if self.dashboard:
            self.dashboard.llm_calls += 1
            if live:
                live.update(self.dashboard.generate())
        
        # Executive summary
        repeat_issues = self._find_repeat_issues()
        exec_prompt = GenAIPrompts.executive_summary_prompt(stats, top_clusters, repeat_issues)
        exec_summary = self.ollama.generate(exec_prompt, temperature=0.4)
        
        if self.dashboard:
            self.dashboard.llm_calls += 1
            if live:
                live.update(self.dashboard.generate())
        
        self.genai_insights = {
            'pattern_analysis': pattern_analysis,
            'executive_summary': exec_summary
        }
        
        if self.dashboard:
            self.dashboard.add_step("Generated strategic insights")
            self.dashboard.progress_pct = 88
            if live:
                live.update(self.dashboard.generate())
    
    def calculate_statistics(self, live=None) -> Dict:
        if self.dashboard:
            self.dashboard.current_step = "Calculating statistics"
            if live:
                live.update(self.dashboard.generate())
        
        stats = {
            'total_tickets': len(self.tickets),
            'status_breakdown': Counter(),
            'priority_breakdown': Counter(),
            'source_breakdown': Counter(),
            'monthly_volume': defaultdict(int),
            'resolution_times': [],
            'company_ticket_counts': Counter(),
            'avg_conversations_per_ticket': 0,
            'tickets_with_no_response': 0,
            'tags_frequency': Counter(),
        }
        
        total_conversations = 0
        
        for ticket in self.tickets:
            stats['status_breakdown'][ticket.get('status_name', 'Unknown')] += 1
            stats['priority_breakdown'][ticket.get('priority_name', 'Unknown')] += 1
            stats['source_breakdown'][ticket.get('source_name', 'Unknown')] += 1
            
            created = ticket.get('created_at', '')
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    stats['monthly_volume'][dt.strftime('%Y-%m')] += 1
                except:
                    pass
            
            if ticket.get('resolution_time_hours'):
                stats['resolution_times'].append(ticket['resolution_time_hours'])
            
            company = ticket.get('company', {})
            if company and company.get('name'):
                stats['company_ticket_counts'][company['name']] += 1
            
            convos = ticket.get('conversations', [])
            total_conversations += len(convos)
            if len(convos) == 0:
                stats['tickets_with_no_response'] += 1
            
            for tag in ticket.get('tags', []):
                stats['tags_frequency'][tag] += 1
        
        stats['avg_conversations_per_ticket'] = total_conversations / max(len(self.tickets), 1)
        stats['total_conversations'] = total_conversations
        
        if stats['resolution_times']:
            import numpy as np
            rt = np.array(stats['resolution_times'])
            stats['resolution_time_stats'] = {
                'mean_hours': float(np.mean(rt)),
                'median_hours': float(np.median(rt)),
                'p90_hours': float(np.percentile(rt, 90)),
                'p95_hours': float(np.percentile(rt, 95)),
            }
        
        if self.dashboard:
            self.dashboard.add_step("Calculated statistics")
        
        return stats
    
    def _find_repeat_issues(self) -> List[Dict]:
        company_tickets = defaultdict(list)
        for ticket in self.tickets:
            company = ticket.get('company', {})
            if company and company.get('name'):
                company_tickets[company['name']].append(ticket)
        
        repeat_issues = []
        for company, tickets in company_tickets.items():
            if len(tickets) >= 5:
                all_words = []
                for t in tickets:
                    words = re.findall(r'\b[a-z]{4,}\b', t.get('subject', '').lower())
                    all_words.extend(words)
                common = Counter(all_words).most_common(5)
                
                repeat_issues.append({
                    'company': company,
                    'ticket_count': len(tickets),
                    'themes': [w[0] for w in common if w[1] >= 3],
                    'samples': [t.get('subject', '')[:60] for t in tickets[:3]]
                })
        
        return sorted(repeat_issues, key=lambda x: x['ticket_count'], reverse=True)[:20]
    
    def _find_long_resolution(self) -> List[Dict]:
        with_res = [t for t in self.tickets if t.get('resolution_time_hours')]
        sorted_t = sorted(with_res, key=lambda x: x['resolution_time_hours'], reverse=True)
        
        return [{
            'id': t.get('id'),
            'subject': t.get('subject', '')[:60],
            'hours': round(t['resolution_time_hours'], 1),
            'days': round(t['resolution_time_hours'] / 24, 1),
            'company': t.get('company', {}).get('name', 'Unknown'),
        } for t in sorted_t[:20]]
    
    def generate_report(self, output_path: str = "analysis_report.md", live=None):
        """Generate comprehensive report"""
        
        # Run pipeline
        self.generate_embeddings(live)
        self.cluster_tickets(live)
        cluster_data = self.label_clusters_with_genai(live)
        stats = self.calculate_statistics(live)
        
        # Generate strategic insights if using Ollama
        if self.use_ollama:
            self.generate_strategic_insights(cluster_data, stats, live)
        
        repeat_issues = self._find_repeat_issues()
        long_resolution = self._find_long_resolution()
        
        if self.dashboard:
            self.dashboard.current_step = "Writing report"
            self.dashboard.progress_pct = 92
            if live:
                live.update(self.dashboard.generate())
        
        # Build report
        report = []
        
        # Header
        report.append("# 📊 Freshdesk Ticket Analysis Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**Tickets Analyzed:** {stats['total_tickets']:,}")
        report.append(f"**Conversations:** {stats['total_conversations']:,}")
        report.append(f"**Analysis Method:** {'GenAI-Enhanced' if self.ollama and self.ollama.available else 'Statistical'}")
        if self.ollama and self.ollama.model:
            report.append(f"**Model:** {self.ollama.model}")
        report.append("")
        
        # Executive Summary (GenAI)
        if self.genai_insights.get('executive_summary'):
            report.append("## 📋 Executive Summary")
            report.append("")
            report.append(self.genai_insights['executive_summary'])
            report.append("")
        
        # Key Metrics
        report.append("## 📈 Key Metrics")
        report.append("")
        if stats.get('resolution_time_stats'):
            rt = stats['resolution_time_stats']
            report.append(f"| Metric | Value |")
            report.append(f"|--------|-------|")
            report.append(f"| Total Tickets | {stats['total_tickets']:,} |")
            report.append(f"| Avg Resolution | {rt['mean_hours']:.1f} hrs ({rt['mean_hours']/24:.1f} days) |")
            report.append(f"| Median Resolution | {rt['median_hours']:.1f} hrs ({rt['median_hours']/24:.1f} days) |")
            report.append(f"| 90th Percentile | {rt['p90_hours']:.1f} hrs ({rt['p90_hours']/24:.1f} days) |")
            report.append(f"| Avg Conversations | {stats['avg_conversations_per_ticket']:.1f} |")
            report.append(f"| No Response | {stats['tickets_with_no_response']:,} ({stats['tickets_with_no_response']/stats['total_tickets']*100:.1f}%) |")
        report.append("")
        
        # Pattern Analysis (GenAI)
        if self.genai_insights.get('pattern_analysis'):
            report.append("## 🔍 Strategic Pattern Analysis")
            report.append("")
            report.append(self.genai_insights['pattern_analysis'])
            report.append("")
        
        # Issue Categories with Root Cause
        report.append("## 🏷️ Issue Categories")
        report.append("")
        
        sorted_clusters = sorted(
            [(k, v) for k, v in cluster_data.items() if k != -1],
            key=lambda x: x[1]['ticket_count'],
            reverse=True
        )
        
        for i, (cluster_id, data) in enumerate(sorted_clusters[:15], 1):
            ticket_indices = self.clusters[cluster_id]
            pct = data['ticket_count'] / stats['total_tickets'] * 100
            
            report.append(f"### {i}. {data['label']}")
            report.append(f"**{data['ticket_count']:,} tickets ({pct:.1f}%)**")
            
            if data['avg_resolution_hours']:
                report.append(f"- Resolution: {data['avg_resolution_hours']:.1f} hrs ({data['avg_resolution_hours']/24:.1f} days)")
            
            # Status breakdown
            status_counter = Counter()
            company_counter = Counter()
            for idx in ticket_indices:
                status_counter[self.tickets[idx].get('status_name', 'Unknown')] += 1
                co = self.tickets[idx].get('company', {})
                if co and co.get('name'):
                    company_counter[co['name']] += 1
            
            report.append(f"- Status: {', '.join([f'{s}:{c}' for s,c in status_counter.most_common(3)])}")
            
            top_cos = company_counter.most_common(3)
            if top_cos:
                report.append(f"- Top Companies: {', '.join([f'{c[0]} ({c[1]})' for c in top_cos])}")
            
            # Sample tickets
            report.append("\n**Samples:**")
            for idx in ticket_indices[:4]:
                subj = self.tickets[idx].get('subject', 'N/A')[:70]
                report.append(f"- {subj}")
            
            # Root cause analysis
            if data.get('root_cause_analysis'):
                report.append("\n**Root Cause Analysis:**")
                report.append(f"```\n{data['root_cause_analysis']}\n```")
            
            report.append("")
        
        # Status Breakdown
        report.append("## 📊 Status Distribution")
        report.append("")
        report.append("| Status | Count | % |")
        report.append("|--------|-------|---|")
        for status, count in stats['status_breakdown'].most_common():
            report.append(f"| {status} | {count:,} | {count/stats['total_tickets']*100:.1f}% |")
        report.append("")
        
        # Monthly Volume
        report.append("## 📅 Monthly Trend")
        report.append("")
        report.append("| Month | Tickets |")
        report.append("|-------|---------|")
        for month in sorted(stats['monthly_volume'].keys()):
            report.append(f"| {month} | {stats['monthly_volume'][month]:,} |")
        report.append("")
        
        # Top Companies
        report.append("## 🏢 Top 20 Companies")
        report.append("")
        report.append("| # | Company | Tickets | % |")
        report.append("|---|---------|---------|---|")
        for rank, (company, count) in enumerate(stats['company_ticket_counts'].most_common(20), 1):
            report.append(f"| {rank} | {company} | {count:,} | {count/stats['total_tickets']*100:.1f}% |")
        report.append("")
        
        # Repeat Issues
        if repeat_issues:
            report.append("## 🔄 Repeat Issue Patterns")
            report.append("")
            for issue in repeat_issues[:10]:
                report.append(f"**{issue['company']}** ({issue['ticket_count']} tickets)")
                if issue['themes']:
                    report.append(f"- Themes: {', '.join(issue['themes'])}")
                report.append(f"- Samples: {'; '.join(issue['samples'])}")
                report.append("")
        
        # Long Resolution
        if long_resolution:
            report.append("## ⏰ Longest Resolution Times")
            report.append("")
            report.append("| ID | Subject | Days | Company |")
            report.append("|----|---------|------|---------|")
            for t in long_resolution[:15]:
                report.append(f"| {t['id']} | {t['subject']} | {t['days']} | {t['company']} |")
            report.append("")
        
        # Tags
        if stats['tags_frequency']:
            report.append("## 🏷️ Top Tags")
            report.append("")
            report.append("| Tag | Count |")
            report.append("|-----|-------|")
            for tag, count in stats['tags_frequency'].most_common(15):
                report.append(f"| {tag} | {count:,} |")
            report.append("")
        
        # Questions for Claude
        report.append("## 🎯 Questions for Strategic Review")
        report.append("")
        report.append("Please analyze this report and provide:")
        report.append("")
        report.append("1. **Validation:** Do the identified patterns align with your understanding of the product/support?")
        report.append("2. **Missing Context:** What additional context would change these conclusions?")
        report.append("3. **Prioritization:** Given limited resources, what's the #1 thing to fix first?")
        report.append("4. **ROI Estimate:** Which improvements would have the highest impact per effort?")
        report.append("5. **Customer Impact:** Which changes would most improve customer satisfaction?")
        report.append("")
        
        # Write files
        report_text = '\n'.join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Save data JSON
        data_path = output_path.replace('.md', '_data.json')
        
        cluster_export = []
        for cid, data in sorted_clusters[:20]:
            cluster_export.append({
                'id': cid,
                'label': data['label'],
                'ticket_count': data['ticket_count'],
                'percentage': round(data['ticket_count'] / stats['total_tickets'] * 100, 1),
                'avg_resolution_hours': data['avg_resolution_hours'],
                'root_cause_analysis': data.get('root_cause_analysis'),
                'sample_subjects': [self.tickets[i].get('subject', '')[:80] for i in self.clusters[cid][:5]]
            })
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'meta': {
                    'generated': datetime.now().isoformat(),
                    'total_tickets': stats['total_tickets'],
                    'model_used': self.ollama.model if self.ollama else None,
                    'llm_calls': self.dashboard.llm_calls if self.dashboard else 0
                },
                'clusters': cluster_export,
                'stats': {
                    'total': stats['total_tickets'],
                    'conversations': stats['total_conversations'],
                    'resolution': stats.get('resolution_time_stats', {}),
                    'status': dict(stats['status_breakdown']),
                    'priority': dict(stats['priority_breakdown']),
                    'monthly': dict(stats['monthly_volume']),
                    'top_companies': dict(stats['company_ticket_counts'].most_common(50)),
                    'tags': dict(stats['tags_frequency'].most_common(30))
                },
                'genai_insights': self.genai_insights,
                'repeat_issues': repeat_issues,
                'long_resolution': long_resolution
            }, f, indent=2, default=str)
        
        if self.dashboard:
            self.dashboard.progress_pct = 100
            self.dashboard.add_step(f"Saved report ({len(report_text):,} chars)")
            self.dashboard.current_step = ""
            if live:
                live.update(self.dashboard.generate())
        
        return output_path, data_path, len(report_text)


# =============================================================================
# UTILITIES
# =============================================================================

def load_tickets(input_path: str) -> List[Dict]:
    path = Path(input_path)
    
    if path.is_file():
        print_status(f"[dim]Loading from: {path}[/dim]")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif path.is_dir():
        print_status(f"[dim]Loading from directory: {path}[/dim]")
        tickets = []
        json_files = list(path.glob("ticket_*.json"))
        
        if not json_files:
            print_status(f"[red]No ticket_*.json files in {path}[/red]")
            sys.exit(1)
        
        if RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Loading...", total=len(json_files))
                for file in json_files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            tickets.append(json.load(f))
                    except:
                        pass
                    progress.update(task, advance=1)
        else:
            for i, file in enumerate(json_files):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        tickets.append(json.load(f))
                except:
                    pass
        
        return tickets
    
    print_status(f"[red]Not found: {input_path}[/red]")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Production-grade ticket analyzer with GenAI')
    parser.add_argument('--input', '-i', default='output/tickets.json')
    parser.add_argument('--output', '-o', default='analysis_report.md')
    parser.add_argument('--use-ollama', action='store_true', help='Enable GenAI analysis')
    
    args = parser.parse_args()
    
    if RICH:
        console.print(Panel(
            Align.center(Text.from_markup(
                "[bold white]📊 Ticket Analyzer v3.0[/bold white]\n"
                "[dim]Production-Grade Analysis with GenAI[/dim]"
            )),
            border_style="cyan"
        ))
        console.print()
    
    tickets = load_tickets(args.input)
    print_status(f"[green]✓ Loaded {len(tickets):,} tickets[/green]\n")
    
    if not tickets:
        print_status("[red]No tickets found.[/red]")
        sys.exit(1)
    
    analyzer = TicketAnalyzer(tickets, use_ollama=args.use_ollama)
    
    if RICH:
        analyzer.dashboard.tickets_loaded = len(tickets)
        analyzer.dashboard.phase = "Analyzing Tickets"
        analyzer.dashboard.phase_icon = "📊"
        
        with Live(analyzer.dashboard.generate(), console=console, refresh_per_second=4) as live:
            report_path, data_path, report_size = analyzer.generate_report(args.output, live)
        
        console.print()
        table = Table(title="✅ Analysis Complete", box=box.ROUNDED, border_style="green")
        table.add_column("Output", style="cyan")
        table.add_column("Details", style="white")
        table.add_row("📄 Report", report_path)
        table.add_row("📊 Data", data_path)
        table.add_row("📏 Size", f"{report_size:,} chars (~{report_size//4:,} tokens)")
        if analyzer.ollama and analyzer.ollama.model:
            table.add_row("🤖 Model", analyzer.ollama.model)
            table.add_row("💬 LLM Calls", str(analyzer.dashboard.llm_calls))
        console.print(table)
        console.print(f"\n[bold]Next:[/bold] Share [cyan]{report_path}[/cyan] with Claude.\n")
    else:
        report_path, data_path, report_size = analyzer.generate_report(args.output)
        print(f"\nReport: {report_path}")
        print(f"Data: {data_path}")


if __name__ == '__main__':
    main()