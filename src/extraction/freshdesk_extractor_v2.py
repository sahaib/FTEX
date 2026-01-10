#!/usr/bin/env python3
"""
Freshdesk Ticket Extractor v2.1
================================
Advanced extraction with:
- Rich live dashboard with real-time stats
- Checkpoint/resume support
- Incremental disk saves (low memory)
- Automatic crash recovery

Usage:
    pip3 install requests rich python-dotenv
    python3 freshdesk_extractor_v2.py --api-key YOUR_KEY --days 180 --group-id YOUR_GROUP_ID

Resume interrupted extraction:
    python3 freshdesk_extractor_v2.py --api-key YOUR_KEY --resume
"""

import argparse
import csv
import json
import os
import sys
import time
import html
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any, Set
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
FRESHDESK_DOMAIN = os.getenv('FRESHDESK_DOMAIN', 'your-domain')
FRESHDESK_API_KEY = os.getenv('FRESHDESK_API_KEY', '')
FRESHDESK_GROUP_ID = os.getenv('FRESHDESK_GROUP_ID', '')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Rich imports for beautiful terminal output
try:
    from rich.console import Console, Group
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    from rich.rule import Rule
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for beautiful terminal UI: pip3 install rich")
    print("   Falling back to basic output...\n")


class LiveDashboard:
    """Real-time dashboard for extraction progress"""
    
    def __init__(self, console: Console):
        self.console = console
        self.phase = "Initializing"
        self.phase_icon = "🚀"
        self.current_action = ""
        self.tickets_discovered = 0
        self.tickets_processed = 0
        self.tickets_total = 0
        self.conversations = 0
        self.attachments = 0
        self.errors = 0
        self.rate_limit_remaining = 0
        self.rate_limit_total = 0
        self.start_time = datetime.now()
        self.recent_tickets = []  # Last 5 processed
        self.chunk_progress = (0, 0)  # current, total
        
    def set_phase(self, phase: str, icon: str = "⚙️"):
        self.phase = phase
        self.phase_icon = icon
        
    def add_recent_ticket(self, ticket_id: int, subject: str):
        self.recent_tickets.insert(0, (ticket_id, subject[:50]))
        self.recent_tickets = self.recent_tickets[:5]
    
    def generate(self) -> Panel:
        """Generate the dashboard layout"""
        
        # Header
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        
        # Calculate ETA
        eta_str = "--:--:--"
        if self.tickets_total > 0 and self.tickets_processed > 0:
            rate = self.tickets_processed / max(elapsed.total_seconds(), 1)
            remaining = self.tickets_total - self.tickets_processed
            if rate > 0:
                eta_seconds = remaining / rate
                eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Stats table
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        stats_table.add_column("Label", style="dim")
        stats_table.add_column("Value", style="bold green")
        stats_table.add_column("Label2", style="dim")
        stats_table.add_column("Value2", style="bold cyan")
        
        stats_table.add_row(
            "📋 Discovered", f"{self.tickets_discovered:,}",
            "💬 Conversations", f"{self.conversations:,}"
        )
        stats_table.add_row(
            "✅ Processed", f"{self.tickets_processed:,}",
            "📎 Attachments", f"{self.attachments:,}"
        )
        stats_table.add_row(
            "📦 Total", f"{self.tickets_total:,}",
            "❌ Errors", f"[red]{self.errors}[/red]" if self.errors > 0 else "0"
        )
        
        # Progress bar
        if self.tickets_total > 0:
            pct = (self.tickets_processed / self.tickets_total) * 100
            filled = int(pct / 2)
            bar = f"[green]{'█' * filled}[/green][dim]{'░' * (50 - filled)}[/dim]"
            progress_text = f"{bar} {pct:.1f}%"
        else:
            progress_text = "[dim]Waiting...[/dim]"
        
        # Rate limit indicator
        if self.rate_limit_total > 0:
            rl_pct = (self.rate_limit_remaining / self.rate_limit_total) * 100
            if rl_pct > 50:
                rl_style = "green"
            elif rl_pct > 20:
                rl_style = "yellow"
            else:
                rl_style = "red"
            rate_limit_text = f"[{rl_style}]●[/{rl_style}] API: {self.rate_limit_remaining:,}/{self.rate_limit_total:,}"
        else:
            rate_limit_text = "[dim]● API: --[/dim]"
        
        # Recent activity
        recent_lines = []
        for tid, subj in self.recent_tickets:
            recent_lines.append(f"[dim]#{tid}[/dim] {subj}")
        recent_text = "\n".join(recent_lines) if recent_lines else "[dim]No tickets processed yet[/dim]"
        
        # Build layout
        content = Table.grid(padding=(0, 2))
        content.add_column()
        
        # Phase header
        phase_text = Text()
        phase_text.append(f" {self.phase_icon} ", style="bold")
        phase_text.append(self.phase, style="bold cyan")
        if self.current_action:
            phase_text.append(f" • ", style="dim")
            phase_text.append(self.current_action, style="italic")
        
        content.add_row(phase_text)
        content.add_row("")
        content.add_row(progress_text)
        content.add_row("")
        content.add_row(stats_table)
        content.add_row("")
        content.add_row(Rule(style="dim"))
        content.add_row("")
        
        # Time and rate info
        time_table = Table(box=None, show_header=False, padding=(0, 3))
        time_table.add_column()
        time_table.add_column()
        time_table.add_column()
        time_table.add_row(
            f"[dim]⏱️  Elapsed:[/dim] [white]{elapsed_str}[/white]",
            f"[dim]⏳ ETA:[/dim] [white]{eta_str}[/white]",
            rate_limit_text
        )
        content.add_row(time_table)
        content.add_row("")
        content.add_row(Rule("Recent Activity", style="dim"))
        content.add_row(Text(recent_text, style=""))
        
        # Main panel
        panel = Panel(
            content,
            title="[bold white]🚀 Freshdesk Extractor v2.1[/bold white]",
            subtitle="[dim]Press Ctrl+C to stop (progress is saved)[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        return panel


class FreshdeskExtractorV2:
    """
    Production-grade Freshdesk extractor with checkpoint/resume support.
    """
    
    BASE_URL = "https://{domain}.freshdesk.com/api/v2"
    
    STATUS_MAP = {2: 'Open', 3: 'Pending', 4: 'Resolved', 5: 'Closed'}
    PRIORITY_MAP = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Urgent'}
    SOURCE_MAP = {1: 'Email', 2: 'Portal', 3: 'Phone', 7: 'Chat', 9: 'Feedback Widget', 10: 'Outbound Email'}
    
    def __init__(
        self,
        domain: str,
        api_key: str,
        output_dir: str = "output",
        requests_per_minute: int = 40,
        download_attachments: bool = True,
        group_id: Optional[int] = None,
    ):
        self.domain = domain
        self.api_key = api_key
        self.base_url = self.BASE_URL.format(domain=domain)
        self.output_dir = Path(output_dir)
        self.attachments_dir = self.output_dir / "attachments"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.tickets_dir = self.output_dir / "tickets"
        self.requests_per_minute = requests_per_minute
        self.download_attachments = download_attachments
        self.group_id = group_id
        
        # Rate limiting
        self.min_request_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.rate_limit_remaining = 4000
        self.rate_limit_total = 4000
        
        # Stats
        self.stats = {
            'tickets_discovered': 0,
            'tickets_processed': 0,
            'conversations_fetched': 0,
            'attachments_downloaded': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Session
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(api_key, 'X')
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Rich console and dashboard
        self.console = Console() if RICH_AVAILABLE else None
        self.dashboard = LiveDashboard(self.console) if RICH_AVAILABLE else None
        
        # Create directories
        for d in [self.output_dir, self.attachments_dir, self.checkpoint_dir, self.tickets_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _print(self, message: str, style: str = None):
        """Print with rich formatting if available"""
        if self.console and not hasattr(self, '_live_active'):
            self.console.print(message, style=style)
        elif not RICH_AVAILABLE:
            print(message)
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        if self.rate_limit_remaining < 100:
            if self.dashboard:
                self.dashboard.current_action = "⚠️ Rate limit low, waiting 60s..."
            time.sleep(60)
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retry_count: int = 3) -> Optional[requests.Response]:
        """Make rate-limited API request with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(retry_count):
            self._wait_for_rate_limit()
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                self.last_request_time = time.time()
                
                # Update rate limits
                self.rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', self.rate_limit_remaining))
                self.rate_limit_total = int(response.headers.get('x-ratelimit-total', self.rate_limit_total))
                
                if self.dashboard:
                    self.dashboard.rate_limit_remaining = self.rate_limit_remaining
                    self.dashboard.rate_limit_total = self.rate_limit_total
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    if self.dashboard:
                        self.dashboard.current_action = f"Rate limited, waiting {retry_after}s..."
                    time.sleep(retry_after)
                    continue
                elif response.status_code == 404:
                    return None
                else:
                    if attempt < retry_count - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    self.stats['errors'] += 1
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt < retry_count - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                self.stats['errors'] += 1
                return None
        
        return None
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        self._print("\n[bold blue]🔌 Testing API connection...[/bold blue]")
        response = self._make_request('tickets', params={'per_page': 1})
        
        if response:
            self._print(f"[green]✓ Connected![/green] Rate limit: {self.rate_limit_remaining}/{self.rate_limit_total}")
            return True
        else:
            self._print("[red]✗ Connection failed. Check API key and domain.[/red]")
            return False
    
    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================
    
    def _get_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "extraction_state.json"
    
    def _save_checkpoint(self, state: Dict):
        """Save extraction state for resume"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state['last_updated'] = datetime.now(timezone.utc).isoformat()
        with open(self._get_checkpoint_path(), 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """Load previous extraction state"""
        path = self._get_checkpoint_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_processed_ticket_ids(self) -> Set[int]:
        """Get set of already processed ticket IDs from saved files"""
        processed = set()
        for file in self.tickets_dir.glob("ticket_*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    processed.add(data['id'])
            except:
                pass
        return processed
    
    def _save_ticket(self, ticket: Dict):
        """Save single ticket to disk immediately"""
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        path = self.tickets_dir / f"ticket_{ticket['id']}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ticket, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_ticket_ids(self, ticket_ids: List[int]):
        """Save discovered ticket IDs"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "ticket_ids.json"
        with open(path, 'w') as f:
            json.dump(ticket_ids, f)
    
    def _load_ticket_ids(self) -> Optional[List[int]]:
        """Load previously discovered ticket IDs"""
        path = self.checkpoint_dir / "ticket_ids.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    # =========================================================================
    # TICKET DISCOVERY (Phase 1)
    # =========================================================================
    
    def discover_tickets(self, days: int, group_id: int) -> List[int]:
        """
        Phase 1: Discover all ticket IDs using Search API with date chunking.
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Create weekly chunks
        chunks = []
        chunk_end = end_date
        while chunk_end > start_date:
            chunk_start = max(chunk_end - timedelta(days=7), start_date)
            chunks.append((chunk_start, chunk_end))
            chunk_end = chunk_start
        
        all_ticket_ids = []
        seen_ids = set()
        
        if self.dashboard:
            self.dashboard.set_phase("Phase 1: Discovering Tickets", "🔍")
        
        self._print(f"\n[bold cyan]📋 Phase 1: Discovering tickets[/bold cyan]")
        self._print(f"   Scanning {len(chunks)} weekly chunks for group {group_id}\n")
        
        if RICH_AVAILABLE and self.dashboard:
            with Live(self.dashboard.generate(), console=self.console, refresh_per_second=4) as live:
                self._live_active = True
                for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                    self.dashboard.current_action = f"Chunk {chunk_idx+1}/{len(chunks)}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
                    self.dashboard.chunk_progress = (chunk_idx + 1, len(chunks))
                    
                    chunk_tickets = self._search_chunk(group_id, chunk_start, chunk_end, seen_ids)
                    
                    for tid in chunk_tickets:
                        if tid not in seen_ids:
                            seen_ids.add(tid)
                            all_ticket_ids.append(tid)
                    
                    self.dashboard.tickets_discovered = len(all_ticket_ids)
                    live.update(self.dashboard.generate())
                
                del self._live_active
        else:
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                print(f"  Chunk {chunk_idx+1}/{len(chunks)}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}", end="")
                chunk_tickets = self._search_chunk(group_id, chunk_start, chunk_end, seen_ids)
                
                for tid in chunk_tickets:
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        all_ticket_ids.append(tid)
                
                print(f" → {len(chunk_tickets)} tickets (total: {len(all_ticket_ids)})")
        
        self.stats['tickets_discovered'] = len(all_ticket_ids)
        self._save_ticket_ids(all_ticket_ids)
        
        self._print(f"\n[green]✓ Discovered {len(all_ticket_ids):,} tickets[/green]")
        
        return all_ticket_ids
    
    def _search_chunk(self, group_id: int, start_date: datetime, end_date: datetime, seen_ids: Set[int]) -> List[int]:
        """Search tickets in a date range, auto-split to daily if needed"""
        ticket_ids = []
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f'"group_id:{group_id} AND updated_at:>\'{start_str}\' AND updated_at:<\'{end_str}\'"'
        
        page = 1
        hit_limit = False
        
        while page <= 10:
            response = self._make_request('search/tickets', params={'query': query, 'page': page})
            
            if not response:
                break
            
            data = response.json()
            tickets = data.get('results', [])
            
            if not tickets:
                break
            
            for t in tickets:
                if t['id'] not in seen_ids:
                    ticket_ids.append(t['id'])
            
            if len(tickets) < 30:
                break
            
            page += 1
            
            if page > 10:
                hit_limit = True
                break
        
        if hit_limit:
            ticket_ids = self._search_daily(group_id, start_date, end_date, seen_ids)
        
        return ticket_ids
    
    def _search_daily(self, group_id: int, start_date: datetime, end_date: datetime, seen_ids: Set[int]) -> List[int]:
        """Search day by day for busy periods"""
        ticket_ids = []
        current = start_date
        
        while current < end_date:
            next_day = current + timedelta(days=1)
            start_str = current.strftime('%Y-%m-%d')
            end_str = next_day.strftime('%Y-%m-%d')
            
            query = f'"group_id:{group_id} AND updated_at:>\'{start_str}\' AND updated_at:<\'{end_str}\'"'
            
            page = 1
            while page <= 10:
                response = self._make_request('search/tickets', params={'query': query, 'page': page})
                
                if not response:
                    break
                
                data = response.json()
                tickets = data.get('results', [])
                
                if not tickets:
                    break
                
                for t in tickets:
                    if t['id'] not in seen_ids:
                        ticket_ids.append(t['id'])
                
                if len(tickets) < 30:
                    break
                
                page += 1
            
            current = next_day
        
        return ticket_ids
    
    # =========================================================================
    # TICKET PROCESSING (Phase 2)
    # =========================================================================
    
    def process_tickets(self, ticket_ids: List[int]) -> int:
        """
        Phase 2: Fetch full details + conversations for each ticket.
        """
        processed_ids = self._get_processed_ticket_ids()
        remaining_ids = [tid for tid in ticket_ids if tid not in processed_ids]
        
        if processed_ids:
            self._print(f"\n[yellow]📌 Resuming: {len(processed_ids):,} already processed, {len(remaining_ids):,} remaining[/yellow]")
        
        if self.dashboard:
            self.dashboard.set_phase("Phase 2: Processing Tickets", "⚙️")
            self.dashboard.tickets_total = len(ticket_ids)
            self.dashboard.tickets_processed = len(processed_ids)
        
        self._print(f"\n[bold cyan]⚙️  Phase 2: Processing tickets[/bold cyan]")
        self._print(f"   Fetching details + conversations for {len(remaining_ids):,} tickets\n")
        
        if RICH_AVAILABLE and self.dashboard:
            with Live(self.dashboard.generate(), console=self.console, refresh_per_second=4) as live:
                self._live_active = True
                
                for i, ticket_id in enumerate(remaining_ids):
                    self.dashboard.current_action = f"Ticket #{ticket_id}"
                    
                    ticket = self._process_single_ticket(ticket_id)
                    
                    if ticket:
                        self._save_ticket(ticket)
                        self.stats['tickets_processed'] += 1
                        self.dashboard.tickets_processed = len(processed_ids) + self.stats['tickets_processed']
                        self.dashboard.conversations = self.stats['conversations_fetched']
                        self.dashboard.attachments = self.stats['attachments_downloaded']
                        self.dashboard.errors = self.stats['errors']
                        self.dashboard.add_recent_ticket(ticket_id, ticket.get('subject', 'Unknown'))
                    
                    live.update(self.dashboard.generate())
                    
                    # Checkpoint every 100 tickets
                    if (i + 1) % 100 == 0:
                        self._save_checkpoint({
                            'phase': 'processing',
                            'total_tickets': len(ticket_ids),
                            'processed': len(processed_ids) + i + 1,
                            'stats': self.stats
                        })
                
                del self._live_active
        else:
            for i, ticket_id in enumerate(remaining_ids):
                ticket = self._process_single_ticket(ticket_id)
                
                if ticket:
                    self._save_ticket(ticket)
                    self.stats['tickets_processed'] += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(remaining_ids)} tickets...")
                    self._save_checkpoint({
                        'phase': 'processing',
                        'total_tickets': len(ticket_ids),
                        'processed': len(processed_ids) + i + 1,
                        'stats': self.stats
                    })
        
        return self.stats['tickets_processed']
    
    def _process_single_ticket(self, ticket_id: int) -> Optional[Dict]:
        """Fetch full ticket details and conversations"""
        response = self._make_request(
            f'tickets/{ticket_id}',
            params={'include': 'requester,company,stats'}
        )
        
        if not response:
            return None
        
        ticket = response.json()
        
        # Get conversations
        conversations = self._get_conversations(ticket_id)
        ticket['conversations'] = conversations
        self.stats['conversations_fetched'] += len(conversations)
        
        # Download attachments if enabled
        if self.download_attachments:
            self._download_ticket_attachments(ticket)
        
        # Add derived fields
        ticket['status_name'] = self.STATUS_MAP.get(ticket.get('status'), 'Unknown')
        ticket['priority_name'] = self.PRIORITY_MAP.get(ticket.get('priority'), 'Unknown')
        ticket['source_name'] = self.SOURCE_MAP.get(ticket.get('source'), 'Unknown')
        ticket['response_count'] = len(conversations)
        
        # Calculate resolution time
        if ticket.get('stats') and ticket['stats'].get('resolved_at'):
            try:
                created = datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00'))
                resolved = datetime.fromisoformat(ticket['stats']['resolved_at'].replace('Z', '+00:00'))
                ticket['resolution_time_hours'] = (resolved - created).total_seconds() / 3600
            except:
                pass
        
        return ticket
    
    def _get_conversations(self, ticket_id: int) -> List[Dict]:
        """Fetch all conversations for a ticket"""
        all_conversations = []
        page = 1
        
        while True:
            response = self._make_request(
                f'tickets/{ticket_id}/conversations',
                params={'per_page': 100, 'page': page}
            )
            
            if not response:
                break
            
            conversations = response.json()
            
            if not conversations:
                break
            
            all_conversations.extend(conversations)
            
            if len(conversations) < 100:
                break
            
            page += 1
        
        return all_conversations
    
    def _download_ticket_attachments(self, ticket: Dict):
        """Download attachments for a ticket"""
        ticket_id = ticket['id']
        
        for att in ticket.get('attachments', []):
            local_path = self._download_attachment(ticket_id, att)
            if local_path:
                att['local_path'] = local_path
        
        for conv in ticket.get('conversations', []):
            for att in conv.get('attachments', []):
                local_path = self._download_attachment(ticket_id, att)
                if local_path:
                    att['local_path'] = local_path
    
    def _download_attachment(self, ticket_id: int, attachment: Dict) -> Optional[str]:
        """Download single attachment"""
        attachment_url = attachment.get('attachment_url')
        if not attachment_url:
            return None
        
        attachment_name = attachment.get('name', f"attachment_{attachment.get('id')}")
        
        ticket_dir = self.attachments_dir / str(ticket_id)
        ticket_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = "".join(c for c in attachment_name if c.isalnum() or c in '._- ')[:100]
        local_path = ticket_dir / safe_name
        
        if local_path.exists():
            return str(local_path)
        
        try:
            self._wait_for_rate_limit()
            response = self.session.get(attachment_url, timeout=60, stream=True)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.stats['attachments_downloaded'] += 1
                return str(local_path)
        except:
            self.stats['errors'] += 1
        
        return None
    
    # =========================================================================
    # FINALIZATION (Phase 3)
    # =========================================================================
    
    def finalize(self) -> Dict:
        """Phase 3: Combine all saved tickets into final output files."""
        
        if self.dashboard:
            self.dashboard.set_phase("Phase 3: Finalizing", "📦")
            self.dashboard.current_action = "Combining ticket files..."
        
        self._print(f"\n[bold cyan]📦 Phase 3: Finalizing output[/bold cyan]")
        
        # Load all saved tickets
        tickets = []
        for file in sorted(self.tickets_dir.glob("ticket_*.json")):
            with open(file, 'r', encoding='utf-8') as f:
                tickets.append(json.load(f))
        
        self._print(f"   Loaded {len(tickets):,} tickets from disk")
        
        # Save combined JSON
        json_path = self.output_dir / 'tickets.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, indent=2, ensure_ascii=False, default=str)
        json_size = json_path.stat().st_size / (1024 * 1024)
        self._print(f"   [green]✓[/green] Saved {json_path} ({json_size:.1f} MB)")
        
        # Save CSV
        csv_path = self.output_dir / 'tickets.csv'
        if tickets:
            flattened = [self._flatten_ticket(t) for t in tickets]
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
            csv_size = csv_path.stat().st_size / (1024 * 1024)
            self._print(f"   [green]✓[/green] Saved {csv_path} ({csv_size:.1f} MB)")
        
        # Save extraction log
        self.stats['end_time'] = datetime.now(timezone.utc).isoformat()
        log_path = self.output_dir / 'extraction_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'stats': self.stats,
                'output_files': {
                    'json': str(json_path),
                    'csv': str(csv_path),
                    'attachments': str(self.attachments_dir),
                    'individual_tickets': str(self.tickets_dir)
                }
            }, f, indent=2)
        self._print(f"   [green]✓[/green] Saved {log_path}")
        
        return {'tickets': tickets, 'stats': self.stats}
    
    def _flatten_ticket(self, ticket: Dict) -> Dict:
        """Flatten ticket for CSV"""
        description = ticket.get('description', '') or ''
        clean_desc = html.unescape(description)
        clean_desc = ''.join(c if c.isprintable() or c in '\n\t' else ' ' for c in clean_desc)
        
        conv_preview = []
        for conv in ticket.get('conversations', [])[:3]:
            body = conv.get('body_text') or conv.get('body', '')
            if body:
                direction = 'IN' if conv.get('incoming') else 'OUT'
                conv_preview.append(f"[{direction}] {body[:200]}")
        
        return {
            'id': ticket.get('id'),
            'subject': ticket.get('subject'),
            'description': clean_desc[:2000],
            'status': ticket.get('status'),
            'status_name': ticket.get('status_name'),
            'priority': ticket.get('priority'),
            'priority_name': ticket.get('priority_name'),
            'source': ticket.get('source'),
            'source_name': ticket.get('source_name'),
            'type': ticket.get('type'),
            'tags': ','.join(ticket.get('tags', [])),
            'group_id': ticket.get('group_id'),
            'company_id': ticket.get('company_id'),
            'requester_id': ticket.get('requester_id'),
            'responder_id': ticket.get('responder_id'),
            'created_at': ticket.get('created_at'),
            'updated_at': ticket.get('updated_at'),
            'resolved_at': ticket.get('stats', {}).get('resolved_at') if ticket.get('stats') else None,
            'resolution_time_hours': ticket.get('resolution_time_hours'),
            'response_count': ticket.get('response_count'),
            'requester_email': ticket.get('requester', {}).get('email') if ticket.get('requester') else None,
            'requester_name': ticket.get('requester', {}).get('name') if ticket.get('requester') else None,
            'company_name': ticket.get('company', {}).get('name') if ticket.get('company') else None,
            'conversations_preview': ' ||| '.join(conv_preview)
        }
    
    # =========================================================================
    # MAIN EXTRACTION FLOW
    # =========================================================================
    
    def extract(self, days: int = 180, resume: bool = False) -> Dict:
        """Main extraction entry point."""
        
        self.stats['start_time'] = datetime.now(timezone.utc).isoformat()
        
        if self.dashboard:
            self.dashboard.start_time = datetime.now()
        
        # Show banner
        if RICH_AVAILABLE:
            banner = Panel(
                Align.center(
                    Text.from_markup(
                        "[bold white]🚀 Freshdesk Ticket Extractor v2.1[/bold white]\n"
                        "[dim]With checkpoint/resume • Real-time dashboard[/dim]"
                    )
                ),
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(banner)
        else:
            print("\n" + "="*50)
            print("  Freshdesk Ticket Extractor v2.1")
            print("  With checkpoint/resume support")
            print("="*50)
        
        # Check for resume
        ticket_ids = None
        if resume:
            ticket_ids = self._load_ticket_ids()
            if ticket_ids:
                self._print(f"[yellow]📌 Resuming with {len(ticket_ids):,} previously discovered tickets[/yellow]")
        
        # Phase 1: Discover tickets
        if not ticket_ids:
            if not self.group_id:
                self._print("[red]Error: --group-id required for discovery[/red]")
                return {}
            ticket_ids = self.discover_tickets(days, self.group_id)
        
        if self.dashboard:
            self.dashboard.tickets_total = len(ticket_ids)
        
        # Phase 2: Process tickets
        self.process_tickets(ticket_ids)
        
        # Phase 3: Finalize
        result = self.finalize()
        
        # Show summary
        self._show_summary()
        
        return result
    
    def _show_summary(self):
        """Display extraction summary"""
        if RICH_AVAILABLE:
            # Calculate duration
            duration_str = "--"
            if self.stats['start_time'] and self.stats['end_time']:
                start = datetime.fromisoformat(self.stats['start_time'])
                end = datetime.fromisoformat(self.stats['end_time'])
                duration = end - start
                hours, remainder = divmod(int(duration.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{hours}h {minutes}m {seconds}s"
            
            # Summary table
            table = Table(
                title="✅ Extraction Complete",
                box=box.ROUNDED,
                title_style="bold green",
                border_style="green"
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold white", justify="right")
            
            table.add_row("📋 Tickets Discovered", f"{self.stats['tickets_discovered']:,}")
            table.add_row("✅ Tickets Processed", f"{self.stats['tickets_processed']:,}")
            table.add_row("💬 Conversations", f"{self.stats['conversations_fetched']:,}")
            table.add_row("📎 Attachments", f"{self.stats['attachments_downloaded']:,}")
            table.add_row("❌ Errors", f"{self.stats['errors']}")
            table.add_row("⏱️  Duration", duration_str)
            
            self.console.print("\n")
            self.console.print(table)
            
            # Output location
            self.console.print(f"\n[bold]📁 Output Location:[/bold] [cyan]{self.output_dir.absolute()}[/cyan]")
            self.console.print(f"   • tickets.json - Full data with conversations")
            self.console.print(f"   • tickets.csv  - Flattened for analysis")
            self.console.print(f"   • tickets/     - Individual ticket files")
            self.console.print()
        else:
            print("\n" + "="*50)
            print("EXTRACTION COMPLETE")
            print("="*50)
            print(f"Tickets Discovered:  {self.stats['tickets_discovered']:,}")
            print(f"Tickets Processed:   {self.stats['tickets_processed']:,}")
            print(f"Conversations:       {self.stats['conversations_fetched']:,}")
            print(f"Attachments:         {self.stats['attachments_downloaded']:,}")
            print(f"Errors:              {self.stats['errors']}")
            print(f"Output:              {self.output_dir}")
            print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Freshdesk Ticket Extractor v2.1 - Advanced extraction with live dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Fresh extraction:
    python3 freshdesk_extractor_v2.py --api-key YOUR_KEY --days 180 --group-id YOUR_GROUP_ID

  Resume interrupted extraction:
    python3 freshdesk_extractor_v2.py --api-key YOUR_KEY --resume

  Skip attachments (faster):
    python3 freshdesk_extractor_v2.py --api-key YOUR_KEY --days 180 --no-attachments

Environment Variables:
  FRESHDESK_DOMAIN    - Your Freshdesk subdomain
  FRESHDESK_API_KEY   - Your API key (alternative to --api-key)
  FRESHDESK_GROUP_ID  - Default group ID to filter
        """
    )
    
    parser.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY, help='Freshdesk API key (or set FRESHDESK_API_KEY)')
    parser.add_argument('--domain', '-d', default=FRESHDESK_DOMAIN, help=f'Freshdesk domain (default: {FRESHDESK_DOMAIN})')
    parser.add_argument('--days', type=int, default=180, help='Days to look back (default: 180)')
    parser.add_argument('--output', '-o', default='output', help='Output directory (default: output)')
    parser.add_argument('--group-id', type=int, default=int(FRESHDESK_GROUP_ID) if FRESHDESK_GROUP_ID else None, help='Filter by group ID')
    parser.add_argument('--rate-limit', type=int, default=40, help='Max requests/minute (default: 40)')
    parser.add_argument('--no-attachments', action='store_true', help='Skip downloading attachments')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='Test connection only')
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        parser.error("API key required. Use --api-key or set FRESHDESK_API_KEY environment variable.")
    
    if args.domain == 'your-domain':
        parser.error("Domain required. Use --domain or set FRESHDESK_DOMAIN environment variable.")
    
    extractor = FreshdeskExtractorV2(
        domain=args.domain,
        api_key=args.api_key,
        output_dir=args.output,
        requests_per_minute=args.rate_limit,
        download_attachments=not args.no_attachments,
        group_id=args.group_id
    )
    
    if not extractor.test_connection():
        sys.exit(1)
    
    if args.test_only:
        sys.exit(0)
    
    try:
        extractor.extract(days=args.days, resume=args.resume)
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            Console().print("\n[yellow]⚠️  Interrupted! Progress saved. Use --resume to continue.[/yellow]\n")
        else:
            print("\n⚠️  Interrupted! Progress saved. Use --resume to continue.\n")
        sys.exit(0)


if __name__ == '__main__':
    main()