#!/usr/bin/env python3
"""
Freshdesk Ticket Extractor (Legacy v1)
======================================
Extracts tickets, conversations, and attachments from Freshdesk API
with rate limiting, pagination, and robust error handling.

Note: Consider using freshdesk_extractor_v2.py for improved features.

Usage:
    python freshdesk_extractor.py --api-key YOUR_API_KEY --days 180

Output structure:
    output/
    ├── tickets.json          # All ticket data with conversations
    ├── tickets.csv           # Flattened ticket data for analysis
    ├── attachments/
    │   ├── {ticket_id}/
    │   │   ├── {attachment_name}
    │   │   └── ...
    └── extraction_log.json   # Extraction metadata and stats
"""

import argparse
import csv
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin
import html

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('freshdesk_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Track rate limit status"""
    total: int = 4000
    remaining: int = 4000
    used_current: int = 0
    last_updated: datetime = None
    
    def update_from_headers(self, headers: Dict):
        """Update state from response headers"""
        self.total = int(headers.get('x-ratelimit-total', self.total))
        self.remaining = int(headers.get('x-ratelimit-remaining', self.remaining))
        self.used_current = int(headers.get('x-ratelimit-used-currentrequest', 1))
        self.last_updated = datetime.now()


@dataclass
class ExtractionStats:
    """Track extraction statistics"""
    tickets_fetched: int = 0
    conversations_fetched: int = 0
    attachments_downloaded: int = 0
    errors: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    def to_dict(self):
        return {
            'tickets_fetched': self.tickets_fetched,
            'conversations_fetched': self.conversations_fetched,
            'attachments_downloaded': self.attachments_downloaded,
            'errors': self.errors,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None
        }


class FreshdeskExtractor:
    """
    Robust Freshdesk data extractor with rate limiting and error handling.
    """
    
    BASE_URL = "https://{domain}.freshdesk.com/api/v2"
    
    # Group ID cache
    _groups_cache = None
    
    # Status mapping
    STATUS_MAP = {
        2: 'Open',
        3: 'Pending',
        4: 'Resolved',
        5: 'Closed'
    }
    
    # Priority mapping  
    PRIORITY_MAP = {
        1: 'Low',
        2: 'Medium',
        3: 'High',
        4: 'Urgent'
    }
    
    # Source mapping
    SOURCE_MAP = {
        1: 'Email',
        2: 'Portal',
        3: 'Phone',
        7: 'Chat',
        9: 'Feedback Widget',
        10: 'Outbound Email'
    }
    
    def __init__(
        self,
        domain: str,
        api_key: str,
        output_dir: str = "output",
        requests_per_minute: int = 40,  # Conservative limit (actual is ~67/min for 4000/hr)
        download_attachments: bool = True,
        group_id: Optional[int] = None,
        group_name: Optional[str] = None
    ):
        self.domain = domain
        self.api_key = api_key
        self.base_url = self.BASE_URL.format(domain=domain)
        self.output_dir = Path(output_dir)
        self.attachments_dir = self.output_dir / "attachments"
        self.requests_per_minute = requests_per_minute
        self.download_attachments = download_attachments
        self.group_id = group_id
        self.group_name = group_name
        
        # Rate limiting
        self.min_request_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.rate_limit_state = RateLimitState()
        
        # Stats
        self.stats = ExtractionStats()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(api_key, 'X')
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        
        # If we're running low on rate limit, wait longer
        if self.rate_limit_state.remaining < 100:
            logger.warning(f"Rate limit low ({self.rate_limit_state.remaining} remaining), waiting 60s...")
            time.sleep(60)
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        retry_count: int = 3
    ) -> Optional[requests.Response]:
        """Make a rate-limited API request with retry logic"""
        url = urljoin(self.base_url + '/', endpoint)
        
        for attempt in range(retry_count):
            self._wait_for_rate_limit()
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                self.last_request_time = time.time()
                
                # Update rate limit state
                self.rate_limit_state.update_from_headers(response.headers)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s before retry...")
                    time.sleep(retry_after)
                    continue
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.error(f"Request failed: {response.status_code} - {response.text[:200]}")
                    if attempt < retry_count - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                self.stats.errors += 1
                return None
        
        return None
    
    def test_connection(self) -> bool:
        """Test API connectivity and authentication"""
        logger.info("Testing API connection...")
        response = self._make_request('tickets', params={'per_page': 1})
        
        if response:
            logger.info(f"✓ Connection successful. Rate limit: {self.rate_limit_state.remaining}/{self.rate_limit_state.total}")
            return True
        else:
            logger.error("✗ Connection failed. Check your API key and domain.")
            return False
    
    def get_groups(self) -> List[Dict]:
        """Fetch all groups from Freshdesk"""
        if self._groups_cache is not None:
            return self._groups_cache
        
        logger.info("Fetching groups...")
        response = self._make_request('groups', params={'per_page': 100})
        
        if response:
            self._groups_cache = response.json()
            return self._groups_cache
        return []
    
    def list_groups(self) -> None:
        """Print available groups"""
        groups = self.get_groups()
        print("\nAvailable Groups:")
        print("-" * 50)
        for group in groups:
            print(f"  ID: {group['id']:15} | Name: {group['name']}")
        print("-" * 50)
    
    def resolve_group_id(self) -> Optional[int]:
        """Resolve group name to group ID if needed"""
        if self.group_id:
            return self.group_id
        
        if self.group_name:
            groups = self.get_groups()
            for group in groups:
                if group['name'].lower() == self.group_name.lower():
                    logger.info(f"Resolved group '{self.group_name}' to ID: {group['id']}")
                    self.group_id = group['id']
                    return self.group_id
            
            # Partial match
            for group in groups:
                if self.group_name.lower() in group['name'].lower():
                    logger.info(f"Resolved group '{self.group_name}' to '{group['name']}' (ID: {group['id']})")
                    self.group_id = group['id']
                    return self.group_id
            
            logger.error(f"Group '{self.group_name}' not found. Use --list-groups to see available groups.")
            return None
        
        return None
    
    def get_tickets(
        self,
        updated_since: Optional[datetime] = None,
        per_page: int = 100
    ) -> List[Dict]:
        """
        Fetch all tickets updated since a given date.
        Uses search API if filtering by group, otherwise standard API.
        """
        # Resolve group ID if group name was provided
        target_group_id = self.resolve_group_id()
        
        if target_group_id:
            return self._get_tickets_by_group(target_group_id, updated_since)
        else:
            return self._get_tickets_standard(updated_since, per_page)
    
    def _get_tickets_by_group(
        self,
        group_id: int,
        updated_since: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Fetch tickets filtered by group using Search API with date chunking.
        Chunks by week to stay under 300 results per query.
        """
        from datetime import timezone
        
        all_tickets = []
        seen_ids = set()
        
        # Define date range
        end_date = datetime.now(timezone.utc)
        start_date = updated_since or (end_date - timedelta(days=180))
        
        # Create weekly chunks (working backwards from today)
        chunks = []
        chunk_end = end_date
        while chunk_end > start_date:
            chunk_start = max(chunk_end - timedelta(days=7), start_date)
            chunks.append((chunk_start, chunk_end))
            chunk_end = chunk_start
        
        logger.info(f"Searching {len(chunks)} weekly chunks for group {group_id}")
        
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
            start_str = chunk_start.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')
            
            # Search API query - note: updated_at uses >= and <= semantics with > and <
            query = f'"group_id:{group_id} AND updated_at:>\'{start_str}\' AND updated_at:<\'{end_str}\'"'
            
            logger.info(f"Chunk {chunk_idx}/{len(chunks)}: {start_str} to {end_str}")
            
            page = 1
            chunk_tickets = 0
            
            while page <= 10:  # Search API max 10 pages
                response = self._make_request(
                    'search/tickets',
                    params={'query': query, 'page': page}
                )
                
                if not response:
                    break
                
                data = response.json()
                tickets = data.get('results', [])
                
                if not tickets:
                    break
                
                # Deduplicate
                for ticket in tickets:
                    if ticket['id'] not in seen_ids:
                        seen_ids.add(ticket['id'])
                        all_tickets.append(ticket)
                        chunk_tickets += 1
                
                if len(tickets) < 30:
                    break
                
                page += 1
            
            logger.info(f"  Found {chunk_tickets} tickets (total: {len(all_tickets)})")
            
            # If we hit 300 in a week, split that week into days
            if page > 10:
                logger.warning(f"  Week {start_str} to {end_str} has 300+ tickets, fetching daily...")
                all_tickets, seen_ids = self._fetch_daily_chunk(
                    group_id, chunk_start, chunk_end, all_tickets, seen_ids
                )
        
        self.stats.tickets_fetched = len(all_tickets)
        logger.info(f"Total unique tickets found: {len(all_tickets)}")
        
        # Fetch full ticket details (search returns limited fields)
        logger.info("Fetching full ticket details...")
        detailed_tickets = []
        for i, ticket in enumerate(all_tickets, 1):
            if i % 100 == 0:
                logger.info(f"Fetching details: {i}/{len(all_tickets)}")
            
            response = self._make_request(
                f"tickets/{ticket['id']}",
                params={'include': 'requester,company,stats'}
            )
            if response:
                detailed_tickets.append(response.json())
            else:
                detailed_tickets.append(ticket)
        
        return detailed_tickets
    
    def _fetch_daily_chunk(
        self,
        group_id: int,
        start_date: datetime,
        end_date: datetime,
        all_tickets: List[Dict],
        seen_ids: set
    ) -> tuple:
        """Fetch tickets day by day for busy weeks"""
        current = start_date
        while current < end_date:
            next_day = current + timedelta(days=1)
            start_str = current.strftime('%Y-%m-%d')
            end_str = next_day.strftime('%Y-%m-%d')
            
            query = f'"group_id:{group_id} AND updated_at:>\'{start_str}\' AND updated_at:<\'{end_str}\'"'
            
            page = 1
            day_count = 0
            
            while page <= 10:
                response = self._make_request(
                    'search/tickets',
                    params={'query': query, 'page': page}
                )
                
                if not response:
                    break
                
                data = response.json()
                tickets = data.get('results', [])
                
                if not tickets:
                    break
                
                for ticket in tickets:
                    if ticket['id'] not in seen_ids:
                        seen_ids.add(ticket['id'])
                        all_tickets.append(ticket)
                        day_count += 1
                
                if len(tickets) < 30:
                    break
                
                page += 1
            
            if day_count > 0:
                logger.info(f"    {start_str}: {day_count} tickets")
            
            current = next_day
        
        return all_tickets, seen_ids
    
    def _get_tickets_standard(
        self,
        updated_since: Optional[datetime] = None,
        per_page: int = 100
    ) -> List[Dict]:
        """Fetch tickets using standard API (no group filter)"""
        all_tickets = []
        page = 1
        
        params = {
            'per_page': per_page,
            'include': 'requester,company,stats',
            'order_by': 'updated_at',
            'order_type': 'desc'
        }
        
        if updated_since:
            params['updated_since'] = updated_since.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        logger.info(f"Fetching tickets updated since {updated_since}...")
        
        while True:
            params['page'] = page
            response = self._make_request('tickets', params=params)
            
            if not response:
                break
            
            tickets = response.json()
            
            if not tickets:
                break
            
            all_tickets.extend(tickets)
            self.stats.tickets_fetched += len(tickets)
            
            logger.info(f"Fetched page {page}: {len(tickets)} tickets (total: {len(all_tickets)})")
            
            if len(tickets) < per_page:
                break
            
            page += 1
            
            if page > 100:
                logger.warning("Reached page limit (100). Stopping pagination.")
                break
        
        return all_tickets
    
    def get_conversations(self, ticket_id: int) -> List[Dict]:
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
            self.stats.conversations_fetched += len(conversations)
            
            if len(conversations) < 100:
                break
            
            page += 1
        
        return all_conversations
    
    def download_attachment(self, ticket_id: int, attachment: Dict) -> Optional[str]:
        """Download an attachment and return local path"""
        if not self.download_attachments:
            return None
        
        attachment_url = attachment.get('attachment_url')
        attachment_name = attachment.get('name', f"attachment_{attachment.get('id')}")
        
        if not attachment_url:
            return None
        
        # Create ticket attachment directory
        ticket_dir = self.attachments_dir / str(ticket_id)
        ticket_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_name = "".join(c for c in attachment_name if c.isalnum() or c in '._- ')
        local_path = ticket_dir / safe_name
        
        # Skip if already downloaded
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
                self.stats.attachments_downloaded += 1
                return str(local_path)
            else:
                logger.warning(f"Failed to download attachment: {attachment_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading attachment: {e}")
            self.stats.errors += 1
            return None
    
    def process_ticket(self, ticket: Dict) -> Dict:
        """
        Process a single ticket: fetch conversations, download attachments,
        and return enriched ticket data.
        """
        ticket_id = ticket['id']
        
        # Fetch conversations
        conversations = self.get_conversations(ticket_id)
        
        # Process attachments from ticket
        ticket_attachments = ticket.get('attachments', [])
        for att in ticket_attachments:
            local_path = self.download_attachment(ticket_id, att)
            if local_path:
                att['local_path'] = local_path
        
        # Process attachments from conversations
        for conv in conversations:
            conv_attachments = conv.get('attachments', [])
            for att in conv_attachments:
                local_path = self.download_attachment(ticket_id, att)
                if local_path:
                    att['local_path'] = local_path
        
        # Add conversations to ticket
        ticket['conversations'] = conversations
        
        # Add derived fields
        ticket['status_name'] = self.STATUS_MAP.get(ticket.get('status'), 'Unknown')
        ticket['priority_name'] = self.PRIORITY_MAP.get(ticket.get('priority'), 'Unknown')
        ticket['source_name'] = self.SOURCE_MAP.get(ticket.get('source'), 'Unknown')
        
        # Calculate resolution time if resolved
        if ticket.get('stats') and ticket['stats'].get('resolved_at'):
            created = datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00'))
            resolved = datetime.fromisoformat(ticket['stats']['resolved_at'].replace('Z', '+00:00'))
            ticket['resolution_time_hours'] = (resolved - created).total_seconds() / 3600
        
        # Response count
        ticket['response_count'] = len(conversations)
        
        return ticket
    
    def flatten_ticket_for_csv(self, ticket: Dict) -> Dict:
        """Flatten ticket data for CSV export"""
        
        # Clean HTML from description
        description = ticket.get('description', '') or ''
        # Simple HTML tag removal
        clean_desc = html.unescape(description)
        clean_desc = ''.join(c if c.isprintable() or c in '\n\t' else ' ' for c in clean_desc)
        
        # Combine conversation bodies
        conv_text = []
        for conv in ticket.get('conversations', []):
            body = conv.get('body_text') or conv.get('body', '')
            if body:
                direction = 'Incoming' if conv.get('incoming') else 'Outgoing'
                conv_text.append(f"[{direction}] {body[:500]}")
        
        return {
            'id': ticket.get('id'),
            'subject': ticket.get('subject'),
            'description': clean_desc[:2000],  # Truncate for CSV
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
            'first_response_time_hours': ticket.get('stats', {}).get('first_responded_at') if ticket.get('stats') else None,
            'requester_email': ticket.get('requester', {}).get('email') if ticket.get('requester') else None,
            'requester_name': ticket.get('requester', {}).get('name') if ticket.get('requester') else None,
            'company_name': ticket.get('company', {}).get('name') if ticket.get('company') else None,
            'conversations_preview': ' ||| '.join(conv_text[:3])  # First 3 conversations
        }
    
    def extract(self, days: int = 180) -> Dict:
        """
        Main extraction method.
        Fetches tickets from the last N days, enriches with conversations and attachments.
        """
        from datetime import timezone
        self.stats.start_time = datetime.now()
        
        # Calculate date range
        updated_since = datetime.now(timezone.utc) - timedelta(days=days)
        
        logger.info(f"Starting extraction for last {days} days...")
        logger.info(f"Extracting tickets updated since: {updated_since}")
        
        # Fetch all tickets
        tickets = self.get_tickets(updated_since=updated_since)
        logger.info(f"Found {len(tickets)} tickets to process")
        
        # Process each ticket
        processed_tickets = []
        for i, ticket in enumerate(tickets, 1):
            logger.info(f"Processing ticket {i}/{len(tickets)}: #{ticket['id']} - {ticket.get('subject', '')[:50]}")
            processed_ticket = self.process_ticket(ticket)
            processed_tickets.append(processed_ticket)
            
            # Progress checkpoint every 100 tickets
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(tickets)} tickets processed")
                self._save_checkpoint(processed_tickets)
        
        self.stats.end_time = datetime.now()
        
        # Save results
        self._save_results(processed_tickets)
        
        return {
            'tickets': processed_tickets,
            'stats': self.stats.to_dict()
        }
    
    def _save_checkpoint(self, tickets: List[Dict]):
        """Save intermediate checkpoint"""
        checkpoint_path = self.output_dir / 'checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump({'tickets_count': len(tickets), 'timestamp': datetime.now().isoformat()}, f)
    
    def _save_results(self, tickets: List[Dict]):
        """Save final results to files"""
        
        # Save full JSON
        json_path = self.output_dir / 'tickets.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved {len(tickets)} tickets to {json_path}")
        
        # Save flattened CSV
        csv_path = self.output_dir / 'tickets.csv'
        if tickets:
            flattened = [self.flatten_ticket_for_csv(t) for t in tickets]
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
            logger.info(f"Saved flattened data to {csv_path}")
        
        # Save extraction log
        log_path = self.output_dir / 'extraction_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'stats': self.stats.to_dict(),
                'rate_limit': {
                    'total': self.rate_limit_state.total,
                    'remaining': self.rate_limit_state.remaining
                },
                'output_files': {
                    'json': str(json_path),
                    'csv': str(csv_path),
                    'attachments_dir': str(self.attachments_dir)
                }
            }, f, indent=2)
        logger.info(f"Saved extraction log to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract tickets, conversations, and attachments from Freshdesk',
        epilog="""
Environment Variables:
  FRESHDESK_DOMAIN    - Your Freshdesk subdomain
  FRESHDESK_API_KEY   - Your API key (alternative to --api-key)
  FRESHDESK_GROUP_ID  - Default group ID to filter
        """
    )
    parser.add_argument(
        '--api-key', '-k',
        default=FRESHDESK_API_KEY,
        help='Freshdesk API key (or set FRESHDESK_API_KEY)'
    )
    parser.add_argument(
        '--domain', '-d',
        default=FRESHDESK_DOMAIN,
        help=f'Freshdesk domain (default: {FRESHDESK_DOMAIN})'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=180,
        help='Number of days to look back (default: 180)'
    )
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=40,
        help='Max requests per minute (default: 40)'
    )
    parser.add_argument(
        '--no-attachments',
        action='store_true',
        help='Skip downloading attachments'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only test API connection, don\'t extract data'
    )
    parser.add_argument(
        '--group', '-g',
        help='Filter by group name (e.g., "Digital Logs")'
    )
    parser.add_argument(
        '--group-id',
        type=int,
        default=int(FRESHDESK_GROUP_ID) if FRESHDESK_GROUP_ID else None,
        help='Filter by group ID (use --list-groups to find IDs)'
    )
    parser.add_argument(
        '--list-groups',
        action='store_true',
        help='List all available groups and exit'
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        parser.error("API key required. Use --api-key or set FRESHDESK_API_KEY environment variable.")
    
    if args.domain == 'your-domain':
        parser.error("Domain required. Use --domain or set FRESHDESK_DOMAIN environment variable.")
    
    extractor = FreshdeskExtractor(
        domain=args.domain,
        api_key=args.api_key,
        output_dir=args.output,
        requests_per_minute=args.rate_limit,
        download_attachments=not args.no_attachments,
        group_id=args.group_id,
        group_name=args.group
    )
    
    # Test connection
    if not extractor.test_connection():
        sys.exit(1)
    
    # List groups and exit
    if args.list_groups:
        extractor.list_groups()
        sys.exit(0)
    
    if args.test_only:
        logger.info("Test mode - exiting without extraction")
        sys.exit(0)
    
    # Run extraction
    result = extractor.extract(days=args.days)
    
    # Print summary
    stats = result['stats']
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Tickets fetched:      {stats['tickets_fetched']}")
    print(f"Conversations:        {stats['conversations_fetched']}")
    print(f"Attachments:          {stats['attachments_downloaded']}")
    print(f"Errors:               {stats['errors']}")
    print(f"Duration:             {stats['duration_seconds']:.1f} seconds")
    print(f"Output directory:     {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()