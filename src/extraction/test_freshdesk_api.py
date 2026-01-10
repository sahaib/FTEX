#!/usr/bin/env python3
"""
Freshdesk API Test Script
Tests API connectivity and shows sample response structure.

Usage:
    python test_freshdesk_api.py --api-key YOUR_API_KEY
    python test_freshdesk_api.py --api-key YOUR_API_KEY --domain your-company
"""

import sys
import os
import json
import argparse
import requests
from requests.auth import HTTPBasicAuth

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
DOMAIN = os.getenv('FRESHDESK_DOMAIN', 'your-domain')


def test_api(api_key: str, domain: str = None):
    """Test Freshdesk API and show response structure"""
    
    domain = domain or DOMAIN
    base_url = f"https://{domain}.freshdesk.com/api/v2"
    
    session = requests.Session()
    session.auth = HTTPBasicAuth(api_key, 'X')
    session.headers.update({'Content-Type': 'application/json'})
    
    print("="*60)
    print("FRESHDESK API TEST")
    print(f"Domain: {domain}.freshdesk.com")
    print("="*60)
    
    # Test 1: Get single ticket
    print("\n[1] Testing /tickets endpoint...")
    response = session.get(f"{base_url}/tickets", params={'per_page': 1})
    
    print(f"    Status: {response.status_code}")
    print(f"    Rate Limit Total: {response.headers.get('x-ratelimit-total')}")
    print(f"    Rate Limit Remaining: {response.headers.get('x-ratelimit-remaining')}")
    
    if response.status_code != 200:
        print(f"    ERROR: {response.text[:500]}")
        return False
    
    tickets = response.json()
    if tickets:
        ticket = tickets[0]
        print(f"\n    Sample Ticket Fields:")
        for key in sorted(ticket.keys()):
            val = ticket[key]
            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
            print(f"      - {key}: {val_str}")
    
    # Test 2: Get ticket with includes
    print("\n[2] Testing /tickets with includes...")
    response = session.get(
        f"{base_url}/tickets",
        params={'per_page': 1, 'include': 'requester,company,stats'}
    )
    
    print(f"    Status: {response.status_code}")
    
    if response.status_code == 200:
        tickets = response.json()
        if tickets:
            ticket = tickets[0]
            print(f"\n    Included Fields:")
            if 'requester' in ticket and ticket['requester']:
                print(f"      - requester: {json.dumps(ticket['requester'], indent=8)[:200]}")
            if 'company' in ticket and ticket['company']:
                print(f"      - company: {json.dumps(ticket['company'], indent=8)[:200]}")
            if 'stats' in ticket and ticket['stats']:
                print(f"      - stats: {json.dumps(ticket['stats'], indent=8)[:200]}")
    
    # Test 3: Get conversations for a ticket
    if tickets:
        ticket_id = tickets[0]['id']
        print(f"\n[3] Testing /tickets/{ticket_id}/conversations...")
        response = session.get(f"{base_url}/tickets/{ticket_id}/conversations")
        
        print(f"    Status: {response.status_code}")
        
        if response.status_code == 200:
            conversations = response.json()
            print(f"    Conversations found: {len(conversations)}")
            if conversations:
                conv = conversations[0]
                print(f"\n    Sample Conversation Fields:")
                for key in sorted(conv.keys()):
                    val = conv[key]
                    val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                    print(f"      - {key}: {val_str}")
    
    # Test 4: Count tickets in date range
    print("\n[4] Testing date filter (last 180 days)...")
    from datetime import datetime, timedelta
    date_filter = (datetime.utcnow() - timedelta(days=180)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Get count by fetching with pagination
    response = session.get(
        f"{base_url}/tickets",
        params={'per_page': 100, 'updated_since': date_filter}
    )
    
    if response.status_code == 200:
        first_page = response.json()
        print(f"    First page tickets: {len(first_page)}")
        if len(first_page) == 100:
            print("    (More pages available - full extraction will paginate)")
    
    # Test 5: List groups (to help user find group ID)
    print("\n[5] Listing available groups...")
    response = session.get(f"{base_url}/groups")
    
    if response.status_code == 200:
        groups = response.json()
        print(f"    Found {len(groups)} groups:")
        for g in groups[:10]:
            print(f"      - {g.get('id')}: {g.get('name')}")
        if len(groups) > 10:
            print(f"      ... and {len(groups) - 10} more")
    
    print("\n" + "="*60)
    print("✓ API TEST COMPLETE - Connection successful!")
    print("="*60)
    print("\nYou can now run the full extraction:")
    print(f"  python freshdesk_extractor_v2.py --api-key YOUR_KEY --domain {domain} --days 180")
    print("="*60)
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Freshdesk API connection')
    parser.add_argument('--api-key', '-k', required=True, help='Freshdesk API key')
    parser.add_argument('--domain', '-d', default=None, help=f'Freshdesk domain (default: {DOMAIN})')
    
    # Also support legacy positional argument
    parser.add_argument('api_key_legacy', nargs='?', help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    api_key = args.api_key or args.api_key_legacy
    if not api_key:
        print("Usage: python test_freshdesk_api.py --api-key YOUR_API_KEY")
        sys.exit(1)
    
    success = test_api(api_key, args.domain)
    sys.exit(0 if success else 1)