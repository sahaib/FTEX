#!/usr/bin/env python3
"""
Actionable Ticket Report Generator
===================================
Generates reports with specific ticket IDs for each finding.
Output: Excel file with multiple sheets, each actionable.

Usage:
    pip3 install pandas openpyxl
    python3 generate_actionable_report.py --input output/tickets.json
"""

import json
import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("❌ Install pandas: pip3 install pandas openpyxl")
    exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    console = None

# Configuration
FRESHDESK_DOMAIN = os.getenv('FRESHDESK_DOMAIN', 'your-domain')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_tickets(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_freshdesk_url(ticket_id):
    """Generate Freshdesk ticket URL"""
    return f"https://{FRESHDESK_DOMAIN}.freshdesk.com/a/tickets/{ticket_id}"


def generate_actionable_report(tickets, output_path="actionable_report.xlsx"):
    """Generate Excel with multiple actionable sheets"""
    
    print(f"\n📊 Analyzing {len(tickets):,} tickets...\n")
    
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    
    # ==========================================================================
    # 1. ZOMBIE TICKETS - No Response (580 tickets)
    # ==========================================================================
    print("1️⃣  Finding zombie tickets (no response)...")
    
    zombie_tickets = []
    for t in tickets:
        convos = t.get('conversations', [])
        if len(convos) == 0:
            zombie_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Priority': t.get('priority_name', 'Unknown'),
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': t.get('created_at', '')[:10],
                'Days Open': None,
                'Requester Email': t.get('requester', {}).get('email', '') if t.get('requester') else ''
            })
    
    # Calculate days open
    for z in zombie_tickets:
        if z['Created']:
            try:
                created = datetime.fromisoformat(z['Created'])
                z['Days Open'] = (datetime.now() - created).days
            except:
                pass
    
    df_zombie = pd.DataFrame(zombie_tickets)
    df_zombie = df_zombie.sort_values('Days Open', ascending=False)
    df_zombie.to_excel(writer, sheet_name='1_No_Response_Zombies', index=False)
    print(f"   ✅ {len(zombie_tickets)} zombie tickets")
    
    # ==========================================================================
    # 2. EXTREME RESOLUTION TIME (Top 100 longest)
    # ==========================================================================
    print("2️⃣  Finding extreme resolution time tickets...")
    
    long_resolution = []
    for t in tickets:
        res_hours = t.get('resolution_time_hours')
        if res_hours and res_hours > 500:  # > 20 days
            long_resolution.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Resolution Hours': round(res_hours, 1),
                'Resolution Days': round(res_hours / 24, 1),
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': t.get('created_at', '')[:10],
                'Resolved': t.get('stats', {}).get('resolved_at', '')[:10] if t.get('stats') else ''
            })
    
    df_long = pd.DataFrame(long_resolution)
    df_long = df_long.sort_values('Resolution Hours', ascending=False)
    df_long.to_excel(writer, sheet_name='2_Long_Resolution_500h+', index=False)
    print(f"   ✅ {len(long_resolution)} tickets with >500 hours resolution")
    
    # ==========================================================================
    # 3. OPEN/PENDING TICKETS (Need attention)
    # ==========================================================================
    print("3️⃣  Finding open/pending tickets...")
    
    open_pending = []
    for t in tickets:
        status = t.get('status_name', '')
        if status in ['Open', 'Pending']:
            created = t.get('created_at', '')
            days_open = None
            if created:
                try:
                    days_open = (datetime.now() - datetime.fromisoformat(created[:10])).days
                except:
                    pass
            
            open_pending.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': status,
                'Priority': t.get('priority_name', 'Unknown'),
                'Days Open': days_open,
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': created[:10] if created else '',
                'Last Updated': t.get('updated_at', '')[:10],
                'Conversations': len(t.get('conversations', []))
            })
    
    df_open = pd.DataFrame(open_pending)
    df_open = df_open.sort_values('Days Open', ascending=False)
    df_open.to_excel(writer, sheet_name='3_Open_Pending_Tickets', index=False)
    print(f"   ✅ {len(open_pending)} open/pending tickets")
    
    # ==========================================================================
    # 4. LICENSE/UPDATE TICKETS (Automation candidates)
    # ==========================================================================
    print("4️⃣  Finding license/update tickets (automation candidates)...")
    
    license_keywords = ['license', 'renewal', 'extension', 'update', 'application update']
    license_tickets = []
    
    for t in tickets:
        subject = (t.get('subject', '') or '').lower()
        if any(kw in subject for kw in license_keywords):
            license_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': t.get('created_at', '')[:10],
                'Resolution Hours': t.get('resolution_time_hours'),
                'Keyword Match': next((kw for kw in license_keywords if kw in subject), '')
            })
    
    df_license = pd.DataFrame(license_tickets)
    df_license.to_excel(writer, sheet_name='4_License_Update_Automate', index=False)
    print(f"   ✅ {len(license_tickets)} license/update tickets")
    
    # ==========================================================================
    # 5. ONBOARDING/GO-LIVE TICKETS
    # ==========================================================================
    print("5️⃣  Finding onboarding/go-live tickets...")
    
    onboarding_keywords = ['go-live', 'golive', 'go live', 'onboarding', 'pre-live', 'prelive', 'reset & live', 'reset and live']
    onboarding_tickets = []
    
    for t in tickets:
        subject = (t.get('subject', '') or '').lower()
        if any(kw in subject for kw in onboarding_keywords):
            onboarding_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': t.get('created_at', '')[:10],
                'Resolution Hours': t.get('resolution_time_hours'),
                'Resolution Days': round(t.get('resolution_time_hours', 0) / 24, 1) if t.get('resolution_time_hours') else None
            })
    
    df_onboard = pd.DataFrame(onboarding_tickets)
    df_onboard.to_excel(writer, sheet_name='5_Onboarding_GoLive', index=False)
    print(f"   ✅ {len(onboarding_tickets)} onboarding/go-live tickets")
    
    # ==========================================================================
    # 6. TOP CUSTOMERS - Tickets by Company
    # ==========================================================================
    print("6️⃣  Grouping by company...")
    
    company_tickets = defaultdict(list)
    for t in tickets:
        company = t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown'
        company_tickets[company].append(t)
    
    # Sort by count
    company_summary = []
    for company, tix in sorted(company_tickets.items(), key=lambda x: len(x[1]), reverse=True)[:50]:
        open_count = len([t for t in tix if t.get('status_name') in ['Open', 'Pending']])
        res_times = [t.get('resolution_time_hours') for t in tix if t.get('resolution_time_hours')]
        avg_res = sum(res_times) / len(res_times) if res_times else 0
        
        company_summary.append({
            'Company': company,
            'Total Tickets': len(tix),
            'Open/Pending': open_count,
            'Avg Resolution Hours': round(avg_res, 1),
            'Avg Resolution Days': round(avg_res / 24, 1),
            'Sample Ticket IDs': ', '.join([str(t.get('id')) for t in tix[:10]])
        })
    
    df_company = pd.DataFrame(company_summary)
    df_company.to_excel(writer, sheet_name='6_Top_Companies', index=False)
    print(f"   ✅ {len(company_summary)} companies analyzed")
    
    # ==========================================================================
    # 7. STOLT TANKERS - Deep Dive (Top Customer)
    # ==========================================================================
    print("7️⃣  Deep dive: Stolt Tankers...")
    
    stolt_tickets = []
    for t in tickets:
        company = t.get('company', {}).get('name', '') if t.get('company') else ''
        if 'stolt' in company.lower():
            stolt_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Priority': t.get('priority_name', 'Unknown'),
                'Created': t.get('created_at', '')[:10],
                'Resolution Hours': t.get('resolution_time_hours'),
                'Conversations': len(t.get('conversations', []))
            })
    
    df_stolt = pd.DataFrame(stolt_tickets)
    df_stolt.to_excel(writer, sheet_name='7_Stolt_Tankers_DeepDive', index=False)
    print(f"   ✅ {len(stolt_tickets)} Stolt Tankers tickets")
    
    # ==========================================================================
    # 8. NYK - Deep Dive (Second Top Customer)
    # ==========================================================================
    print("8️⃣  Deep dive: NYK...")
    
    nyk_tickets = []
    for t in tickets:
        company = t.get('company', {}).get('name', '') if t.get('company') else ''
        if 'nyk' in company.lower():
            nyk_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Priority': t.get('priority_name', 'Unknown'),
                'Created': t.get('created_at', '')[:10],
                'Resolution Hours': t.get('resolution_time_hours'),
                'Conversations': len(t.get('conversations', []))
            })
    
    df_nyk = pd.DataFrame(nyk_tickets)
    df_nyk.to_excel(writer, sheet_name='8_NYK_DeepDive', index=False)
    print(f"   ✅ {len(nyk_tickets)} NYK tickets")
    
    # ==========================================================================
    # 9. DUPLICATE COMPANY NAMES (Data cleanup)
    # ==========================================================================
    print("9️⃣  Finding duplicate company names...")
    
    # Normalize company names and find potential duplicates
    company_names = [t.get('company', {}).get('name', '') if t.get('company') else '' for t in tickets]
    company_counts = Counter(company_names)
    
    # Find similar names
    duplicates = []
    seen = set()
    sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
    
    for name1, count1 in sorted_companies:
        if not name1 or name1 in seen:
            continue
        normalized1 = re.sub(r'[^a-z0-9]', '', name1.lower())
        
        for name2, count2 in sorted_companies:
            if name1 == name2 or not name2 or name2 in seen:
                continue
            normalized2 = re.sub(r'[^a-z0-9]', '', name2.lower())
            
            # Check similarity
            if normalized1 in normalized2 or normalized2 in normalized1:
                if len(normalized1) > 5 and len(normalized2) > 5:
                    duplicates.append({
                        'Company Name 1': name1,
                        'Tickets 1': count1,
                        'Company Name 2': name2,
                        'Tickets 2': count2,
                        'Combined Total': count1 + count2,
                        'Action': 'MERGE'
                    })
                    seen.add(name2)
    
    df_dupes = pd.DataFrame(duplicates)
    df_dupes.to_excel(writer, sheet_name='9_Duplicate_Companies', index=False)
    print(f"   ✅ {len(duplicates)} potential duplicate company pairs")
    
    # ==========================================================================
    # 10. OVERDUE TAGGED TICKETS
    # ==========================================================================
    print("🔟 Finding overdue-tagged tickets...")
    
    overdue_tickets = []
    for t in tickets:
        tags = t.get('tags', [])
        if 'Overdue' in tags or 'overdue' in [tag.lower() for tag in tags]:
            overdue_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Priority': t.get('priority_name', 'Unknown'),
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': t.get('created_at', '')[:10],
                'Tags': ', '.join(tags)
            })
    
    df_overdue = pd.DataFrame(overdue_tickets)
    df_overdue.to_excel(writer, sheet_name='10_Overdue_Tagged', index=False)
    print(f"   ✅ {len(overdue_tickets)} overdue-tagged tickets")
    
    # ==========================================================================
    # 11. WEEKLY PLANNER TICKETS (Recurring work)
    # ==========================================================================
    print("1️⃣1️⃣ Finding weekly planner tickets...")
    
    weekly_tickets = []
    for t in tickets:
        subject = (t.get('subject', '') or '').lower()
        if 'weekly planner' in subject or 'weekly status' in subject:
            weekly_tickets.append({
                'Ticket ID': t.get('id'),
                'URL': get_freshdesk_url(t.get('id')),
                'Subject': t.get('subject', '')[:80],
                'Status': t.get('status_name', 'Unknown'),
                'Company': t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown',
                'Created': t.get('created_at', '')[:10],
            })
    
    df_weekly = pd.DataFrame(weekly_tickets)
    df_weekly.to_excel(writer, sheet_name='11_Weekly_Planner', index=False)
    print(f"   ✅ {len(weekly_tickets)} weekly planner tickets")
    
    # ==========================================================================
    # 12. SUMMARY SHEET
    # ==========================================================================
    print("1️⃣2️⃣ Creating summary sheet...")
    
    summary = [
        {'Category': '🧟 Zombie (No Response)', 'Count': len(zombie_tickets), 'Sheet': '1_No_Response_Zombies', 'Action': 'Triage - close stale or escalate'},
        {'Category': '⏰ Long Resolution (>500h)', 'Count': len(long_resolution), 'Sheet': '2_Long_Resolution_500h+', 'Action': 'Investigate why stuck'},
        {'Category': '📬 Open/Pending', 'Count': len(open_pending), 'Sheet': '3_Open_Pending_Tickets', 'Action': 'Review and action'},
        {'Category': '🔑 License/Update', 'Count': len(license_tickets), 'Sheet': '4_License_Update_Automate', 'Action': 'Automation candidates'},
        {'Category': '🚀 Onboarding/Go-Live', 'Count': len(onboarding_tickets), 'Sheet': '5_Onboarding_GoLive', 'Action': 'Standardize process'},
        {'Category': '🏢 Stolt Tankers', 'Count': len(stolt_tickets), 'Sheet': '7_Stolt_Tankers_DeepDive', 'Action': 'Assign dedicated CSM'},
        {'Category': '🏢 NYK', 'Count': len(nyk_tickets), 'Sheet': '8_NYK_DeepDive', 'Action': 'Assign dedicated CSM'},
        {'Category': '⚠️ Overdue Tagged', 'Count': len(overdue_tickets), 'Sheet': '10_Overdue_Tagged', 'Action': 'Immediate attention'},
        {'Category': '📅 Weekly Planner', 'Count': len(weekly_tickets), 'Sheet': '11_Weekly_Planner', 'Action': 'Automate reporting'},
        {'Category': '🔀 Duplicate Companies', 'Count': len(duplicates), 'Sheet': '9_Duplicate_Companies', 'Action': 'Merge in Freshdesk'},
    ]
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(writer, sheet_name='0_SUMMARY', index=False)
    
    # Save
    writer.close()
    
    print(f"\n✅ Report saved: {output_path}")
    print(f"   📊 12 sheets with actionable ticket IDs")
    
    # Print summary table
    if RICH:
        table = Table(title="📋 Actionable Report Summary", box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right", style="bold white")
        table.add_column("Action", style="yellow")
        
        for row in summary:
            table.add_row(row['Category'], str(row['Count']), row['Action'])
        
        console.print()
        console.print(table)
        console.print(f"\n📁 Open [bold cyan]{output_path}[/bold cyan] in Excel to see ticket IDs\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate actionable report with ticket IDs')
    parser.add_argument('--input', '-i', default='output/tickets.json', help='Path to tickets.json')
    parser.add_argument('--output', '-o', default='actionable_report.xlsx', help='Output Excel file')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ Not found: {args.input}")
        exit(1)
    
    tickets = load_tickets(args.input)
    print(f"✅ Loaded {len(tickets):,} tickets from {args.input}")
    
    generate_actionable_report(tickets, args.output)


if __name__ == '__main__':
    main()