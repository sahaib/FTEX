#!/usr/bin/env python3
"""
FTEX Advanced Analytics & SLA Report Generator
===============================================
Comprehensive analytics for TLs, Managers, and MDs.

Features:
- SLA compliance analysis (FRT, Resolution)
- Agent performance metrics
- Customer health scoring
- Trend analysis
- Priority-based breakdown
- Time-based patterns

Usage:
    pip3 install pandas openpyxl numpy rich
    python3 generate_sla_report.py --input output/tickets.json
"""

import json
import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    print("❌ Install: pip3 install pandas openpyxl numpy")
    exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
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


# =============================================================================
# SLA CONFIGURATION (Customize these based on your Freshdesk SLA policies)
# =============================================================================
SLA_CONFIG = {
    # First Response Time targets (in hours) by priority
    'first_response': {
        'Urgent': 1,      # 1 hour
        'High': 4,        # 4 hours
        'Medium': 8,      # 8 hours (1 business day)
        'Low': 24,        # 24 hours
    },
    # Resolution Time targets (in hours) by priority
    'resolution': {
        'Urgent': 4,      # 4 hours
        'High': 24,       # 1 day
        'Medium': 72,     # 3 days
        'Low': 168,       # 7 days
    },
    # Business hours (for future enhancement)
    'business_hours': {
        'start': 9,
        'end': 17,
        'timezone': 'UTC'
    }
}


def load_tickets(path: str) -> List[Dict]:
    """Load tickets from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse ISO datetime string"""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except:
        return None


def get_freshdesk_url(ticket_id) -> str:
    """Generate Freshdesk ticket URL"""
    return f"https://{FRESHDESK_DOMAIN}.freshdesk.com/a/tickets/{ticket_id}"


def calculate_first_response_time(ticket: Dict) -> Optional[float]:
    """Calculate time to first agent response in hours"""
    created = parse_datetime(ticket.get('created_at'))
    if not created:
        return None
    
    conversations = ticket.get('conversations', [])
    
    # Find first outgoing (agent) response
    for conv in sorted(conversations, key=lambda x: x.get('created_at', '')):
        if conv.get('incoming') == False:  # Outgoing = agent response
            response_time = parse_datetime(conv.get('created_at'))
            if response_time:
                delta = (response_time - created).total_seconds() / 3600
                return max(0, delta)  # Return hours
    
    return None  # No agent response yet


def calculate_metrics(tickets: List[Dict]) -> Dict:
    """Calculate comprehensive metrics from tickets"""
    
    metrics = {
        'overview': {},
        'sla': {},
        'agents': defaultdict(lambda: {
            'total': 0, 'resolved': 0, 'open': 0,
            'frt_times': [], 'resolution_times': [],
            'frt_breaches': 0, 'resolution_breaches': 0,
            'conversations': []
        }),
        'customers': defaultdict(lambda: {
            'total': 0, 'resolved': 0, 'open': 0, 'pending': 0,
            'frt_times': [], 'resolution_times': [],
            'priorities': Counter(), 'categories': Counter(),
            'escalations': 0, 'reopens': 0
        }),
        'priorities': defaultdict(lambda: {
            'total': 0, 'resolved': 0,
            'frt_times': [], 'resolution_times': [],
            'frt_breaches': 0, 'resolution_breaches': 0
        }),
        'time_analysis': {
            'by_hour': Counter(),
            'by_day': Counter(),
            'by_month': Counter(),
            'by_week': Counter()
        },
        'trends': {
            'monthly_volume': defaultdict(int),
            'monthly_resolved': defaultdict(int),
            'monthly_frt_breach': defaultdict(int),
            'monthly_resolution_breach': defaultdict(int),
            'monthly_avg_resolution': defaultdict(list)
        },
        'escalations': [],
        'aging': {
            '0-1 day': 0,
            '1-3 days': 0,
            '3-7 days': 0,
            '7-14 days': 0,
            '14-30 days': 0,
            '30+ days': 0
        },
        'reopen_candidates': [],
        'top_issues': Counter(),
        'sources': Counter(),
        'tags': Counter()
    }
    
    now = datetime.now()
    total_frt_times = []
    total_resolution_times = []
    total_frt_breaches = 0
    total_resolution_breaches = 0
    resolved_count = 0
    open_count = 0
    pending_count = 0
    
    for ticket in tickets:
        ticket_id = ticket.get('id')
        status = ticket.get('status_name', 'Unknown')
        priority = ticket.get('priority_name', 'Medium')
        created = parse_datetime(ticket.get('created_at'))
        
        # Agent (responder)
        responder = ticket.get('responder_id')
        agent_name = f"Agent_{responder}" if responder else "Unassigned"
        
        # Company
        company = ticket.get('company', {})
        company_name = company.get('name', 'Unknown') if company else 'Unknown'
        
        # Source
        source = ticket.get('source_name', 'Unknown')
        metrics['sources'][source] += 1
        
        # Tags
        for tag in ticket.get('tags', []):
            metrics['tags'][tag] += 1
        
        # Status counts
        if status == 'Resolved' or status == 'Closed':
            resolved_count += 1
            metrics['agents'][agent_name]['resolved'] += 1
            metrics['customers'][company_name]['resolved'] += 1
            metrics['priorities'][priority]['resolved'] += 1
        elif status == 'Open':
            open_count += 1
            metrics['agents'][agent_name]['open'] += 1
            metrics['customers'][company_name]['open'] += 1
        elif status == 'Pending':
            pending_count += 1
            metrics['customers'][company_name]['pending'] += 1
        
        metrics['agents'][agent_name]['total'] += 1
        metrics['customers'][company_name]['total'] += 1
        metrics['customers'][company_name]['priorities'][priority] += 1
        metrics['priorities'][priority]['total'] += 1
        
        # Time analysis
        if created:
            metrics['time_analysis']['by_hour'][created.hour] += 1
            metrics['time_analysis']['by_day'][created.strftime('%A')] += 1
            metrics['time_analysis']['by_month'][created.strftime('%Y-%m')] += 1
            metrics['time_analysis']['by_week'][created.strftime('%Y-W%W')] += 1
            
            month_key = created.strftime('%Y-%m')
            metrics['trends']['monthly_volume'][month_key] += 1
            if status in ['Resolved', 'Closed']:
                metrics['trends']['monthly_resolved'][month_key] += 1
        
        # First Response Time
        frt = calculate_first_response_time(ticket)
        if frt is not None:
            total_frt_times.append(frt)
            metrics['agents'][agent_name]['frt_times'].append(frt)
            metrics['customers'][company_name]['frt_times'].append(frt)
            metrics['priorities'][priority]['frt_times'].append(frt)
            
            # Check SLA breach
            frt_target = SLA_CONFIG['first_response'].get(priority, 24)
            if frt > frt_target:
                total_frt_breaches += 1
                metrics['agents'][agent_name]['frt_breaches'] += 1
                metrics['priorities'][priority]['frt_breaches'] += 1
                if created:
                    metrics['trends']['monthly_frt_breach'][created.strftime('%Y-%m')] += 1
        
        # Resolution Time
        resolution_hours = ticket.get('resolution_time_hours')
        if resolution_hours:
            total_resolution_times.append(resolution_hours)
            metrics['agents'][agent_name]['resolution_times'].append(resolution_hours)
            metrics['customers'][company_name]['resolution_times'].append(resolution_hours)
            metrics['priorities'][priority]['resolution_times'].append(resolution_hours)
            
            if created:
                metrics['trends']['monthly_avg_resolution'][created.strftime('%Y-%m')].append(resolution_hours)
            
            # Check SLA breach
            res_target = SLA_CONFIG['resolution'].get(priority, 168)
            if resolution_hours > res_target:
                total_resolution_breaches += 1
                metrics['agents'][agent_name]['resolution_breaches'] += 1
                metrics['priorities'][priority]['resolution_breaches'] += 1
                if created:
                    metrics['trends']['monthly_resolution_breach'][created.strftime('%Y-%m')] += 1
        
        # Aging (for open/pending tickets)
        if status in ['Open', 'Pending'] and created:
            age_hours = (now - created.replace(tzinfo=None)).total_seconds() / 3600
            age_days = age_hours / 24
            
            if age_days <= 1:
                metrics['aging']['0-1 day'] += 1
            elif age_days <= 3:
                metrics['aging']['1-3 days'] += 1
            elif age_days <= 7:
                metrics['aging']['3-7 days'] += 1
            elif age_days <= 14:
                metrics['aging']['7-14 days'] += 1
            elif age_days <= 30:
                metrics['aging']['14-30 days'] += 1
            else:
                metrics['aging']['30+ days'] += 1
        
        # Conversation count per agent
        convos = ticket.get('conversations', [])
        metrics['agents'][agent_name]['conversations'].append(len(convos))
        
        # Detect potential escalations (high priority + long resolution)
        if priority in ['Urgent', 'High'] and resolution_hours and resolution_hours > 48:
            metrics['escalations'].append({
                'id': ticket_id,
                'subject': ticket.get('subject', '')[:60],
                'priority': priority,
                'resolution_hours': resolution_hours,
                'company': company_name
            })
        
        # Detect potential reopens (resolved but many conversations after resolution)
        # This is a heuristic - adjust based on your workflow
        if status in ['Resolved', 'Closed'] and len(convos) > 10:
            metrics['reopen_candidates'].append({
                'id': ticket_id,
                'subject': ticket.get('subject', '')[:60],
                'conversations': len(convos),
                'company': company_name
            })
    
    # Calculate overview stats
    metrics['overview'] = {
        'total_tickets': len(tickets),
        'resolved': resolved_count,
        'open': open_count,
        'pending': pending_count,
        'resolution_rate': round(resolved_count / len(tickets) * 100, 1) if tickets else 0,
        'unique_companies': len(metrics['customers']),
        'unique_agents': len([a for a in metrics['agents'] if a != 'Unassigned'])
    }
    
    # Calculate SLA stats
    metrics['sla'] = {
        'frt': {
            'total_measured': len(total_frt_times),
            'breaches': total_frt_breaches,
            'compliance_pct': round((1 - total_frt_breaches / len(total_frt_times)) * 100, 1) if total_frt_times else 0,
            'avg_hours': round(np.mean(total_frt_times), 1) if total_frt_times else 0,
            'median_hours': round(np.median(total_frt_times), 1) if total_frt_times else 0,
            'p90_hours': round(np.percentile(total_frt_times, 90), 1) if total_frt_times else 0,
        },
        'resolution': {
            'total_measured': len(total_resolution_times),
            'breaches': total_resolution_breaches,
            'compliance_pct': round((1 - total_resolution_breaches / len(total_resolution_times)) * 100, 1) if total_resolution_times else 0,
            'avg_hours': round(np.mean(total_resolution_times), 1) if total_resolution_times else 0,
            'median_hours': round(np.median(total_resolution_times), 1) if total_resolution_times else 0,
            'p90_hours': round(np.percentile(total_resolution_times, 90), 1) if total_resolution_times else 0,
            'avg_days': round(np.mean(total_resolution_times) / 24, 1) if total_resolution_times else 0,
            'median_days': round(np.median(total_resolution_times) / 24, 1) if total_resolution_times else 0,
        }
    }
    
    return metrics


def generate_excel_report(tickets: List[Dict], metrics: Dict, output_path: str):
    """Generate comprehensive Excel report with multiple sheets"""
    
    print("\n📊 Generating SLA & Analytics Report...\n")
    
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    
    # =========================================================================
    # Sheet 1: Executive Dashboard (MD Level)
    # =========================================================================
    print("  📈 Creating Executive Dashboard...")
    
    exec_data = {
        'Metric': [
            '📊 Total Tickets',
            '✅ Resolved',
            '📬 Open',
            '⏳ Pending',
            '📈 Resolution Rate',
            '',
            '⏱️ SLA: First Response',
            '   - Compliance %',
            '   - Avg Response Time',
            '   - Breaches',
            '',
            '⏱️ SLA: Resolution',
            '   - Compliance %',
            '   - Avg Resolution Time',
            '   - Median Resolution',
            '   - Breaches',
            '',
            '🏢 Unique Customers',
            '👥 Active Agents',
        ],
        'Value': [
            metrics['overview']['total_tickets'],
            metrics['overview']['resolved'],
            metrics['overview']['open'],
            metrics['overview']['pending'],
            f"{metrics['overview']['resolution_rate']}%",
            '',
            '',
            f"{metrics['sla']['frt']['compliance_pct']}%",
            f"{metrics['sla']['frt']['avg_hours']:.1f} hrs",
            metrics['sla']['frt']['breaches'],
            '',
            '',
            f"{metrics['sla']['resolution']['compliance_pct']}%",
            f"{metrics['sla']['resolution']['avg_days']:.1f} days",
            f"{metrics['sla']['resolution']['median_days']:.1f} days",
            metrics['sla']['resolution']['breaches'],
            '',
            metrics['overview']['unique_companies'],
            metrics['overview']['unique_agents'],
        ],
        'Target/Benchmark': [
            '', '', '', '', '>90%',
            '',
            '',
            '>95%',
            '<4 hrs',
            '0',
            '',
            '',
            '>90%',
            '<3 days',
            '<2 days',
            '0',
            '',
            '', ''
        ],
        'Status': [
            '', '', '', '',
            '🟢' if metrics['overview']['resolution_rate'] > 90 else '🟡' if metrics['overview']['resolution_rate'] > 75 else '🔴',
            '',
            '',
            '🟢' if metrics['sla']['frt']['compliance_pct'] > 95 else '🟡' if metrics['sla']['frt']['compliance_pct'] > 85 else '🔴',
            '🟢' if metrics['sla']['frt']['avg_hours'] < 4 else '🟡' if metrics['sla']['frt']['avg_hours'] < 8 else '🔴',
            '🟢' if metrics['sla']['frt']['breaches'] == 0 else '🔴',
            '',
            '',
            '🟢' if metrics['sla']['resolution']['compliance_pct'] > 90 else '🟡' if metrics['sla']['resolution']['compliance_pct'] > 75 else '🔴',
            '🟢' if metrics['sla']['resolution']['avg_days'] < 3 else '🟡' if metrics['sla']['resolution']['avg_days'] < 7 else '🔴',
            '🟢' if metrics['sla']['resolution']['median_days'] < 2 else '🟡' if metrics['sla']['resolution']['median_days'] < 5 else '🔴',
            '🟢' if metrics['sla']['resolution']['breaches'] == 0 else '🔴',
            '',
            '', ''
        ]
    }
    
    df_exec = pd.DataFrame(exec_data)
    df_exec.to_excel(writer, sheet_name='1_Executive_Dashboard', index=False)
    
    # =========================================================================
    # Sheet 2: SLA by Priority (Manager Level)
    # =========================================================================
    print("  📊 Creating SLA by Priority...")
    
    priority_data = []
    for priority in ['Urgent', 'High', 'Medium', 'Low']:
        p_data = metrics['priorities'].get(priority, {})
        frt_times = p_data.get('frt_times', [])
        res_times = p_data.get('resolution_times', [])
        
        frt_target = SLA_CONFIG['first_response'].get(priority, 24)
        res_target = SLA_CONFIG['resolution'].get(priority, 168)
        
        frt_breaches = p_data.get('frt_breaches', 0)
        res_breaches = p_data.get('resolution_breaches', 0)
        
        priority_data.append({
            'Priority': priority,
            'Total Tickets': p_data.get('total', 0),
            'Resolved': p_data.get('resolved', 0),
            'FRT Target (hrs)': frt_target,
            'FRT Avg (hrs)': round(np.mean(frt_times), 1) if frt_times else 'N/A',
            'FRT Breaches': frt_breaches,
            'FRT Compliance %': round((1 - frt_breaches / len(frt_times)) * 100, 1) if frt_times else 'N/A',
            'Resolution Target (hrs)': res_target,
            'Resolution Avg (hrs)': round(np.mean(res_times), 1) if res_times else 'N/A',
            'Resolution Breaches': res_breaches,
            'Resolution Compliance %': round((1 - res_breaches / len(res_times)) * 100, 1) if res_times else 'N/A',
        })
    
    df_priority = pd.DataFrame(priority_data)
    df_priority.to_excel(writer, sheet_name='2_SLA_by_Priority', index=False)
    
    # =========================================================================
    # Sheet 3: Agent Performance (TL Level)
    # =========================================================================
    print("  👥 Creating Agent Performance...")
    
    agent_data = []
    for agent, data in sorted(metrics['agents'].items(), key=lambda x: x[1]['total'], reverse=True):
        if agent == 'Unassigned':
            continue
            
        frt_times = data['frt_times']
        res_times = data['resolution_times']
        convos = data['conversations']
        
        agent_data.append({
            'Agent': agent,
            'Total Tickets': data['total'],
            'Resolved': data['resolved'],
            'Open': data['open'],
            'Resolution Rate %': round(data['resolved'] / data['total'] * 100, 1) if data['total'] else 0,
            'Avg FRT (hrs)': round(np.mean(frt_times), 1) if frt_times else 'N/A',
            'FRT Breaches': data['frt_breaches'],
            'Avg Resolution (hrs)': round(np.mean(res_times), 1) if res_times else 'N/A',
            'Resolution Breaches': data['resolution_breaches'],
            'Avg Conversations': round(np.mean(convos), 1) if convos else 0,
            'SLA Score': round(
                ((1 - data['frt_breaches'] / len(frt_times)) * 50 + 
                 (1 - data['resolution_breaches'] / len(res_times)) * 50)
                if frt_times and res_times else 0, 1
            )
        })
    
    df_agents = pd.DataFrame(agent_data)
    df_agents = df_agents.sort_values('SLA Score', ascending=False)
    df_agents.to_excel(writer, sheet_name='3_Agent_Performance', index=False)
    
    # =========================================================================
    # Sheet 4: Customer Health Score (Manager Level)
    # =========================================================================
    print("  🏢 Creating Customer Health Scores...")
    
    customer_data = []
    for company, data in sorted(metrics['customers'].items(), key=lambda x: x[1]['total'], reverse=True)[:50]:
        frt_times = data['frt_times']
        res_times = data['resolution_times']
        
        # Health score calculation (0-100)
        # Factors: resolution rate, avg resolution time, ticket volume trend
        resolution_rate = data['resolved'] / data['total'] if data['total'] else 0
        avg_res = np.mean(res_times) if res_times else 0
        res_score = max(0, 100 - (avg_res / 24) * 5)  # Penalize long resolution
        
        health_score = round((resolution_rate * 50 + res_score * 0.5), 1)
        
        customer_data.append({
            'Company': company,
            'Total Tickets': data['total'],
            'Resolved': data['resolved'],
            'Open': data['open'],
            'Pending': data['pending'],
            'Resolution Rate %': round(resolution_rate * 100, 1),
            'Avg FRT (hrs)': round(np.mean(frt_times), 1) if frt_times else 'N/A',
            'Avg Resolution (days)': round(np.mean(res_times) / 24, 1) if res_times else 'N/A',
            'Health Score': min(100, health_score),
            'Risk Level': '🔴 High' if health_score < 50 else '🟡 Medium' if health_score < 75 else '🟢 Low',
            'Top Priority': data['priorities'].most_common(1)[0][0] if data['priorities'] else 'N/A'
        })
    
    df_customers = pd.DataFrame(customer_data)
    df_customers.to_excel(writer, sheet_name='4_Customer_Health', index=False)
    
    # =========================================================================
    # Sheet 5: Monthly Trends (MD Level)
    # =========================================================================
    print("  📈 Creating Monthly Trends...")
    
    trend_data = []
    for month in sorted(metrics['trends']['monthly_volume'].keys()):
        volume = metrics['trends']['monthly_volume'][month]
        resolved = metrics['trends']['monthly_resolved'].get(month, 0)
        frt_breach = metrics['trends']['monthly_frt_breach'].get(month, 0)
        res_breach = metrics['trends']['monthly_resolution_breach'].get(month, 0)
        res_times = metrics['trends']['monthly_avg_resolution'].get(month, [])
        
        trend_data.append({
            'Month': month,
            'Ticket Volume': volume,
            'Resolved': resolved,
            'Resolution Rate %': round(resolved / volume * 100, 1) if volume else 0,
            'FRT Breaches': frt_breach,
            'Resolution Breaches': res_breach,
            'Avg Resolution (days)': round(np.mean(res_times) / 24, 1) if res_times else 'N/A',
            'Total Breaches': frt_breach + res_breach
        })
    
    df_trends = pd.DataFrame(trend_data)
    df_trends.to_excel(writer, sheet_name='5_Monthly_Trends', index=False)
    
    # =========================================================================
    # Sheet 6: Ticket Aging (TL Level)
    # =========================================================================
    print("  ⏰ Creating Ticket Aging Analysis...")
    
    aging_data = []
    for bucket, count in metrics['aging'].items():
        aging_data.append({
            'Age Bucket': bucket,
            'Ticket Count': count,
            'Percentage': round(count / (metrics['overview']['open'] + metrics['overview']['pending']) * 100, 1) 
                         if (metrics['overview']['open'] + metrics['overview']['pending']) else 0,
            'Action': '✅ On Track' if '0-1' in bucket or '1-3' in bucket 
                     else '⚠️ Monitor' if '3-7' in bucket 
                     else '🔴 Escalate'
        })
    
    df_aging = pd.DataFrame(aging_data)
    df_aging.to_excel(writer, sheet_name='6_Ticket_Aging', index=False)
    
    # =========================================================================
    # Sheet 7: Time Patterns (Manager Level)
    # =========================================================================
    print("  🕐 Creating Time Patterns...")
    
    # By hour
    hour_data = [{'Hour': f'{h:02d}:00', 'Tickets': metrics['time_analysis']['by_hour'].get(h, 0)} 
                 for h in range(24)]
    df_hours = pd.DataFrame(hour_data)
    df_hours.to_excel(writer, sheet_name='7_Time_Patterns', index=False, startrow=0)
    
    # By day of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = [{'Day': d, 'Tickets': metrics['time_analysis']['by_day'].get(d, 0)} for d in days_order]
    df_days = pd.DataFrame(day_data)
    df_days.to_excel(writer, sheet_name='7_Time_Patterns', index=False, startrow=len(hour_data) + 3, startcol=0)
    
    # =========================================================================
    # Sheet 8: SLA Breaches Detail (TL Level)
    # =========================================================================
    print("  ⚠️ Creating SLA Breach Details...")
    
    breach_tickets = []
    for ticket in tickets:
        priority = ticket.get('priority_name', 'Medium')
        frt = calculate_first_response_time(ticket)
        resolution = ticket.get('resolution_time_hours')
        
        frt_target = SLA_CONFIG['first_response'].get(priority, 24)
        res_target = SLA_CONFIG['resolution'].get(priority, 168)
        
        frt_breached = frt and frt > frt_target
        res_breached = resolution and resolution > res_target
        
        if frt_breached or res_breached:
            breach_tickets.append({
                'Ticket ID': ticket.get('id'),
                'URL': get_freshdesk_url(ticket.get('id')),
                'Subject': ticket.get('subject', '')[:50],
                'Priority': priority,
                'Status': ticket.get('status_name'),
                'Company': ticket.get('company', {}).get('name', 'Unknown') if ticket.get('company') else 'Unknown',
                'FRT Target (hrs)': frt_target,
                'FRT Actual (hrs)': round(frt, 1) if frt else 'N/A',
                'FRT Breached': '❌' if frt_breached else '✅',
                'Resolution Target (hrs)': res_target,
                'Resolution Actual (hrs)': round(resolution, 1) if resolution else 'N/A',
                'Resolution Breached': '❌' if res_breached else '✅',
                'Created': ticket.get('created_at', '')[:10]
            })
    
    df_breaches = pd.DataFrame(breach_tickets)
    df_breaches.to_excel(writer, sheet_name='8_SLA_Breaches', index=False)
    
    # =========================================================================
    # Sheet 9: Source Analysis (Manager Level)
    # =========================================================================
    print("  📨 Creating Source Analysis...")
    
    source_data = [{'Source': source, 'Tickets': count, 'Percentage': round(count / len(tickets) * 100, 1)}
                   for source, count in metrics['sources'].most_common()]
    df_sources = pd.DataFrame(source_data)
    df_sources.to_excel(writer, sheet_name='9_Source_Analysis', index=False)
    
    # =========================================================================
    # Sheet 10: Tags Analysis (Manager Level)
    # =========================================================================
    print("  🏷️ Creating Tags Analysis...")
    
    tag_data = [{'Tag': tag, 'Count': count} for tag, count in metrics['tags'].most_common(50)]
    df_tags = pd.DataFrame(tag_data)
    df_tags.to_excel(writer, sheet_name='10_Tags_Analysis', index=False)
    
    # =========================================================================
    # Sheet 11: Escalation Candidates (TL Level)
    # =========================================================================
    print("  🚨 Creating Escalation List...")
    
    escalation_data = [{
        'Ticket ID': e['id'],
        'URL': get_freshdesk_url(e['id']),
        'Subject': e['subject'],
        'Priority': e['priority'],
        'Resolution Hours': round(e['resolution_hours'], 1),
        'Resolution Days': round(e['resolution_hours'] / 24, 1),
        'Company': e['company']
    } for e in sorted(metrics['escalations'], key=lambda x: x['resolution_hours'], reverse=True)[:100]]
    
    df_escalations = pd.DataFrame(escalation_data)
    df_escalations.to_excel(writer, sheet_name='11_Escalations', index=False)
    
    # =========================================================================
    # Sheet 12: SLA Config (Reference)
    # =========================================================================
    print("  ⚙️ Creating SLA Config Reference...")
    
    config_data = []
    for priority in ['Urgent', 'High', 'Medium', 'Low']:
        config_data.append({
            'Priority': priority,
            'First Response Target (hrs)': SLA_CONFIG['first_response'].get(priority),
            'Resolution Target (hrs)': SLA_CONFIG['resolution'].get(priority),
            'Resolution Target (days)': round(SLA_CONFIG['resolution'].get(priority, 0) / 24, 1)
        })
    
    df_config = pd.DataFrame(config_data)
    df_config.to_excel(writer, sheet_name='12_SLA_Config', index=False)
    
    # Save
    writer.close()
    print(f"\n✅ Report saved: {output_path}")
    
    return metrics


def print_summary(metrics: Dict):
    """Print summary to console"""
    if not RICH:
        return
    
    console.print()
    
    # Executive Summary Table
    table = Table(title="📊 SLA & Analytics Summary", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold white")
    table.add_column("Status", justify="center")
    
    # Overview
    table.add_row("Total Tickets", f"{metrics['overview']['total_tickets']:,}", "")
    table.add_row("Resolution Rate", f"{metrics['overview']['resolution_rate']}%",
                  "🟢" if metrics['overview']['resolution_rate'] > 90 else "🟡" if metrics['overview']['resolution_rate'] > 75 else "🔴")
    table.add_row("", "", "")
    
    # FRT SLA
    table.add_row("FRT Compliance", f"{metrics['sla']['frt']['compliance_pct']}%",
                  "🟢" if metrics['sla']['frt']['compliance_pct'] > 95 else "🟡" if metrics['sla']['frt']['compliance_pct'] > 85 else "🔴")
    table.add_row("Avg First Response", f"{metrics['sla']['frt']['avg_hours']:.1f} hrs",
                  "🟢" if metrics['sla']['frt']['avg_hours'] < 4 else "🟡" if metrics['sla']['frt']['avg_hours'] < 8 else "🔴")
    table.add_row("FRT Breaches", f"{metrics['sla']['frt']['breaches']:,}",
                  "🟢" if metrics['sla']['frt']['breaches'] == 0 else "🔴")
    table.add_row("", "", "")
    
    # Resolution SLA
    table.add_row("Resolution Compliance", f"{metrics['sla']['resolution']['compliance_pct']}%",
                  "🟢" if metrics['sla']['resolution']['compliance_pct'] > 90 else "🟡" if metrics['sla']['resolution']['compliance_pct'] > 75 else "🔴")
    table.add_row("Avg Resolution", f"{metrics['sla']['resolution']['avg_days']:.1f} days",
                  "🟢" if metrics['sla']['resolution']['avg_days'] < 3 else "🟡" if metrics['sla']['resolution']['avg_days'] < 7 else "🔴")
    table.add_row("Resolution Breaches", f"{metrics['sla']['resolution']['breaches']:,}",
                  "🟢" if metrics['sla']['resolution']['breaches'] == 0 else "🔴")
    
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description='Generate SLA & Analytics Report')
    parser.add_argument('--input', '-i', default='output/tickets.json', help='Path to tickets.json')
    parser.add_argument('--output', '-o', default='sla_analytics_report.xlsx', help='Output Excel file')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ Not found: {args.input}")
        exit(1)
    
    # Banner
    if RICH:
        console.print(Panel(
            "[bold white]📊 FTEX SLA & Analytics Report Generator[/bold white]\n"
            "[dim]Comprehensive metrics for TLs, Managers, and MDs[/dim]",
            border_style="cyan"
        ))
    
    # Load tickets
    tickets = load_tickets(args.input)
    print(f"\n✅ Loaded {len(tickets):,} tickets")
    
    # Calculate metrics
    metrics = calculate_metrics(tickets)
    
    # Generate report
    generate_excel_report(tickets, metrics, args.output)
    
    # Print summary
    print_summary(metrics)
    
    if RICH:
        console.print(f"[bold]Open [cyan]{args.output}[/cyan] for detailed analysis.[/bold]\n")


if __name__ == '__main__':
    main()