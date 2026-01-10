#!/usr/bin/env python3
"""
FTEX CLI - Unified Entry Point
==============================
Single command interface for all FTEX operations.

Usage:
    python run.py extract --api-key KEY --days 180 --group-id ID
    python run.py analyze --deep
    python run.py report --type all
    python run.py full --api-key KEY  # Run entire pipeline

Environment Variables:
    FRESHDESK_DOMAIN   - Your Freshdesk subdomain
    FRESHDESK_API_KEY  - Your API key
    FRESHDESK_GROUP_ID - Default group ID
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
FRESHDESK_API_KEY = os.getenv('FRESHDESK_API_KEY', '')
FRESHDESK_GROUP_ID = os.getenv('FRESHDESK_GROUP_ID', '')


def run_command(cmd, description):
    """Run a command with description"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=isinstance(cmd, str))
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description='FTEX - Freshdesk Ticket Extraction & Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py extract --api-key YOUR_KEY --days 180 --group-id YOUR_GROUP_ID
  python run.py analyze --use-ollama --deep
  python run.py report --type all
  python run.py full --api-key YOUR_KEY --days 90

Environment Variables:
  FRESHDESK_DOMAIN   - Your Freshdesk subdomain  
  FRESHDESK_API_KEY  - Your API key (alternative to --api-key)
  FRESHDESK_GROUP_ID - Default group ID
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # EXTRACT COMMAND
    # =========================================================================
    extract = subparsers.add_parser('extract', help='Extract tickets from Freshdesk')
    extract.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY, help='Freshdesk API key (or set FRESHDESK_API_KEY)')
    extract.add_argument('--days', '-d', type=int, default=180, help='Days of history (default: 180)')
    extract.add_argument('--group-id', '-g', type=int, default=int(FRESHDESK_GROUP_ID) if FRESHDESK_GROUP_ID else None, help='Filter by group ID')
    extract.add_argument('--no-attachments', action='store_true', help='Skip attachment downloads')
    extract.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    # =========================================================================
    # ANALYZE COMMAND
    # =========================================================================
    analyze = subparsers.add_parser('analyze', help='Run ticket analysis')
    analyze.add_argument('--input', '-i', default='output/tickets.json', help='Input file')
    analyze.add_argument('--use-ollama', action='store_true', help='Enable GenAI analysis')
    analyze.add_argument('--deep', action='store_true', help='Run deep AI content analysis')
    
    # =========================================================================
    # REPORT COMMAND
    # =========================================================================
    report = subparsers.add_parser('report', help='Generate reports')
    report.add_argument('--input', '-i', default='output/tickets.json', help='Input file')
    report.add_argument('--type', '-t', choices=['actionable', 'sla', 'all'], default='all',
                       help='Report type (default: all)')
    report.add_argument('--output-dir', '-o', default='reports', help='Output directory')
    
    # =========================================================================
    # FULL PIPELINE COMMAND
    # =========================================================================
    full = subparsers.add_parser('full', help='Run full pipeline (extract → analyze → report)')
    full.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY, help='Freshdesk API key (or set FRESHDESK_API_KEY)')
    full.add_argument('--days', '-d', type=int, default=180, help='Days of history')
    full.add_argument('--group-id', '-g', type=int, default=int(FRESHDESK_GROUP_ID) if FRESHDESK_GROUP_ID else None, help='Filter by group ID')
    full.add_argument('--no-attachments', action='store_true', help='Skip attachments')
    full.add_argument('--skip-extract', action='store_true', help='Skip extraction (use existing data)')
    
    # =========================================================================
    # TEST COMMAND
    # =========================================================================
    test = subparsers.add_parser('test', help='Test Freshdesk API connection')
    test.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY, help='Freshdesk API key (or set FRESHDESK_API_KEY)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate API key for commands that need it
    if args.command in ['extract', 'full', 'test']:
        if not args.api_key:
            print("❌ API key required. Use --api-key or set FRESHDESK_API_KEY environment variable.")
            sys.exit(1)
    
    # Determine script paths (handle both flat and organized structures)
    src_extraction = Path('src/extraction')
    src_analysis = Path('src/analysis')
    src_reports = Path('src/reports')
    
    # Check if organized structure exists
    organized = src_extraction.exists()
    
    if organized:
        extractor = 'src/extraction/freshdesk_extractor_v2.py'
        analyzer = 'src/analysis/analyze_tickets.py'
        deep_analyzer = 'src/analysis/deep_ai_analysis.py'
        actionable_report = 'src/reports/generate_actionable_report.py'
        sla_report = 'src/reports/generate_sla_report.py'
        test_api = 'src/extraction/test_freshdesk_api.py'
    else:
        extractor = 'freshdesk_extractor_v2.py'
        analyzer = 'analyze_tickets.py'
        deep_analyzer = 'deep_ai_analysis.py'
        actionable_report = 'generate_actionable_report.py'
        sla_report = 'generate_sla_report.py'
        test_api = 'test_freshdesk_api.py'
    
    # =========================================================================
    # EXECUTE COMMANDS
    # =========================================================================
    
    if args.command == 'test':
        cmd = f'python3 {test_api} --api-key {args.api_key}'
        run_command(cmd, 'Testing Freshdesk API connection')
        
    elif args.command == 'extract':
        cmd = f'python3 {extractor} --api-key {args.api_key} --days {args.days}'
        if args.group_id:
            cmd += f' --group-id {args.group_id}'
        if args.no_attachments:
            cmd += ' --no-attachments'
        if args.resume:
            cmd += ' --resume'
        run_command(cmd, f'Extracting tickets (last {args.days} days)')
        
    elif args.command == 'analyze':
        success = True
        
        # Standard analysis
        cmd = f'python3 {analyzer} --input {args.input}'
        if args.use_ollama:
            cmd += ' --use-ollama'
        success = run_command(cmd, 'Running cluster analysis')
        
        # Deep AI analysis (generates slides + action items)
        if args.deep and success:
            cmd = f'python3 {deep_analyzer} --input {args.input} --output-dir reports/'
            run_command(cmd, 'Running deep AI analysis (+ slides + action items)')
            
    elif args.command == 'report':
        # Ensure reports directory exists
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        if args.type in ['actionable', 'all']:
            output_file = Path(args.output_dir) / 'actionable_report.xlsx'
            cmd = f'python3 {actionable_report} --input {args.input} --output {output_file}'
            run_command(cmd, 'Generating actionable report')
            
        if args.type in ['sla', 'all']:
            output_file = Path(args.output_dir) / 'sla_analytics_report.xlsx'
            cmd = f'python3 {sla_report} --input {args.input} --output {output_file}'
            run_command(cmd, 'Generating SLA analytics report')
            
    elif args.command == 'full':
        print("\n" + "="*60)
        print("🚀 FTEX FULL PIPELINE")
        print("="*60)
        
        # Step 1: Extract
        if not args.skip_extract:
            cmd = f'python3 {extractor} --api-key {args.api_key} --days {args.days}'
            if args.group_id:
                cmd += f' --group-id {args.group_id}'
            if args.no_attachments:
                cmd += ' --no-attachments'
            if not run_command(cmd, 'Step 1/4: Extracting tickets'):
                print("❌ Extraction failed")
                return
        else:
            print("\n⏭️  Skipping extraction (using existing data)")
        
        # Step 2: Analyze
        cmd = f'python3 {analyzer} --input output/tickets.json --use-ollama'
        if not run_command(cmd, 'Step 2/4: Running cluster analysis'):
            print("⚠️  Analysis failed, continuing...")
        
        # Step 3: Deep AI Analysis (generates slides + action items)
        cmd = f'python3 {deep_analyzer} --input output/tickets.json --output-dir reports/'
        if not run_command(cmd, 'Step 3/4: Running deep AI analysis (+ slides + action items)'):
            print("⚠️  Deep analysis failed, continuing...")
        
        # Step 4: Reports (ensure reports/ directory exists)
        Path('reports').mkdir(parents=True, exist_ok=True)
        run_command(f'python3 {actionable_report} --input output/tickets.json --output reports/actionable_report.xlsx',
                   'Step 4a/4: Generating actionable report')
        run_command(f'python3 {sla_report} --input output/tickets.json --output reports/sla_analytics_report.xlsx',
                   'Step 4b/4: Generating SLA report')
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60)
        print("\nOutputs:")
        print("  📊 output/tickets.json           - Extracted tickets")
        print("  📈 reports/analysis_report.md    - Cluster analysis")
        print("  🧠 reports/deep_ai_analysis.md   - AI insights")
        print("  📽️ reports/presentation_slides.md - Sli.dev slides")
        print("  🎯 reports/action_items.md       - Action checklist")
        print("  📋 reports/actionable_report.xlsx - Ticket IDs")
        print("  📉 reports/sla_analytics_report.xlsx - SLA metrics")


if __name__ == '__main__':
    main()
