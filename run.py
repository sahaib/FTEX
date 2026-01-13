#!/usr/bin/env python3
"""
FTEX CLI - Unified Entry Point
==============================
Single command interface for all FTEX operations.

Commands:
    python3 run.py test                     # Test API connection
    python3 run.py extract --days 180       # Extract tickets
    python3 run.py analyze                  # Run AI analysis + reports
    python3 run.py full --days 180          # Complete pipeline
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False

# Config from environment
FRESHDESK_API_KEY = os.getenv('FRESHDESK_API_KEY', '')
FRESHDESK_GROUP_ID = os.getenv('FRESHDESK_GROUP_ID', '')
FRESHDESK_DOMAIN = os.getenv('FRESHDESK_DOMAIN', '')


def print_header():
    if console:
        console.print(Panel.fit(
            "[bold cyan]FTEX[/bold cyan] - Freshdesk Ticket Export & Analysis\n"
            "[dim]Self-Validating AI-Powered Analysis[/dim]",
            border_style="cyan"
        ))
        console.print()
    else:
        print("\n" + "="*60)
        print("FTEX - Freshdesk Ticket Export & Analysis")
        print("="*60 + "\n")


def print_success(msg):
    if console:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"✓ {msg}")


def print_error(msg):
    if console:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"✗ {msg}")


def print_step(step, total, msg):
    if console:
        console.print(f"\n[bold blue]Step {step}/{total}:[/bold blue] {msg}")
    else:
        print(f"\n[Step {step}/{total}] {msg}")


def run_command(cmd, description=None):
    if description and console:
        console.print(f"  [dim]→ {description}[/dim]")
    elif description:
        print(f"  → {description}")
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Command failed: {e}")
        return False


def get_script_path(name):
    """Get script path (organized or flat structure)."""
    paths = {
        'extractor': 'src/extraction/freshdesk_extractor_v2.py',
        'test_api': 'src/extraction/test_freshdesk_api.py',
        'analyze': 'src/analysis/analyze.py',
    }
    
    path = paths.get(name)
    if path and Path(path).exists():
        return path
    
    # Flat fallback
    flat = {
        'extractor': 'freshdesk_extractor_v2.py',
        'test_api': 'test_freshdesk_api.py',
        'analyze': 'analyze.py',
    }
    return flat.get(name, name)


def cmd_test(args):
    """Test Freshdesk API."""
    if not args.api_key:
        print_error("API key required. Use --api-key or set FRESHDESK_API_KEY")
        return False
    
    script = get_script_path('test_api')
    cmd = ['python3', script, '--api-key', args.api_key]
    if args.domain:
        cmd.extend(['--domain', args.domain])
    
    return run_command(cmd, "Testing Freshdesk API...")


def cmd_extract(args):
    """Extract tickets."""
    if not args.api_key:
        print_error("API key required. Use --api-key or set FRESHDESK_API_KEY")
        return False
    
    script = get_script_path('extractor')
    cmd = ['python3', script, '--api-key', args.api_key, '--days', str(args.days)]
    
    if args.group_id:
        cmd.extend(['--group-id', str(args.group_id)])
    if args.no_attachments:
        cmd.append('--no-attachments')
    if args.resume:
        cmd.append('--resume')
    
    print_step(1, 1, f"Extracting tickets (last {args.days} days)")
    return run_command(cmd)


def cmd_analyze(args):
    """Run analysis."""
    if not Path(args.input).exists():
        print_error(f"File not found: {args.input}")
        print_error("Run 'python3 run.py extract' first")
        return False
    
    # Try direct import
    try:
        from analysis.analyze import main as analyze_main
        
        # Call main() with keyword arguments directly
        analyze_main(
            input_path=args.input,
            output_dir=args.output,
            use_ai=not args.no_ai,
            clear_cache=args.clear_cache
        )
        return True
    except ImportError:
        # Subprocess fallback
        script = get_script_path('analyze')
        cmd = ['python3', script, '--input', args.input, '--output', args.output]
        if args.no_ai:
            cmd.append('--no-ai')
        if args.clear_cache:
            cmd.append('--clear-cache')
        
        return run_command(cmd, "Running analysis...")


def cmd_full(args):
    """Full pipeline: extract → analyze."""
    print_header()
    
    if console:
        console.print(Panel("[bold]Full Pipeline[/bold]: Extract → Analyze → Report", border_style="blue"))
    
    # Step 1: Extract
    if not args.skip_extract:
        print_step(1, 2, f"Extracting tickets (last {args.days} days)")
        
        if not args.api_key:
            print_error("API key required")
            return False
        
        script = get_script_path('extractor')
        cmd = ['python3', script, '--api-key', args.api_key, '--days', str(args.days)]
        if args.group_id:
            cmd.extend(['--group-id', str(args.group_id)])
        if args.no_attachments:
            cmd.append('--no-attachments')
        
        if not run_command(cmd):
            print_error("Extraction failed")
            return False
        
        print_success("Extraction complete")
    else:
        if console:
            console.print("  [yellow]⏭[/yellow] Skipping extraction")
        else:
            print("  ⏭ Skipping extraction")
    
    # Step 2: Analyze
    print_step(2, 2, "Running AI analysis + generating reports")
    
    class AnalyzeArgs:
        input = 'output/tickets.json'
        output = 'reports'
        no_ai = getattr(args, 'no_ai', False)
        clear_cache = False
    
    if not cmd_analyze(AnalyzeArgs()):
        print_error("Analysis failed")
        return False
    
    # Summary
    if console:
        console.print()
        summary = Table(title="Pipeline Complete", box=box.ROUNDED, border_style="green")
        summary.add_column("Output", style="cyan")
        summary.add_column("Location")
        
        summary.add_row("Extracted Tickets", "output/tickets.json")
        summary.add_row("Excel Report", "reports/analysis_report.xlsx")
        summary.add_row("Summary (MD)", "reports/analysis_summary.md")
        summary.add_row("Summary (PDF)", "reports/analysis_summary.pdf")
        summary.add_row("Raw Data", "reports/analysis_data.json")
        
        console.print(summary)
    else:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print("\nOutputs:")
        print("  📊 output/tickets.json")
        print("  📋 reports/analysis_report.xlsx")
        print("  📄 reports/analysis_summary.md")
        print("  📄 reports/analysis_summary.pdf")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='FTEX - Freshdesk Ticket Export & Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run.py test --api-key YOUR_KEY
  python3 run.py extract --days 180 --group-id 12345
  python3 run.py analyze
  python3 run.py analyze --no-ai
  python3 run.py full --days 180

Environment Variables:
  FRESHDESK_DOMAIN    Your Freshdesk subdomain
  FRESHDESK_API_KEY   Your API key
  FRESHDESK_GROUP_ID  Default group ID
  OLLAMA_URL          Ollama server URL
  OLLAMA_MODEL        LLM model name
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # TEST
    test = subparsers.add_parser('test', help='Test Freshdesk API')
    test.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY)
    test.add_argument('--domain', '-d', default=FRESHDESK_DOMAIN)
    
    # EXTRACT
    extract = subparsers.add_parser('extract', help='Extract tickets')
    extract.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY)
    extract.add_argument('--days', '-d', type=int, default=180)
    extract.add_argument('--group-id', '-g', type=int,
                        default=int(FRESHDESK_GROUP_ID) if FRESHDESK_GROUP_ID else None)
    extract.add_argument('--no-attachments', action='store_true')
    extract.add_argument('--resume', action='store_true')
    
    # ANALYZE
    analyze = subparsers.add_parser('analyze', help='Run AI analysis')
    analyze.add_argument('--input', '-i', default='output/tickets.json')
    analyze.add_argument('--output', '-o', default='reports')
    analyze.add_argument('--no-ai', action='store_true')
    analyze.add_argument('--clear-cache', action='store_true')
    
    # FULL
    full = subparsers.add_parser('full', help='Full pipeline')
    full.add_argument('--api-key', '-k', default=FRESHDESK_API_KEY)
    full.add_argument('--days', '-d', type=int, default=180)
    full.add_argument('--group-id', '-g', type=int,
                     default=int(FRESHDESK_GROUP_ID) if FRESHDESK_GROUP_ID else None)
    full.add_argument('--no-attachments', action='store_true')
    full.add_argument('--skip-extract', action='store_true')
    full.add_argument('--no-ai', action='store_true')
    
    args = parser.parse_args()
    
    if not args.command:
        print_header()
        parser.print_help()
        return
    
    if args.command == 'test':
        success = cmd_test(args)
    elif args.command == 'extract':
        success = cmd_extract(args)
    elif args.command == 'analyze':
        success = cmd_analyze(args)
    elif args.command == 'full':
        success = cmd_full(args)
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()