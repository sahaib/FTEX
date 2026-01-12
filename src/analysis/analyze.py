#!/usr/bin/env python3
"""
FTEX Analyzer v6.0
==================
Single-command ticket analysis with beautiful terminal UI.

Usage:
    python3 analyze.py                          # Analyze output/tickets.json
    python3 analyze.py --input data/tickets.json
    python3 analyze.py --no-ai                  # Statistical only
    python3 analyze.py --clear-cache            # Force re-discovery

Outputs:
    reports/
    ├── analysis_report.xlsx    # Multi-sheet Excel
    ├── analysis_summary.md     # Markdown summary
    ├── analysis_summary.pdf    # PDF version
    └── analysis_data.json      # Raw data

Author: FTEX Project
License: MIT
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Rich imports for beautiful terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.style import Style
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for beautiful terminal UI: pip install rich")

# Import analysis engine
try:
    from smart_detection import (
        AnalysisEngine,
        UserConfig,
        OllamaClient,
        get_zombie_stats,
    )
except ImportError:
    from src.shared.smart_detection import (
        AnalysisEngine,
        UserConfig,
        OllamaClient,
        get_zombie_stats,
    )

# Excel/DataFrame support
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# PDF support
try:
    from weasyprint import HTML, CSS
    PDF_AVAILABLE = True
except ImportError:
    try:
        from markdown import markdown
        import pdfkit
        PDF_AVAILABLE = True
        PDF_ENGINE = 'pdfkit'
    except ImportError:
        PDF_AVAILABLE = False
        PDF_ENGINE = None


# =============================================================================
# TERMINAL UI
# =============================================================================

class TerminalUI:
    """Rich terminal interface."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.start_time = datetime.now()
    
    def print_header(self):
        """Print application header."""
        if self.console:
            header = Panel(
                Text.assemble(
                    ("FTEX Deep Analyzer", Style(color="cyan", bold=True)),
                    "\n",
                    ("v6.0 • Self-Validating AI Analysis", Style(color="white", dim=True)),
                ),
                box=box.DOUBLE,
                border_style="cyan",
                padding=(1, 2),
            )
            self.console.print(header)
            self.console.print()
        else:
            print("\n" + "="*60)
            print("FTEX Deep Analyzer v6.0")
            print("="*60 + "\n")
    
    def print_config(self, ai_available: bool, model: str = None):
        """Print configuration summary."""
        if self.console:
            config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
            config_table.add_column("Key", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_table.add_row("Product", UserConfig.PRODUCT_NAME)
            config_table.add_row("Entity Type", UserConfig.ENTITY_NAME)
            config_table.add_row("Known Solutions", str(len(UserConfig.KNOWN_SOLUTIONS)))
            
            if ai_available:
                config_table.add_row("AI Status", f"[green]✓ Connected[/green] ({model})")
            else:
                config_table.add_row("AI Status", "[yellow]⚠ Fallback mode[/yellow]")
            
            self.console.print(Panel(config_table, title="Configuration", border_style="blue"))
            self.console.print()
        else:
            print(f"Product: {UserConfig.PRODUCT_NAME}")
            print(f"Entity: {UserConfig.ENTITY_NAME}")
            print(f"AI: {'Available' if ai_available else 'Fallback mode'}")
            print()
    
    def print_loading(self, path: str, count: int):
        """Print ticket loading status."""
        if self.console:
            self.console.print(f"  📂 Loaded [cyan]{count:,}[/cyan] tickets from [dim]{path}[/dim]")
        else:
            print(f"  Loaded {count:,} tickets from {path}")
    
    def create_progress(self):
        """Create progress display."""
        if self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                console=self.console,
                transient=False,
            )
        return None
    
    def print_stage(self, stage: str, current: int, total: int):
        """Print stage progress."""
        if not self.console:
            print(f"  [{current}/{total}] {stage}")
    
    def print_info(self, message: str):
        """Print info message."""
        if self.console:
            self.console.print(f"    [dim]→ {message}[/dim]")
        else:
            print(f"    → {message}")
    
    def print_results_summary(self, results: Dict):
        """Print analysis results summary."""
        if self.console:
            self.console.print()
            
            # Overview table
            overview = Table(title="📊 Analysis Results", box=box.ROUNDED, border_style="green")
            overview.add_column("Metric", style="cyan")
            overview.add_column("Value", justify="right")
            
            meta = results.get('metadata', {})
            foundation = results.get('foundation', {})
            zombies = results.get('zombies', {})
            
            overview.add_row("Total Tickets", f"{meta.get('total_tickets', 0):,}")
            overview.add_row("Date Range", f"{foundation.get('date_range', {}).get('earliest', 'N/A')} → {foundation.get('date_range', {}).get('latest', 'N/A')}")
            overview.add_row("True Zombies", f"[red]{zombies.get('true_zombie_count', 0):,}[/red] ({zombies.get('zombie_rate', 0)}%)")
            overview.add_row("Categories Found", str(len(results.get('categories', {}))))
            overview.add_row(f"{UserConfig.ENTITY_NAME_PLURAL.title()}", str(results.get('entities', {}).get('total_entities', 0)))
            overview.add_row("Anomalies Detected", f"[yellow]{len(results.get('anomalies', []))}[/yellow]")
            overview.add_row("Findings Generated", str(len(results.get('findings', []))))
            
            self.console.print(overview)
            
            # Top categories
            categories = results.get('categories', {})
            if categories:
                self.console.print()
                cat_table = Table(title="🏷️ Top Issue Categories", box=box.ROUNDED)
                cat_table.add_column("Category", style="cyan")
                cat_table.add_column("Tickets", justify="right")
                cat_table.add_column("Zombies", justify="right")
                cat_table.add_column("Avg Resolution", justify="right")
                
                for cat_name, cat_data in sorted(categories.items(), key=lambda x: x[1].get('total_tickets', 0), reverse=True)[:7]:
                    res_days = cat_data.get('avg_resolution_days')
                    res_str = f"{res_days:.1f}d" if res_days else "—"
                    cat_table.add_row(
                        cat_name,
                        str(cat_data.get('total_tickets', 0)),
                        f"[red]{cat_data.get('zombie_count', 0)}[/red]",
                        res_str
                    )
                
                self.console.print(cat_table)
            
            # Anomalies
            anomalies = results.get('anomalies', [])
            if anomalies:
                self.console.print()
                anom_table = Table(title="⚠️ Anomalies", box=box.ROUNDED, border_style="yellow")
                anom_table.add_column("Type", style="yellow")
                anom_table.add_column("Description")
                anom_table.add_column("Severity", justify="center")
                
                for anom in anomalies[:5]:
                    severity = anom.get('severity', 'medium')
                    sev_style = "red" if severity == 'high' else "yellow"
                    anom_table.add_row(
                        anom.get('type', '').replace('_', ' '),
                        anom.get('description', '')[:60] + "..." if len(anom.get('description', '')) > 60 else anom.get('description', ''),
                        f"[{sev_style}]{severity}[/{sev_style}]"
                    )
                
                if len(anomalies) > 5:
                    self.console.print(f"  [dim]... and {len(anomalies) - 5} more[/dim]")
                
                self.console.print(anom_table)
            
            # SLA
            sla = results.get('sla', {})
            if sla:
                self.console.print()
                sla_table = Table(title="📈 SLA Performance", box=box.ROUNDED)
                sla_table.add_column("Metric", style="cyan")
                sla_table.add_column("Value", justify="right")
                sla_table.add_column("Compliance", justify="right")
                
                frt = sla.get('first_response', {})
                res = sla.get('resolution', {})
                
                frt_compliance = frt.get('compliance_rate', 0)
                res_compliance = res.get('compliance_rate', 0)
                
                frt_style = "green" if frt_compliance >= 90 else "yellow" if frt_compliance >= 70 else "red"
                res_style = "green" if res_compliance >= 90 else "yellow" if res_compliance >= 70 else "red"
                
                sla_table.add_row(
                    "First Response",
                    f"{frt.get('avg', 'N/A')} hrs",
                    f"[{frt_style}]{frt_compliance}%[/{frt_style}]"
                )
                sla_table.add_row(
                    "Resolution",
                    f"{res.get('avg', 'N/A')} hrs",
                    f"[{res_style}]{res_compliance}%[/{res_style}]"
                )
                
                self.console.print(sla_table)
        else:
            # Plain text output
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)
            meta = results.get('metadata', {})
            zombies = results.get('zombies', {})
            print(f"Total Tickets: {meta.get('total_tickets', 0):,}")
            print(f"Zombies: {zombies.get('true_zombie_count', 0)} ({zombies.get('zombie_rate', 0)}%)")
            print(f"Categories: {len(results.get('categories', {}))}")
            print(f"Anomalies: {len(results.get('anomalies', []))}")
    
    def print_outputs(self, outputs: Dict[str, str]):
        """Print generated output files."""
        if self.console:
            self.console.print()
            output_panel = Panel(
                "\n".join([
                    f"  📄 [cyan]{name}[/cyan]: {path}"
                    for name, path in outputs.items()
                ]),
                title="📁 Generated Files",
                border_style="green",
            )
            self.console.print(output_panel)
        else:
            print("\nGenerated Files:")
            for name, path in outputs.items():
                print(f"  {name}: {path}")
    
    def print_completion(self):
        """Print completion message."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.console:
            self.console.print()
            self.console.print(Panel(
                f"[bold green]✓ Analysis complete[/bold green] in {elapsed:.1f}s",
                border_style="green",
            ))
        else:
            print(f"\n✓ Analysis complete in {elapsed:.1f}s")
    
    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[bold red]✗ Error:[/bold red] {message}")
        else:
            print(f"ERROR: {message}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generates Excel, Markdown, and PDF reports."""
    
    def __init__(self, results: Dict, output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self) -> Dict[str, str]:
        """Generate all report formats."""
        outputs = {}
        
        # Excel
        if PANDAS_AVAILABLE:
            excel_path = self.generate_excel()
            if excel_path:
                outputs['Excel Report'] = str(excel_path)
        
        # Markdown
        md_path = self.generate_markdown()
        outputs['Markdown Summary'] = str(md_path)
        
        # PDF
        if PDF_AVAILABLE:
            pdf_path = self.generate_pdf(md_path)
            if pdf_path:
                outputs['PDF Summary'] = str(pdf_path)
        
        # JSON
        json_path = self.generate_json()
        outputs['Raw Data'] = str(json_path)
        
        return outputs
    
    def generate_excel(self) -> Optional[Path]:
        """Generate multi-sheet Excel report."""
        if not PANDAS_AVAILABLE:
            return None
        
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils.dataframe import dataframe_to_rows
        except ImportError:
            return None
        
        excel_path = self.output_dir / 'analysis_report.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Overview
            overview_data = self._build_overview_df()
            overview_data.to_excel(writer, sheet_name='Overview', index=False)
            
            # Sheet 2: Issue Categories
            categories_data = self._build_categories_df()
            if not categories_data.empty:
                categories_data.to_excel(writer, sheet_name='Issue Categories', index=False)
            
            # Sheet 3: Entities
            entities_data = self._build_entities_df()
            if not entities_data.empty:
                entities_data.to_excel(writer, sheet_name=UserConfig.ENTITY_NAME_PLURAL.title(), index=False)
            
            # Sheet 4: Anomalies
            anomalies_data = self._build_anomalies_df()
            if not anomalies_data.empty:
                anomalies_data.to_excel(writer, sheet_name='Anomalies', index=False)
            
            # Sheet 5: Zombies
            zombies_data = self._build_zombies_df()
            if not zombies_data.empty:
                zombies_data.to_excel(writer, sheet_name='Zombie Tickets', index=False)
            
            # Sheet 6: SLA Performance
            sla_data = self._build_sla_df()
            if not sla_data.empty:
                sla_data.to_excel(writer, sheet_name='SLA Performance', index=False)
            
            # Sheet 7: Findings
            findings_data = self._build_findings_df()
            if not findings_data.empty:
                findings_data.to_excel(writer, sheet_name='Findings', index=False)
            
            # Sheet 8: Solution Quality
            solutions_data = self._build_solutions_df()
            if not solutions_data.empty:
                solutions_data.to_excel(writer, sheet_name='Solution Quality', index=False)
            
            # Sheet 9: Temporal Trends
            temporal_data = self._build_temporal_df()
            if not temporal_data.empty:
                temporal_data.to_excel(writer, sheet_name='Monthly Trends', index=False)
        
        # Apply formatting
        self._format_excel(excel_path)
        
        return excel_path
    
    def _build_overview_df(self) -> pd.DataFrame:
        """Build overview DataFrame."""
        meta = self.results.get('metadata', {})
        foundation = self.results.get('foundation', {})
        zombies = self.results.get('zombies', {})
        sla = self.results.get('sla', {})
        
        data = [
            {'Metric': 'Total Tickets', 'Value': meta.get('total_tickets', 0)},
            {'Metric': 'Analysis Date', 'Value': meta.get('analyzed_at', '')[:10]},
            {'Metric': 'Date Range', 'Value': f"{foundation.get('date_range', {}).get('earliest', 'N/A')} to {foundation.get('date_range', {}).get('latest', 'N/A')}"},
            {'Metric': 'AI Enabled', 'Value': 'Yes' if meta.get('ai_enabled') else 'No (Fallback)'},
            {'Metric': '', 'Value': ''},
            {'Metric': 'True Zombies', 'Value': zombies.get('true_zombie_count', 0)},
            {'Metric': 'Zombie Rate', 'Value': f"{zombies.get('zombie_rate', 0)}%"},
            {'Metric': 'False Positives Filtered', 'Value': zombies.get('false_positive_count', 0)},
            {'Metric': '', 'Value': ''},
            {'Metric': 'Issue Categories', 'Value': len(self.results.get('categories', {}))},
            {'Metric': f'{UserConfig.ENTITY_NAME_PLURAL.title()}', 'Value': self.results.get('entities', {}).get('total_entities', 0)},
            {'Metric': 'Anomalies Detected', 'Value': len(self.results.get('anomalies', []))},
            {'Metric': '', 'Value': ''},
            {'Metric': 'FRT Compliance', 'Value': f"{sla.get('first_response', {}).get('compliance_rate', 'N/A')}%"},
            {'Metric': 'Resolution Compliance', 'Value': f"{sla.get('resolution', {}).get('compliance_rate', 'N/A')}%"},
        ]
        
        return pd.DataFrame(data)
    
    def _build_categories_df(self) -> pd.DataFrame:
        """Build categories DataFrame."""
        categories = self.results.get('categories', {})
        
        rows = []
        for cat_name, cat_data in sorted(categories.items(), key=lambda x: x[1].get('total_tickets', 0), reverse=True):
            rows.append({
                'Category': cat_name,
                'Description': cat_data.get('description', '')[:100],
                'Total Tickets': cat_data.get('total_tickets', 0),
                'Zombies': cat_data.get('zombie_count', 0),
                'Zombie Rate %': cat_data.get('zombie_rate', 0),
                'Avg Resolution (days)': cat_data.get('avg_resolution_days', ''),
                'Urgent': cat_data.get('by_priority', {}).get('Urgent', 0),
                'High': cat_data.get('by_priority', {}).get('High', 0),
                'Medium': cat_data.get('by_priority', {}).get('Medium', 0),
                'Low': cat_data.get('by_priority', {}).get('Low', 0),
                'Root Causes': ', '.join(cat_data.get('typical_root_causes', [])[:3]),
                'Sample Ticket IDs': ', '.join(map(str, cat_data.get('ticket_ids', [])[:5])),
            })
        
        return pd.DataFrame(rows)
    
    def _build_entities_df(self) -> pd.DataFrame:
        """Build entities DataFrame."""
        entities = self.results.get('entities', {}).get('entities', {})
        
        rows = []
        for entity_name, entity_data in sorted(entities.items(), key=lambda x: x[1].get('total_tickets', 0), reverse=True)[:100]:
            top_issue = entity_data.get('top_issue', ('', 0))
            rows.append({
                UserConfig.ENTITY_NAME.title(): entity_name,
                'Total Tickets': entity_data.get('total_tickets', 0),
                'Zombies': entity_data.get('zombie_count', 0),
                'Zombie Rate %': entity_data.get('zombie_rate', 0),
                'Top Issue': top_issue[0] if isinstance(top_issue, tuple) else top_issue,
                'Top Issue Count': top_issue[1] if isinstance(top_issue, tuple) else '',
                'Avg Resolution (days)': entity_data.get('avg_resolution_days', ''),
                'First Ticket': entity_data.get('date_range', {}).get('first', ''),
                'Last Ticket': entity_data.get('date_range', {}).get('last', ''),
                'Sample IDs': ', '.join(map(str, entity_data.get('ticket_ids', [])[:5])),
            })
        
        return pd.DataFrame(rows)
    
    def _build_anomalies_df(self) -> pd.DataFrame:
        """Build anomalies DataFrame."""
        anomalies = self.results.get('anomalies', [])
        
        rows = []
        for anom in anomalies:
            rows.append({
                'Type': anom.get('type', '').replace('_', ' ').title(),
                'Severity': anom.get('severity', 'medium').title(),
                'Entity': anom.get('entity', ''),
                'Description': anom.get('description', ''),
                'Count/Days': anom.get('count', anom.get('days_apart', '')),
                'Ticket IDs': ', '.join(map(str, anom.get('ticket_ids', [])[:5])),
            })
        
        return pd.DataFrame(rows)
    
    def _build_zombies_df(self) -> pd.DataFrame:
        """Build zombies DataFrame."""
        zombies = self.results.get('zombies', {}).get('true_zombies', [])
        
        rows = []
        for z in zombies[:200]:  # Limit to 200
            rows.append({
                'Ticket ID': z.get('ticket_id'),
                'Subject': z.get('subject', '')[:80],
                'Created': z.get('created_at', '')[:10] if z.get('created_at') else '',
                'Status': z.get('status', ''),
                'Priority': z.get('priority', ''),
                'Reason': z.get('reason', ''),
            })
        
        return pd.DataFrame(rows)
    
    def _build_sla_df(self) -> pd.DataFrame:
        """Build SLA DataFrame."""
        sla = self.results.get('sla', {})
        
        rows = []
        
        # Overall
        frt = sla.get('first_response', {})
        res = sla.get('resolution', {})
        
        rows.append({
            'Priority': 'OVERALL',
            'FRT Avg (hrs)': frt.get('avg', ''),
            'FRT Breaches': frt.get('breach_count', ''),
            'FRT Compliance %': frt.get('compliance_rate', ''),
            'Resolution Avg (hrs)': res.get('avg', ''),
            'Resolution Breaches': res.get('breach_count', ''),
            'Resolution Compliance %': res.get('compliance_rate', ''),
        })
        
        # By priority
        by_priority = sla.get('by_priority', {})
        for priority in ['Urgent', 'High', 'Medium', 'Low']:
            p_data = by_priority.get(priority, {})
            p_frt = p_data.get('first_response', {})
            p_res = p_data.get('resolution', {})
            
            rows.append({
                'Priority': priority,
                'FRT Avg (hrs)': p_frt.get('avg', ''),
                'FRT Breaches': p_frt.get('breach_count', ''),
                'FRT Compliance %': p_frt.get('compliance_rate', ''),
                'Resolution Avg (hrs)': p_res.get('avg', ''),
                'Resolution Breaches': p_res.get('breach_count', ''),
                'Resolution Compliance %': p_res.get('compliance_rate', ''),
            })
        
        return pd.DataFrame(rows)
    
    def _build_findings_df(self) -> pd.DataFrame:
        """Build findings DataFrame."""
        findings = self.results.get('findings', [])
        
        rows = []
        for f in findings:
            rows.append({
                'Type': f.get('type', '').replace('_', ' ').title(),
                'Title': f.get('title', ''),
                'Description': f.get('description', '')[:150],
                'Confidence': f.get('confidence', '').title(),
                'Severity': f.get('severity', '').title(),
                'Evidence Count': f.get('evidence_count', 0),
                'Recommendation': f.get('recommendation', ''),
                'Sample IDs': ', '.join(map(str, f.get('ticket_ids', [])[:5])),
            })
        
        return pd.DataFrame(rows)
    
    def _build_solutions_df(self) -> pd.DataFrame:
        """Build solutions DataFrame."""
        solutions = self.results.get('solutions', {})
        
        rows = []
        
        # Summary row
        rows.append({
            'Type': 'SUMMARY',
            'Ticket ID': '',
            'Category': '',
            'Score': solutions.get('average_score', ''),
            'Rating': f"Analyzed: {solutions.get('analyzed', 0)}",
            'Factors': '',
        })
        
        # Best solutions
        for sol in solutions.get('best_solutions', [])[:10]:
            rows.append({
                'Type': 'Best',
                'Ticket ID': sol.get('ticket_id', ''),
                'Category': sol.get('category', ''),
                'Score': sol.get('score', ''),
                'Rating': sol.get('rating', ''),
                'Factors': ', '.join(sol.get('factors', [])),
            })
        
        # Poor solutions
        for sol in solutions.get('poor_solutions', [])[:10]:
            rows.append({
                'Type': 'Needs Improvement',
                'Ticket ID': sol.get('ticket_id', ''),
                'Category': sol.get('category', ''),
                'Score': sol.get('score', ''),
                'Rating': sol.get('rating', ''),
                'Factors': ', '.join(sol.get('factors', [])),
            })
        
        return pd.DataFrame(rows)
    
    def _build_temporal_df(self) -> pd.DataFrame:
        """Build temporal trends DataFrame."""
        temporal = self.results.get('temporal', {})
        monthly = temporal.get('monthly_volume', {})
        
        rows = []
        for month, count in sorted(monthly.items()):
            rows.append({
                'Month': month,
                'Ticket Count': count,
            })
        
        return pd.DataFrame(rows)
    
    def _format_excel(self, path: Path):
        """Apply professional formatting to Excel."""
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
            
            wb = load_workbook(path)
            
            # Styles
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            for sheet in wb.worksheets:
                # Auto-width columns
                for column in sheet.columns:
                    max_length = 0
                    column_letter = get_column_letter(column[0].column)
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    sheet.column_dimensions[column_letter].width = adjusted_width
                
                # Format header row
                for cell in sheet[1]:
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = Alignment(horizontal='center')
                    cell.border = border
                
                # Format data rows
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.border = border
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                
                # Freeze header row
                sheet.freeze_panes = 'A2'
            
            wb.save(path)
            
        except Exception as e:
            pass  # Formatting failed, but file is still usable
    
    def generate_markdown(self) -> Path:
        """Generate Markdown summary."""
        md_path = self.output_dir / 'analysis_summary.md'
        
        meta = self.results.get('metadata', {})
        foundation = self.results.get('foundation', {})
        zombies = self.results.get('zombies', {})
        categories = self.results.get('categories', {})
        anomalies = self.results.get('anomalies', [])
        sla = self.results.get('sla', {})
        findings = self.results.get('findings', [])
        
        lines = [
            f"# FTEX Analysis Report",
            f"",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Product:** {UserConfig.PRODUCT_NAME}",
            f"**AI Enabled:** {'Yes' if meta.get('ai_enabled') else 'No (Fallback mode)'}",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tickets Analyzed | {meta.get('total_tickets', 0):,} |",
            f"| Date Range | {foundation.get('date_range', {}).get('earliest', 'N/A')} to {foundation.get('date_range', {}).get('latest', 'N/A')} |",
            f"| **True Zombie Tickets** | **{zombies.get('true_zombie_count', 0):,}** ({zombies.get('zombie_rate', 0)}%) |",
            f"| Issue Categories | {len(categories)} |",
            f"| {UserConfig.ENTITY_NAME_PLURAL.title()} Tracked | {self.results.get('entities', {}).get('total_entities', 0)} |",
            f"| Anomalies Detected | {len(anomalies)} |",
            f"",
        ]
        
        # Findings
        if findings:
            lines.extend([
                f"## Key Findings",
                f"",
            ])
            
            for i, finding in enumerate(findings[:10], 1):
                confidence = finding.get('confidence', 'medium')
                confidence_icon = "🟢" if confidence == 'high' else "🟡" if confidence == 'medium' else "🔴"
                
                lines.append(f"### {i}. {finding.get('title', 'Finding')}")
                lines.append(f"")
                lines.append(f"**Confidence:** {confidence_icon} {confidence.title()} ({finding.get('evidence_count', 0)} evidence tickets)")
                lines.append(f"")
                lines.append(f"{finding.get('description', '')}")
                lines.append(f"")
                
                if finding.get('root_cause'):
                    lines.append(f"**Root Cause:** {finding['root_cause']}")
                    lines.append(f"")
                
                if finding.get('recommendation'):
                    lines.append(f"**Recommendation:** {finding['recommendation']}")
                    lines.append(f"")
                
                if finding.get('ticket_ids'):
                    lines.append(f"**Sample Tickets:** {', '.join(map(str, finding['ticket_ids'][:5]))}")
                    lines.append(f"")
        
        # Anomalies
        if anomalies:
            lines.extend([
                f"## Anomalies Detected",
                f"",
            ])
            
            for anom in anomalies[:10]:
                severity_icon = "🔴" if anom.get('severity') == 'high' else "🟡"
                lines.append(f"- {severity_icon} **{anom.get('type', '').replace('_', ' ').title()}**: {anom.get('description', '')}")
            
            lines.append(f"")
        
        # Top Categories
        if categories:
            lines.extend([
                f"## Issue Categories",
                f"",
                f"| Category | Tickets | Zombies | Avg Resolution |",
                f"|----------|---------|---------|----------------|",
            ])
            
            for cat_name, cat_data in sorted(categories.items(), key=lambda x: x[1].get('total_tickets', 0), reverse=True)[:10]:
                res_days = cat_data.get('avg_resolution_days')
                res_str = f"{res_days:.1f}d" if res_days else "—"
                lines.append(f"| {cat_name} | {cat_data.get('total_tickets', 0)} | {cat_data.get('zombie_count', 0)} | {res_str} |")
            
            lines.append(f"")
        
        # SLA
        frt = sla.get('first_response', {})
        res = sla.get('resolution', {})
        
        lines.extend([
            f"## SLA Performance",
            f"",
            f"| Metric | Average | Compliance |",
            f"|--------|---------|------------|",
            f"| First Response | {frt.get('avg', 'N/A')} hrs | {frt.get('compliance_rate', 'N/A')}% |",
            f"| Resolution | {res.get('avg', 'N/A')} hrs | {res.get('compliance_rate', 'N/A')}% |",
            f"",
        ])
        
        # Footer
        lines.extend([
            f"---",
            f"",
            f"*Report generated by FTEX Analyzer v6.0*",
        ])
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return md_path
    
    def generate_pdf(self, md_path: Path) -> Optional[Path]:
        """Generate PDF from Markdown."""
        if not PDF_AVAILABLE:
            return None
        
        pdf_path = self.output_dir / 'analysis_summary.pdf'
        
        try:
            # Read markdown
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            try:
                from markdown import markdown
                html_content = markdown(md_content, extensions=['tables', 'fenced_code'])
            except ImportError:
                # Basic conversion
                html_content = f"<pre>{md_content}</pre>"
            
            # Wrap in HTML template
            html_full = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        line-height: 1.6;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        color: #333;
                    }}
                    h1 {{ color: #2F5496; border-bottom: 2px solid #2F5496; padding-bottom: 10px; }}
                    h2 {{ color: #2F5496; margin-top: 30px; }}
                    h3 {{ color: #404040; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #2F5496; color: white; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                    hr {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Convert to PDF
            try:
                from weasyprint import HTML
                HTML(string=html_full).write_pdf(str(pdf_path))
            except ImportError:
                try:
                    import pdfkit
                    pdfkit.from_string(html_full, str(pdf_path))
                except:
                    return None
            
            return pdf_path
            
        except Exception as e:
            return None
    
    def generate_json(self) -> Path:
        """Generate raw JSON data."""
        json_path = self.output_dir / 'analysis_data.json'
        
        # Clean results for JSON serialization
        clean_results = {}
        
        for key, value in self.results.items():
            if key in ['zombies']:
                # Remove ticket objects, keep IDs
                clean_value = {k: v for k, v in value.items() if k not in ['true_zombies', 'false_positives']}
                clean_value['zombie_ticket_ids'] = [z.get('ticket_id') for z in value.get('true_zombies', [])]
                clean_results[key] = clean_value
            elif key == 'entities':
                # Simplify entity data
                clean_results[key] = {
                    'total_entities': value.get('total_entities', 0),
                    'top_entities': list(value.get('entities', {}).keys())[:20],
                }
            else:
                clean_results[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        return json_path


# =============================================================================
# MAIN
# =============================================================================

def load_tickets(path: str) -> List[dict]:
    """Load tickets from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('tickets', data.get('data', []))
    
    raise ValueError(f"Unknown data format in {path}")


def main():
    parser = argparse.ArgumentParser(
        description='FTEX Deep Analyzer - AI-powered ticket analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--input', '-i', default='output/tickets.json',
                       help='Input tickets JSON file')
    parser.add_argument('--output', '-o', default='reports',
                       help='Output directory')
    parser.add_argument('--no-ai', action='store_true',
                       help='Disable AI (statistical only)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear category cache')
    
    args = parser.parse_args()
    
    # Initialize UI
    ui = TerminalUI()
    ui.print_header()
    
    # Check AI
    ollama = OllamaClient()
    ai_available = not args.no_ai and ollama.available
    ui.print_config(ai_available, ollama.model if ai_available else None)
    
    # Clear cache if requested
    if args.clear_cache:
        cache_path = Path(UserConfig.CACHE_FILE)
        if cache_path.exists():
            cache_path.unlink()
            ui.print_info("Cache cleared")
    
    # Load tickets
    try:
        tickets = load_tickets(args.input)
        ui.print_loading(args.input, len(tickets))
    except FileNotFoundError:
        ui.print_error(f"File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        ui.print_error(f"Failed to load tickets: {e}")
        sys.exit(1)
    
    if not tickets:
        ui.print_error("No tickets found in file")
        sys.exit(1)
    
    # Run analysis
    engine = AnalysisEngine(tickets, use_ai=ai_available)
    
    if ui.console and RICH_AVAILABLE:
        # Rich progress
        with ui.create_progress() as progress:
            task = progress.add_task("Analyzing...", total=7)
            
            def progress_callback(stage_type, message, current=None, total=None):
                if stage_type == "stage":
                    progress.update(task, description=message, completed=current)
                elif stage_type == "info":
                    ui.print_info(message)
            
            results = engine.run_analysis(progress_callback)
    else:
        # Plain progress
        def progress_callback(stage_type, message, current=None, total=None):
            if stage_type == "stage":
                ui.print_stage(message, current or 0, total or 7)
            elif stage_type == "info":
                ui.print_info(message)
        
        results = engine.run_analysis(progress_callback)
    
    # Print results summary
    ui.print_results_summary(results)
    
    # Generate reports
    output_dir = Path(args.output)
    generator = ReportGenerator(results, output_dir)
    outputs = generator.generate_all()
    
    # Print outputs
    ui.print_outputs(outputs)
    ui.print_completion()


if __name__ == '__main__':
    main()