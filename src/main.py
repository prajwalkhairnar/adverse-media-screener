# src/main.py

"""
Command-line interface (CLI) entry point for the Adverse Media Screener.

This module sets up the environment, initializes the LLM workflow, and handles
input/output via the 'screen' command.
"""
import click
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table

# --- Local Imports ---
# AMENDMENT: Import config.settings as a module since config/ is a sibling to src/.
import config.settings as settings

from src.utils.logger import get_logger
from src.llm import LLMFactory, CostTracker
from src.models.inputs import ScreeningQuery
from src.models.outputs import ScreeningResult
from src.graph.workflow import AdverseMediaWorkflow

# Initialize logger and console immediately
logger = get_logger("CLI")
console = Console()

# --- Helper Functions for Output Formatting ---


def print_summary_table(result: ScreeningResult):
    """Prints a summary table of the screening results."""
    # This line is fine, as 'bold' is opened and closed correctly.
    console.rule(f"[bold]{result.decision} Screening Summary[/bold]", style="bold magenta")

    table = Table(
        title="Processing Metadata",
        show_header=True,
        header_style="bold blue",
        padding=(0, 1),
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    # Core Decision - FIX APPLIED HERE
    decision_color = 'red' if result.decision != 'NO_MATCH' else 'green'
    # Use f"[{decision_color} bold]" to apply both style and color, and use [/] to close
    table.add_row(
        "Screening Decision", 
        f"[{decision_color} bold]{result.decision}[/]", 
        end_section=True
    )

    # Match Details
    table.add_row("Match Confidence", f"{result.match_assessment.confidence} ({result.match_assessment.match_probability:.2f})")
    
    # Sentiment Details (only if present) - FIX APPLIED HERE
    if result.sentiment_assessment:
        color = "red" if result.sentiment_assessment.is_adverse_media else "green"
        # Combine color and bold inside the tag: [color bold]...[/]
        table.add_row("Adverse Media Found", 
                      f"[{color} bold]{result.sentiment_assessment.classification} "
                      f"(Severity: {result.sentiment_assessment.severity})[/]")
    else:
        table.add_row("Adverse Media Found", "N/A (No match found)")

    # Processing Details
    meta = result.processing_metadata
    table.add_row("Total Duration", f"{meta.total_duration_ms / 1000:.2f} seconds")
    table.add_row("Total Cost", f"USD [green]${meta.estimated_cost_usd:.4f}[/green]")
    table.add_row("Total Tokens", f"{meta.total_tokens}")
    table.add_row("Provider/Model", f"{meta.llm_provider}/{meta.llm_model}")
    table.add_row("Errors/Warnings", f"[red]{len(meta.errors_encountered)}[/red] Errors, [yellow]{len(meta.warnings)}[/yellow] Warnings")

    console.print(table)

def print_full_report(report_text: str):
    """Prints the final human-readable report."""
    console.rule("[bold]Final Compliance Report (LLM Generated)[/bold]", style="bold cyan")
    console.print(report_text)


# --- CLI Command Group ---


@click.group()
def cli():
    """Adverse Media Screening System CLI."""
    # Settings are loaded here once for configuration
    settings.get_settings()


@cli.command()
@click.option("--name", required=True, type=str, help="Full name of the person being screened.")
@click.option("--dob", required=True, type=str, help="Date of birth in YYYY-MM-DD format.")
@click.option("--url", required=True, type=str, help="News article URL to be screened.")
@click.option(
    "--provider",
    # Use the module alias to access the LLMProvider Enum
    type=click.Choice([p.value for p in settings.LLMProvider]),
    default=None,
    help="Optional override for the default LLM provider.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Optional override for the default LLM model name.",
)
def screen(name: str, dob: str, url: str, provider: str, model: str):
    """
    Executes an adverse media screening against a single news article URL.
    """
    # Use the module alias to retrieve settings
    settings_instance = settings.get_settings()

    # 1. Prepare the input query model
    try:
        query = ScreeningQuery(
            name=name,
            dob=dob,
            url=url,
            # Use the module alias to reference LLMProvider
            provider=settings.LLMProvider(provider) if provider else None,
            model=model,
        )
    except Exception as e:
        console.print(f"[bold red]Input Error:[/bold red] Could not validate input: {e}")
        return

    logger.info(f"Starting screening for {query.name} (DOB: {query.dob}) against {query.url}")
    
    # 2. Initialize Core Components
    llm_factory = LLMFactory(settings_instance)
    cost_tracker = CostTracker()
    
    # 3. Initialize and Run Workflow
    try:
        workflow = AdverseMediaWorkflow(settings_instance, llm_factory, cost_tracker)
        final_state = workflow.run_workflow(query)
        
    except ConnectionError as e:
        # Catch critical errors like all LLMs or the article fetcher failing
        logger.error(f"Critical Workflow Failure: {e}")
        console.print(f"\n[bold red]CRITICAL FAILURE:[/bold red] The workflow could not complete due to a major error.")
        return
    except Exception as e:
        logger.error(f"Workflow ended with an unhandled exception: {e}", exc_info=True)
        final_state = {"errors": [f"Workflow interrupted by unhandled exception: {e}"]}


    # 4. Final Output Processing
    console.print("\n" + "="*80)

    console.print(f"[bold yellow]DEBUG:[/bold yellow] Final State Keys: {list(final_state.keys())}")
    breakpoint()
    
    if "final_screening_result" in final_state:
        result: ScreeningResult = final_state["final_screening_result"]
        
        # Print structured summary
        print_summary_table(result)
        
        # Print final report text
        print_full_report(result.report)
        
        # Optional: Save raw structured output for audit
        with open(f"src/outputs/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        
    else:
        # Handle case where report generation failed completely
        console.print("[bold red]Screening Failed.[/bold red]")
        if final_state.get("errors"):
            console.print("\n[bold]Errors Encountered:[/bold]")
            for error in final_state["errors"]:
                console.print(f"  [red]- {error}[/red]")


# --- Main Execution ---

if __name__ == "__main__":
    cli()