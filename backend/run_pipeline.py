#!/usr/bin/env python3
"""
Standalone Pipeline Runner — Financial Intelligence Platform
=============================================================
Runs the entire agentic workflow locally without web hosting dependencies.

Usage:
    python run_pipeline.py --file <mizan.xlsx> [--turkish] [--output-dir ./output]

Flow:
    1. Parse Excel → standardized_mizan
    2. Graph: data_ingestion → quant_analyst → verifier ↔ retry → network_mapper → strategist
    3. (Optional) translator — if --turkish flag is set
    4. Evaluation rubric scoring
    5. Generate HTML report (English)
    6. Generate EN PDF (always)
    7. Generate TR PDF (if --turkish)
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend/ is on the Python path
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from excel_parser import parse_mizan_excel, extract_entities, extract_company_name, predict_sector
from graph import build_standalone_graph
from agents.evaluation import rubric
from pdf_generator import save_report_pdf
from html_generator import generate_report_html
from graph_visualizer import save_network_html

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_pipeline(file_path: str, generate_turkish: bool = False, output_dir: str = "./output", no_graph: bool = False):
    """
    Execute the full financial analysis pipeline.

    Args:
        file_path: Path to the Mizan Excel (.xlsx) file
        generate_turkish: If True, translate report to Turkish and generate TR PDF
        output_dir: Directory for output files (HTML, PDFs)
        no_graph: If True, skip generating interactive network HTML graph
    """
    logger.info("=" * 60)
    logger.info("  FINANCIAL INTELLIGENCE PIPELINE — STANDALONE MODE")
    logger.info(f"  File: {file_path}")
    logger.info(f"  Turkish: {'YES' if generate_turkish else 'NO'}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Graph: {'SKIPPED' if no_graph else 'ENABLED'}")
    logger.info("=" * 60)

    # ── Step 1: Parse Excel ──
    logger.info("[1/7] Parsing Mizan Excel file...")
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    if not file_path.suffix.lower() == ".xlsx":
        logger.error(f"Only .xlsx files are supported. Got: {file_path.suffix}")
        sys.exit(1)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    mizan_df = parse_mizan_excel(file_bytes)
    logger.info(f"  Parsed {len(mizan_df)} accounts")

    # ── Step 2: Extract metadata ──
    logger.info("[2/7] Extracting company metadata...")
    company_name = extract_company_name(file_path.name)
    entities = extract_entities(mizan_df)
    customers = entities["customers"]
    suppliers = entities["suppliers"]
    logger.info(f"  Company: {company_name}")
    logger.info(f"  Customers: {len(customers)}, Suppliers: {len(suppliers)}")

    # Predict sector via LLM
    sector = predict_sector(customers, suppliers)
    logger.info(f"  Sector: {sector}")

    # ── Step 3: Build and invoke graph ──
    logger.info("[3/7] Building agent graph...")
    graph = build_standalone_graph(generate_turkish=generate_turkish)

    state = {
        "tax_id": "standalone",
        "company_name": company_name,
        "sector": sector,
        "mizan_data": mizan_df.to_dict(orient="records"),
        "standardized_mizan": None,
        "financial_ratios": None,
        "verification_status": None,
        "verification_errors": "",
        "retry_count": 0,
        "network_data": None,
        "strategy_report": None,
        "product_signals": None,
        "translated_report": None,
        "report_language": "TR" if generate_turkish else "EN",
        "chat_history": [],
        "chat_response": None,
        # Observability
        "agent_metrics": {},
        "execution_timeline": [],
        "pipeline_start_time": datetime.now(timezone.utc).isoformat(),
        "error_log": [],
    }

    logger.info("[4/7] Running agent pipeline...")
    result = graph.invoke(state)

    # ── Step 4: Evaluation ──
    logger.info("[5/7] Running evaluation rubric...")
    eval_result = rubric.score_pipeline(result)
    result["evaluation"] = eval_result
    result["company_name"] = company_name
    result["sector"] = sector
    logger.info(f"  Overall Score: {eval_result.get('overall_score', '?')}")

    # ── Step 5: Generate outputs ──
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')

    # HTML report (always English)
    logger.info("[6/7] Generating HTML report...")
    html_content = generate_report_html(result, language="EN")
    html_path = os.path.join(output_dir, f"{safe_name}_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"  EN HTML: {html_path}")

    # Interactive network graph
    if not no_graph:
        network_data = result.get("network_data", {})
        if network_data and network_data.get("nodes"):
            graph_path = save_network_html(network_data, output_dir, company_name)
            logger.info(f"  Network Graph: {graph_path}")
        else:
            logger.warning("  No network data available for graph visualization")
    else:
        logger.info("  Network Graph generation skipped via --no-graph")
    
    if generate_turkish:
        tr_html_content = generate_report_html(result, language="TR")
        tr_html_path = os.path.join(output_dir, f"{safe_name}_Report_TR.html")
        with open(tr_html_path, "w", encoding="utf-8") as f:
            f.write(tr_html_content)
        logger.info(f"  TR HTML: {tr_html_path}")

    # PDF reports
    logger.info("[7/7] Generating PDF reports...")
    en_pdf_path = save_report_pdf(result, output_dir, company_name, language="EN")
    logger.info(f"  EN PDF: {en_pdf_path}")

    if generate_turkish:
        tr_pdf_path = save_report_pdf(result, output_dir, company_name, language="TR")
        logger.info(f"  TR PDF: {tr_pdf_path}")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE ✅")
    logger.info(f"  Company: {company_name} ({sector})")
    logger.info(f"  Verification: {result.get('verification_status', '?')}")
    logger.info(f"  Eval Score: {eval_result.get('overall_score', '?')}")
    logger.info(f"  Outputs in: {output_dir}/")
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Financial Intelligence Pipeline — Standalone Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --file mizan_ABC.xlsx
  python run_pipeline.py --file mizan_ABC.xlsx --turkish
  python run_pipeline.py --file mizan_ABC.xlsx --turkish --output-dir ./reports
        """,
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the Mizan Excel file (.xlsx)",
    )
    parser.add_argument(
        "--turkish", "-t",
        action="store_true",
        default=False,
        help="Generate Turkish translation of the report (default: English only)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Output directory for HTML and PDF files (default: ./output)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        default=False,
        help="Skip generating the interactive network HTML graph",
    )

    args = parser.parse_args()
    run_pipeline(
        file_path=args.file,
        generate_turkish=args.turkish,
        output_dir=args.output_dir,
        no_graph=args.no_graph,
    )

if __name__ == "__main__":
    main()
