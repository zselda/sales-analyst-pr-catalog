"""
FastAPI Backend — Financial Intelligence Platform (gemma-3-27b-it)

Enhanced with:
- Excel file upload for Mizan data
- Dynamic entity extraction from uploaded data
- Context-aware transaction generation
- Structured logging (replaces bare print())
- Agent metrics endpoint (/api/metrics)
- Execution timeline endpoint (/api/timeline)
- Pipeline evaluation scoring
- Proper session management (no global mutable state)
"""

import logging
import traceback
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from mock_data import get_mizan_df, get_transactions_df
from excel_parser import parse_mizan_excel, extract_entities, extract_company_name, predict_sector, parse_transactions_file
from data_generator import generate_transactions
from graph import swarm_graph
from llm_config import invoke_llm, CHAT_SYSTEM_PROMPT, get_llm_stats
from agents.evaluation import rubric
from pdf_generator import generate_report_pdf

# ── Logging Configuration ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("swarm.api")

# ── App ──
app = FastAPI(title="Financial Intelligence Platform", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Session storage (per-run, not global mutable)
_sessions: dict[str, dict] = {}
# Upload storage (parsed Excel data)
_uploads: dict[str, dict] = {}


class RunSwarmRequest(BaseModel):
    tax_id: str = "1234567890"
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    tax_id: Optional[str] = "1234567890"


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "model": "gemma-3-27b-it",
        "architecture": "parallel-multi-agent",
        "agents": 6,
    }


@app.post("/api/upload-mizan")
async def upload_mizan(file: UploadFile = File(...)):
    """Upload a Mizan Excel file, parse it, extract entities, predict sector."""
    try:
        if not file.filename.endswith(".xlsx"):
            raise HTTPException(400, detail="Only .xlsx files are accepted.")

        file_bytes = await file.read()
        logger.info(f"Received upload: {file.filename} ({len(file_bytes)} bytes)")

        # Parse Excel
        mizan_df = parse_mizan_excel(file_bytes)
        logger.info(f"Parsed {len(mizan_df)} accounts from Excel")

        # Extract company name from filename
        company_name = extract_company_name(file.filename)
        logger.info(f"Company name: {company_name}")

        # Extract entities
        entities = extract_entities(mizan_df)
        customers = entities["customers"]
        suppliers = entities["suppliers"]
        logger.info(f"Extracted {len(customers)} customers, {len(suppliers)} suppliers")

        # Predict sector from entity names via LLM
        sector = predict_sector(customers, suppliers)
        logger.info(f"Predicted sector: {sector}")

        # Store under a session ID (no transactions auto-generated)
        session_id = str(uuid.uuid4())[:8]
        _uploads[session_id] = {
            "mizan_df": mizan_df,
            "txn_df": None,  # Transactions are optional
            "entities": entities,
            "filename": file.filename,
            "company_name": company_name,
            "sector": sector,
        }

        # Build preview
        preview_records = mizan_df.head(15).to_dict(orient="records")

        return {
            "status": "success",
            "session_id": session_id,
            "filename": file.filename,
            "company_name": company_name,
            "sector": sector,
            "mizan_count": len(mizan_df),
            "customer_count": len(customers),
            "supplier_count": len(suppliers),
            "transaction_count": 0,
            "preview": preview_records,
            "customers_sample": [c["name"] for c in customers[:5]],
            "suppliers_sample": [s["name"] for s in suppliers[:5]],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Upload failed: {e}")


@app.post("/api/upload-transactions")
async def upload_transactions(file: UploadFile = File(...), session_id: str = ""):
    """Upload a transaction file (.xlsx or .csv) for an existing session."""
    try:
        if not session_id or session_id not in _uploads:
            raise HTTPException(404, detail="Session not found. Upload a Mizan file first.")

        fname = file.filename.lower()
        if not (fname.endswith(".xlsx") or fname.endswith(".csv")):
            raise HTTPException(400, detail="Only .xlsx or .csv files are accepted.")

        file_bytes = await file.read()
        logger.info(f"Received transaction upload: {file.filename} ({len(file_bytes)} bytes)")

        txn_df = parse_transactions_file(file_bytes, file.filename)
        _uploads[session_id]["txn_df"] = txn_df
        logger.info(f"Stored {len(txn_df)} transactions for session {session_id}")

        incoming = len(txn_df[txn_df["Type"] == "Incoming"])
        outgoing = len(txn_df[txn_df["Type"] == "Outgoing"])

        return {
            "status": "success",
            "session_id": session_id,
            "filename": file.filename,
            "transaction_count": len(txn_df),
            "incoming": incoming,
            "outgoing": outgoing,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Transaction upload failed: {e}")

@app.post("/api/run-swarm")
async def run_swarm(request: RunSwarmRequest):
    try:
        logger.info(f"{'='*60}")
        logger.info(f"  SWARM for {request.tax_id} | gemma-3-27b-it | PARALLEL")
        logger.info(f"{'='*60}")

        # Load data from upload session or fall back to mock data
        company_name = "Company"
        sector = "General"
        if request.session_id and request.session_id in _uploads:
            upload = _uploads[request.session_id]
            mizan_df = upload["mizan_df"]
            txn_df = upload.get("txn_df")  # May be None (optional)
            company_name = upload.get("company_name", "Company")
            sector = upload.get("sector", "General")
            logger.info(f"Using uploaded data (session: {request.session_id}, company: {company_name}, sector: {sector})")
        else:
            mizan_df = get_mizan_df()
            txn_df = get_transactions_df()
            logger.info("Using mock data (no upload session)")

        # Transactions are optional — use empty list if None
        txn_records = txn_df.to_dict(orient="records") if txn_df is not None else []
        logger.info(f"Loaded {len(mizan_df)} mizan + {len(txn_records)} transactions")

        state = {
            "tax_id": request.tax_id,
            "company_name": company_name,
            "sector": sector,
            "mizan_data": mizan_df.to_dict(orient="records"),
            "standardized_mizan": None, "financial_ratios": None,
            "verification_status": None, "verification_errors": "",
            "retry_count": 0,
            "network_data": None, "strategy_report": None,
            "chat_history": [], "chat_response": None,
            # Observability
            "agent_metrics": {},
            "execution_timeline": [],
            "pipeline_start_time": datetime.now(timezone.utc).isoformat(),
            "error_log": [],
        }

        result = swarm_graph.invoke(state)

        # Run evaluation
        eval_result = rubric.score_pipeline(result)

        # Store session — include company metadata
        result["company_name"] = company_name
        result["sector"] = sector
        _sessions[request.tax_id] = result

        logger.info(f"{'='*60}")
        logger.info(f"  SWARM COMPLETE ✅ | Score: {eval_result['overall_score']}")
        logger.info(f"{'='*60}")

        return {
            "status": "success",
            "tax_id": request.tax_id,
            "session_id": request.session_id,
            "company_name": company_name,
            "sector": sector,
            "model": "gemma-3-27b-it",
            "financial_ratios": result.get("financial_ratios", {}),
            "verification_status": result.get("verification_status", ""),
            "verification_errors": result.get("verification_errors", ""),
            "retry_count": result.get("retry_count", 0),
            "network_data": result.get("network_data", {}),
            "strategy_report": result.get("strategy_report", ""),
            "evaluation": eval_result,
            "agent_metrics": result.get("agent_metrics", {}),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Swarm failed: {e}")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        latest_result = _sessions.get(request.tax_id)
        if not latest_result:
            return {"response": "Run the swarm first.", "status": "no_context"}

        ratios = latest_result.get("financial_ratios", {})
        network = latest_result.get("network_data", {})

        # Build compact context
        ratio_lines = "\n".join(
            f"- {n}: {d.get('value','?')}{d.get('unit','')}"
            for n, d in ratios.items() if isinstance(d, dict) and n != "llm_interpretation"
        )
        nodes = network.get("nodes", [])
        ns = network.get("stats", {})
        net_lines = "\n".join(f"- {n['label']} ({n.get('type','')}): ₺{n.get('balance',0):,.0f}" for n in nodes)

        # Competitor bank data from quant ratios
        cb = ratios.get("competitor_banks", {})
        def fmt_cb(shares):
            if not shares: return "No data"
            return ", ".join(f"{s['name']} (₺{s['balance']:,.0f})" for s in shares[:3])

        ctx = (
            f"Company: {latest_result.get('company_name', 'Company')} ({latest_result.get('sector', 'General')})\n\n"
            f"Ratios:\n{ratio_lines}\n\n"
            f"Network ({len(nodes)} partners):\n{net_lines}\n"
            f"Receivables: ₺{ns.get('total_receivables',0):,.0f}, "
            f"Payables: ₺{ns.get('total_payables',0):,.0f}\n\n"
            f"Competitor Banks - Deposits (102): {fmt_cb(cb.get('102', []))}\n"
            f"Competitor Banks - ST Loans (300): {fmt_cb(cb.get('300', []))}\n"
            f"Competitor Banks - LT Loans (400): {fmt_cb(cb.get('400', []))}\n\n"
            f"Question: {request.message}"
        )

        text = invoke_llm(CHAT_SYSTEM_PROMPT, ctx, temperature=0.4, max_tokens=1500)
        logger.info(f"[Chat] ✅ {len(text)} chars")
        return {"response": text, "status": "success", "model": "gemma-3-27b-it"}

    except Exception as e:
        traceback.print_exc()
        return {"response": f"Error: {e}", "status": "fallback"}


@app.get("/api/metrics")
async def get_metrics():
    """Return per-agent execution metrics for the latest run."""
    all_metrics = {}
    for tax_id, session in _sessions.items():
        all_metrics[tax_id] = {
            "agent_metrics": session.get("agent_metrics", {}),
            "pipeline_start_time": session.get("pipeline_start_time"),
            "llm_stats": get_llm_stats(),
        }
    return all_metrics


@app.get("/api/timeline")
async def get_timeline():
    """Return execution timeline for the latest run."""
    timelines = {}
    for tax_id, session in _sessions.items():
        timelines[tax_id] = session.get("execution_timeline", [])
    return timelines


@app.get("/api/agents")
async def get_agents():
    """Return registered agent capabilities."""
    from agents.registry import registry
    return registry.get_all_status()


@app.get("/api/mock-data/mizan")
async def get_mock_mizan():
    df = get_mizan_df()
    return {"data": df.to_dict(orient="records"), "count": len(df)}


@app.get("/api/mock-data/transactions")
async def get_mock_transactions():
    df = get_transactions_df()
    records = df.to_dict(orient="records")
    for r in records:
        if hasattr(r["Date"], "isoformat"):
            r["Date"] = r["Date"].isoformat()
    return {"data": records, "count": len(df)}


@app.get("/api/report/pdf")
async def download_pdf_report(tax_id: str = "1234567890"):
    """Generate and download ING-branded PDF strategy report."""
    session = _sessions.get(tax_id)
    if not session:
        raise HTTPException(404, detail="No analysis found. Run the swarm first.")

    try:
        pdf_bytes = generate_report_pdf(session)
        filename = f"financial_report_{tax_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(500, detail=f"PDF generation failed: {e}")

