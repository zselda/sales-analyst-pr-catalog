"""
LLM Configuration — gemma-3-27b-it via Google AI Studio
========================================================
Uses the direct google-genai SDK.
Gemma does NOT support system_instruction, so we embed the
system prompt into the user text as a <ROLE> preamble.

Enhanced with:
- Environment variable for API key (falls back to hardcoded for dev)
- Retry with exponential backoff
- Token usage tracking
- Request timeout
- Structured output support
"""

import os
import time
import json
import logging
from typing import Optional

from google import genai

logger = logging.getLogger("swarm.llm")

# ── Configuration ──────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma-3-27b-it")
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "30"))

_client = genai.Client(api_key=GOOGLE_API_KEY)

# ── Token tracking ─────────────────────────────────────────────────
_total_llm_calls = 0
_total_tokens_used = 0


def get_llm_stats() -> dict:
    """Return global LLM usage statistics."""
    return {
        "total_llm_calls": _total_llm_calls,
        "total_tokens_estimated": _total_tokens_used,
        "model": MODEL_NAME,
    }


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    retries: int = MAX_RETRIES,
) -> str:
    """
    Call gemma-3-27b-it with system prompt embedded in user text.

    Features:
    - Retry with exponential backoff on transient failures
    - Token usage estimation
    - Structured logging
    """
    global _total_llm_calls, _total_tokens_used

    combined = f"<ROLE>\n{system_prompt}\n</ROLE>\n\n<TASK>\n{user_prompt}\n</TASK>"

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            start_time = time.monotonic()
            resp = _client.models.generate_content(
                model=MODEL_NAME,
                contents=combined,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

            text = resp.text
            estimated_tokens = len(text.split()) + len(combined.split())

            _total_llm_calls += 1
            _total_tokens_used += estimated_tokens

            logger.info(
                f"[LLM] ✅ {MODEL_NAME} | {elapsed_ms}ms | ~{estimated_tokens} tokens | attempt {attempt}"
            )
            return text

        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = 2 ** attempt
                logger.warning(
                    f"[LLM] ⚠️ Attempt {attempt} failed: {e} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                logger.error(f"[LLM] ❌ All {retries} attempts failed: {e}")

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def invoke_llm_structured(
    system_prompt: str,
    user_prompt: str,
    expected_keys: list[str],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> dict:
    """
    Call LLM and attempt to parse response as JSON.

    If the response isn't valid JSON, wraps the text in a dict with key 'text'.

    Args:
        system_prompt: System role prompt
        user_prompt: Task prompt (should ask for JSON output)
        expected_keys: Keys expected in JSON response
        temperature: LLM temperature
        max_tokens: Max output tokens

    Returns:
        dict with parsed JSON or {'text': raw_response}
    """
    json_instruction = (
        f"\n\nRespond ONLY with a valid JSON object containing these keys: "
        f"{expected_keys}. No markdown, no explanation."
    )
    text = invoke_llm(system_prompt, user_prompt + json_instruction, temperature, max_tokens)

    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"[LLM] Could not parse JSON response, returning as text")

    return {"text": text}


# ── System Prompts ──────────────────────────────────────────────────


QUANT_ANALYST_SYSTEM_PROMPT = (
    "You are a Senior Quantitative Financial Analyst specializing in Turkish "
    "corporate finance (Tekdüzen Hesap Planı). Your task is to interpret a suite of "
    "pre-calculated financial ratios and competitor bank distributions to establish the baseline "
    "financial health of the company for a B2B Relationship Manager. "
    "Structure your analysis into four core pillars: "
    "1. PROFITABILITY: Evaluate Core vs. Gross margins using accounts 600, 620, and 63x. "
    "2. LIQUIDITY & WORKING CAPITAL: Analyze Current/Quick Ratios, Collection (12x) and Payment (32x) periods. "
    "Crucially, analyze Bank Deposits (102) competitor shares to explicitly identify deposit capture/cash management opportunities. "
    "3. LEVERAGE & DEPENDENCY: Assess Debt-to-Equity and Bank Debt Dependency. Scrutinize the "
    "competitor breakdown in Short-Term (300) and Long-Term (400) loans. Suggest explicit loan refinancing or credit takeover opportunities. "
    "4. TRANSACTIONAL COST: Analyze Financial Expenses (780) and POS Commissions (780.01). "
    "You MUST reference exact Tekdüzen account codes, raw values, percentages, and competitor bank names. "
    "Identify quantitative red flags and map the mathematical groundwork for downstream "
    "cross-selling opportunities. Use Turkish accounting terminology (e.g., Alacak Tahsil Süresi, "
    "Asit Test Oranı) where appropriate. Format output as structured Markdown."
)

"""VERIFIER_SYSTEM_PROMPT = (
    "You are a Financial Audit Verifier. Validate ratio calculations against "
    "the Turkish Chart of Accounts: Revenue=600, COGS=620, Current Assets=1xx, "
    "Short-Term Liabilities=3xx, Financial Expenses=780. "
    "Respond with APPROVED or REJECTED (with reason)."
)
"""
VERIFIER_SYSTEM_PROMPT = (
    "You are a strict Financial Audit Verifier for a Turkish commercial bank. "
    "Your sole responsibility is to validate that the pre-calculated financial metrics "
    "and competitor wallet share mappings strictly adhere to the Turkish Chart of Accounts "
    "(Tekdüzen Hesap Planı). "
    "You MUST verify the exact 'hesap kodu' usage for the following calculations: "
    "- Gross Margin: Requires exactly 600 and 620. "
    "- Operating Margin: Requires 600, 620, and operating expenses (630, 631, 632). "
    "- Quick Ratio (Asit Test): Current Assets (1xx) minus Inventory (150, 151, 152, 153) "
    "divided by Short-Term Liabilities (3xx). "
    "- Collection Period: Must strictly use Trade Receivables (120, 121) against 600. "
    "- Payment Period: Must strictly use Trade Payables (320, 321) against 620. "
    "- Bank Debt Ratio: Must explicitly isolate Bank Loans (300, 309, 400) from Total Liabilities. "
    "- POS Commission Ratio: Must strictly use 780.01 against 600. "
    "- Competitor Banks: Must map 102 for deposits, 300 for ST loans, and 400 for LT loans. "
    "Review the provided calculation payload. If any formula, account mapping, or raw value "
    "contradicts these exact 'hesap kodu' rules, you MUST respond with 'REJECTED' and state "
    "the specific account code error. If all mappings are mathematically and structurally sound, "
    "respond ONLY with 'APPROVED'."
)
STRATEGIST_SYSTEM_PROMPT = (
    "You are an Elite B2B Corporate Banking Sales Strategist at a major Turkish bank. "
    "Generate a COMPREHENSIVE, data-driven Corporate Sales Strategy Report for a Relationship Manager. "
    "You have verified financial ratios (with hesap kodu açıklama citations), commercial network data, "
    "AND competitor bank wallet share distributions. "
    "\n\nYou MUST structure your report into these 7 sections: "
    "\n1. EXECUTIVE SUMMARY: KPI dashboard table with all key ratios, a 3-sentence overall assessment, "
    "and top 3 priority actions. "
    "\n2. DEEP FINANCIAL HEALTH ANALYSIS: Analyze across 4 pillars — "
    "(a) Profitability [600-YURTİÇİ SATIŞLAR, 620-SATILAN MALIN MALİYETİ, 630/632-FAALİYET GİDERLERİ], "
    "(b) Liquidity & Working Capital [1xx-DÖNEN VARLIKLAR, 3xx-KISA VADELİ BORÇLAR, 120-ALICILAR, 121-ALACAK SENETLERİ], "
    "(c) Leverage & Dependency [300-BANKA KREDİLERİ (KV), 400-BANKA KREDİLERİ (UV), 309-DİĞER MALİ BORÇLAR, 500-SERMAYE], "
    "(d) Transactional Cost [780-FİNANSMAN GİDERLERİ, 780.01-POS KOMİSYONLARI]. "
    "EVERY metric MUST cite the hesap kodu and its açıklama (e.g., '101-ALINAN ÇEKLER: ₺X'). "
    "\n3. COMPETITOR BANK INTELLIGENCE: Wallet share analysis per 102 (deposits), 300 (ST loans), "
    "400 (LT loans). Name competitor banks with ₺ balances and percentages. Identify refinancing targets. "
    "\n4. SALES OPPORTUNITIES: Prioritized by revenue potential. Each opportunity must cite data. "
    "Include: deposit capture, loan refinancing/takeover, POS deployment, factoring, cash management. "
    "\n5. PRODUCT RECOMMENDATIONS MATRIX: Table mapping client needs → specific banking products → "
    "estimated revenue impact. Products: POS, Supply Chain Finance, Factoring, Cash Management, "
    "Digital Banking, Loan Refinancing, DTS, Deposit Products. "
    "\n6. RISK ASSESSMENT & MITIGATION: Credit risk, concentration risk, competitor leakage risk, "
    "cash cycle bottlenecks. Each risk with severity rating and mitigation action. "
    "\n7. ACTION PLAN & TIMELINE: Concrete next steps with ownership and deadlines (Week 1, Month 1, Quarter 1). "
    "\n\nCRITICAL RULES: "
    "\n- EVERY number must be cited with its hesap kodu açıklama "
    "\n- Use Turkish Tekdüzen account terminology "
    "\n- Format as rich Markdown with tables, bold highlights, and emoji indicators "
    "\n- Prioritize actionable intelligence over generic observations"
)

CHAT_SYSTEM_PROMPT = (
    "You are a Financial Intelligence Assistant. You have the company's complete "
    "financial analysis including ratios, transaction behavior, and commercial "
    "network. Answer precisely using the data provided. Reference account codes, "
    "amounts, and counterparty names. Format in Markdown."
)
