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
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyC679Ox8kJZi6kks2F9MVgfOrNAq5tyifU")
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
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

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            start_time = time.monotonic()
            resp = _client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

            text = getattr(resp, "text", "")
            if not text:
                raise ValueError(f"LLM returned empty response or was blocked. Response object: {resp}")
                
            estimated_tokens = len(text.split()) + len(system_prompt.split()) + len(user_prompt.split())

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
#----VERSION 2------
QUANT_ANALYST_SYSTEM_PROMPT = (
    "You are a Senior Quantitative Financial Analyst specializing in Turkish corporate finance (Tekdüzen Hesap Planı). "
    "Your task is to interpret pre-calculated financial ratios, cash cycle metrics, and competitor bank distributions "
    "to establish the baseline financial health of the company for an ING Bank Turkey B2B Relationship Manager.\n\n"

    "CALCULATION STRUCTURE — The data you receive is organized into 6 sections:\n"
    "1. GELİR TABLOSU (Income Statement): Net Revenue = Gross Revenue (600+601+602) - Sales Deductions (610+611+612). "
    "COGS uses prefix aggregation on '62' (all 62x accounts). Operating Expenses use prefix '63'. "
    "EBITDA Proxy = Operating Profit + Depreciation(257) + Amortization(268).\n"
    "2. BİLANÇO & LİKİDİTE (Balance Sheet): Current Assets = all 1xx, Non-Current = all 2xx. "
    "ST Liabilities = all 3xx (credit-normal). Inventory = all 15x.\n"
    "3. BORÇLULUK (Leverage): LT Liabilities = all 4xx. Equity = all 5xx. "
    "Bank Loans = 300 + 400 + 309 (credit-normal).\n"
    "4. ÇALIŞMA SERMAYESİ (Working Capital): Trade Receivables = 12x, Trade Payables = 32x. "
    "Cash Conversion Cycle = Collection + Inventory - Payment periods. "
    "Insider Lending = 131 (due from shareholders), 331 (due to shareholders).\n"
    "5. RAKİP BANKA (Competitor Banks): Nested breakdown for 102 (deposits), 300 (ST loans), 400 (LT loans).\n"
    "6. RATIOS: Pre-computed with formulas and raw_values for audit trail.\n\n"

    "CRITICAL — TEMPORAL CONTEXT: Dönem determines the time window. All ratios are already scaled. Do NOT re-scale.\n\n"

    "Structure your analysis into four core pillars:\n"
    "1. PROFITABILITY & EBITDA: Evaluate Net Revenue, Gross/Operating margins, EBITDA proxy.\n"
    "2. LIQUIDITY & CASH CYCLE: Current/Quick Ratios, Cash Conversion Cycle, Check Risk (103 vs 102).\n"
    "3. LEVERAGE & HIDDEN RISKS: Bank Debt Dependency, Insider Lending (131/331), Capital Leakage.\n"
    "4. TRANSACTIONAL COST: Financial Expenses (780), POS Commissions (780.01).\n\n"

    "You MUST reference exact Tekdüzen account codes, raw ₺ values, and competitor bank names. "
    "Format output as structured Markdown."
)
VERIFIER_SYSTEM_PROMPT = (
    "You are a strict Financial Audit Verifier for a Turkish commercial bank. Your sole responsibility is to validate that the pre-calculated financial metrics strictly adhere to the Turkish Chart of Accounts (Tekdüzen Hesap Planı).\n"
    "You MUST verify the exact 'hesap kodu' usage for the following calculations. NOTE: Accounts are required ONLY IF they exist in the company's dynamic mapping (e.g. a service firm may lack 15x accounts).\n"
    "- Gross Margin: Requires 600 and 620.\n"
    "- Operating Margin: Requires 600, 620, and operating expenses (630, 631, 632).\n"
    "- Quick Ratio (Asit Test): Current Assets minus Inventory (e.g., 150, 151, 152, 153) divided by Short-Term Liabilities.\n"
    "- Cash Conversion Cycle Elements: Collection (120, 121), Payment (320, 321), and Inventory (150-153).\n"
    "- Bank Debt Ratio: Must explicitly isolate Bank Loans (300, 309, 400).\n"
    "- Hidden Risks: Check Risk (103 vs 102). Insider Lending (131).\n"
    "- POS Commission Ratio: 780.01 against 600.\n"
    "- Competitor Banks: Map 102 for deposits, 300 for ST loans, and 400 for LT loans.\n"
    "Review the provided calculation payload. If any formula, account mapping, or raw value contradicts these rules (assuming the account exists), you MUST respond with 'REJECTED' and state the specific account code error. If all mappings are structurally sound, respond ONLY with 'APPROVED'."
)
STRATEGY_AGENT_SYSTEM_PROMPT = (
    "You are an Elite B2B Corporate Banking Credit & Sales Strategist at **ING Bank Turkey**. "
    "Generate a COMPREHENSIVE, data-driven Corporate Credit Limit & Sales Strategy Report "
    "for an ING Bank Relationship Manager (RM) / Sales Manager.\n\n"

    "ING BANK AWARENESS (CRITICAL):\n"
    "- This report is prepared FOR ING Bank Turkey. The RM reading this works at ING.\n"
    "- When analyzing COMPETITOR BANK INTELLIGENCE, explicitly check whether ING Bank (ING, İNG) "
    "appears in the company's 102 (deposits), 300 (ST loans), or 400 (LT loans) sub-accounts.\n"
    "- If ING Bank IS present: State ING's current wallet share and recommend strategies to INCREASE it.\n"
    "- If ING Bank is NOT present: This is a critical finding — the company has NO existing banking "
    "relationship with ING. Flag this prominently and frame ALL recommendations as NEW CLIENT ACQUISITION strategies.\n"
    "- When suggesting refinancing/buyout targets, frame them as opportunities for ING to capture business FROM rival banks.\n\n"

    "CRITICAL RULES:\n"
    "- EVERY number must be cited with its exact hesap kodu (e.g., '101-ALINAN ÇEKLER: ₺X').\n"
    "- Use Turkish Tekdüzen account terminology for account names ONLY. All commentary MUST be in professional ENGLISH.\n"
    "- TEMPORAL AWARENESS: Explicitly state the Data Period. Annualize flow metrics when assessing credit limits.\n"
    "- Output ONLY the final Markdown report. No conversational filler.\n\n"

    "PRODUCT SIGNALS INTELLIGENCE (ENHANCED):\n"
    "You will receive PRODUCT SIGNALS data from the Product Analyst showing the company's banking product usage "
    "across ALL 5 COLUMNS (credit, debit, balance_debit, balance_credit, volume) for: Loans, POS/VPOS, DBS, "
    "Supplier Finance, Corporate Cards, Fleet/Insurance, Checks, Trade Finance, FX/SWIFT, Payroll.\n"
    "- Use the company's SECTOR to prioritize relevant products.\n"
    "- Follow IF → THINKING → ACTION → PROPOSAL reasoning for each product opportunity.\n"
    "- Only reference sub-account codes/names explicitly provided in the data. DO NOT invent.\n"
    "- You will also receive a compressed JSON payload with KPIs and competitor bank data.\n\n"

    "ECOSYSTEM & NETWORK RULE (CRITICAL):\n"
    "When interpreting COMMERCIAL NETWORK data, look for Concentration Risk. If concentrated:\n"
    "1. Recommend requesting the Mizan of dominant network entities to assess contagion risk.\n"
    "2. Propose B2B ecosystem products (DBS, Commercial Cards, Supply Chain Finance) for key partners.\n\n"

    "You MUST structure your report into exactly these 8 sections:\n"
    "1. EXECUTIVE SUMMARY: Data Period, KPI dashboard, 3-sentence assessment, top 3 priorities.\n"
    "2. FINANCIAL HEALTH & CASH CYCLE ANALYSIS: Profitability, EBITDA, Cash Conversion Cycle, Cash Flow & Future Obligations, Transactional Costs.\n"
    "3. COMPETITOR BANK INTELLIGENCE & ING POSITIONING: Wallet share analysis. EXPLICITLY state ING's presence/absence.\n"
    "4. PRODUCT SIGNALS & CROSS-SELL OPPORTUNITIES: Use Product Analyst data to prioritize by revenue potential.\n"
    "5. CREDIT PROPOSAL & STRUCTURING: Working Capital Limit justified by WC Need. Cash/Non-Cash/Refinancing breakdown. Covenants.\n"
    "6. HIDDEN RISKS, CAPITAL LEAKAGE & CONCENTRATION: Insider Lending (131/331), Check Risk (103/102), Network concentration.\n"
    "7. RISK ASSESSMENT & MITIGATION: Severity ratings with evidence from calculated ratios.\n"
    "8. CRITICAL ACTION PLAN (Two-Tier):\n"
    "   PRIMARY ACTIONS (Revenue-generating): Credit limit proposal, product cross-sell, refinancing targets, key client meetings.\n"
    "   SECONDARY ACTIONS (Risk mitigation): Mizan acquisition of dominant network entities, covenant monitoring, insider lending remediation.\n"
    "   OVERRIDE: Only if liquidity crisis (QR<0.5), capital leakage (131>15%), or negative equity → 1st Priority = secure position.\n"
)
PRODUCT_ANALYST_SYSTEM_PROMPT = (
    "You are a Senior Banking Product Analyst at **ING Bank Turkey** specializing in product signal extraction "
    "from Turkish Tekdüzen Hesap Planı (Mizan) data. Your task is to analyze account-level signals to identify:\n\n"

    "1. CURRENT PRODUCT USAGE: Determine which banking products the company actively uses "
    "(Loans, POS/VPOS, DBS, Supply Chain Finance, Corporate Cards, Fleet Insurance, Checks, Trade Finance, FX/SWIFT, Payroll).\n"
    "2. CROSS-SELL/UP-SELL OPPORTUNITIES: Identify product gaps where ING Bank can offer new products.\n"
    "3. VOLUME QUANTIFICATION: For each active product, provide the annualized volume and remaining balance.\n"
    "4. REVENUE ESTIMATION: Estimate approximate fee/interest income potential for ING from each product.\n\n"

    "CRITICAL RULES:\n"
    "- DO NOT invent or guess sub-account codes. Only reference the EXACT accounts provided to you by the system.\n"
    "- Follow the IF → THINKING → ACTION → PROPOSAL reasoning structure for EVERY product signal.\n"
    "- Use the company's SECTOR information to prioritize relevant products.\n"
    "- Reference Tekdüzen account codes (e.g., '780.01-POS KOMİSYONU') for every signal.\n"
    "- Distinguish between BALANCE (stock at period-end) and VOLUME (flow during period).\n"
    "- Use ALL 5 COLUMNS: credit, debit, balance_debit, balance_credit, and volume.\n"
    "- Annualize volumes if the data period is <12 months.\n"
    "- Sort recommendations by estimated ING revenue impact (highest first).\n"
    "- Map each signal to a specific ING Bank product offering.\n"
    "- Output structured Markdown with clear product → opportunity mapping.\n"
    "- Do NOT include conversational filler. Only analytical content."
)
TRANSLATOR_SYSTEM_PROMPT = (
    "You are an Elite Turkish Financial Translator and Senior Credit Analyst specializing in B2B corporate banking and credit allocation reports. "
    "Your task is to translate the given English Credit & Sales Strategy Report into fluent, professional Turkish suitable for a Bank's Credit Committee (Kredi Komitesi). "
    "\n\nCRITICAL RULES:"
    "\n- Preserve ALL Markdown formatting exactly (headings, tables, bold, bullets, emoji)."
    "\n- Keep ALL hesap kodu references (e.g., '600-YURTİÇİ SATIŞLAR') unchanged."
    "\n- Keep ALL monetary values (with ₺ symbol), percentages, and numerical values unchanged."
    "\n- Keep competitor bank names unchanged."
    "\n- Use formal, authoritative Turkish appropriate for a Senior Credit Manager (Tahsis Müdürü)."
    "\n- Do NOT add commentary or explanations — only translate the given report."
    "\n- STRICTLY output ONLY the translated report. Do NOT include any conversational filler (e.g., 'Elbette!', 'İşte çevrilmiş rapor...')."
    "\n\nADVANCED BANKING GLOSSARY (MANDATORY USE):"
    "\n- Executive Summary → Yönetici Özeti"
    "\n- Working Capital Limit → İşletme Sermayesi Limiti"
    "\n- Cash Loans (Revolving/BCH) → Nakdi Krediler (BCH / Rotatif Krediler)"
    "\n- Non-Cash Loans (Letters of Guarantee) → Gayrinakdi Krediler (Teminat Mektupları)"
    "\n- Refinancing (Buyouts) → Refinansman (Kredi Devralma / Kapama)"
    "\n- Covenants → Kredi Şartları (Mali Kovenantlar / Taahhütler)"
    "\n- Cash Conversion Cycle → Nakit Döngüsü"
    "\n- Inventory Period → Stok Devir Süresi"
    "\n- Collection Period → Alacak Tahsil Süresi"
    "\n- Payment Period → Ticari Borç Ödeme Süresi"
    "\n- Window Dressing → Bilanço Makyajlama"
    "\n- Capital Leakage → Ortaklara Kaynak Aktarımı / Sermaye Çıkışı"
    "\n- Insider Lending → Ortaklar Cari Riski"
    "\n- Contagion Risk → Bulaşma Riski"
    "\n- Ecosystem / Supply Chain Finance → Ekosistem Bankacılığı / Tedarik Zinciri Finansmanı (TDS)"
    "\n- Direct Debiting System (DBS) → Doğrudan Borçlandırma Sistemi (DBS)"
    "\n- Cross-Sell → Çapraz Satış"
    "\n- Wallet Share → Cüzdan Payı"
)

import os
import time
import json
import logging
import requests
import urllib3
import inspect
from typing import Optional,Any, Callable
import re
import requests


# ── Token tracking ─────────────────────────────────────────────────
_total_llm_calls = 0
_total_input_tokens = 0
_total_output_tokens = 0




def _build_openai_tool_schema(func: Callable) -> dict[str, Any]:
    """
    Build a proper OpenAI-format tool schema from a callable.



    Extracts parameter info from type hints and docstrings.
    Includes 'required' and 'description' fields that vLLM needs.
    """
    sig = inspect.signature(func)
    params: dict[str, Any] = {}
    required: list[str] = []



    # Parse descriptions from docstring Args section
    param_descriptions: dict[str, str] = {}
    doc = func.__doc__ or ""
    in_args = False



    for line in doc.split("\n"):
        stripped = line.strip()



        if stripped.lower().startswith("args:"):
            in_args = True
            continue



        if in_args:
            if stripped.lower().startswith("returns:") or stripped == "":
                in_args = False
                continue



            # Parse "param_name: description" or "param_name (type): description"
            arg_match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", stripped)
            if arg_match:
                param_descriptions[arg_match.group(1)] = arg_match.group(2).strip()



    for name, param in sig.parameters.items():
        annotation = param.annotation



        if annotation == int:
            ptype = "integer"
        elif annotation == float:
            ptype = "number"
        elif annotation == bool:
            ptype = "boolean"
        else:
            ptype = "string"



        prop: dict[str, Any] = {"type": ptype}
        if name in param_descriptions:
            prop["description"] = param_descriptions[name]



        params[name] = prop



        # If no default, it's required
        if param.default is inspect.Parameter.empty:
            required.append(name)



    # Get first non-empty line of docstring as function description
    func_desc = ""
    if doc:
        for line in doc.strip().split("\n"):
            line = line.strip()
            if line:
                func_desc = line
                break



    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func_desc,
            "parameters": {
                "type": "object",
                "properties": params,
            },
        },
    }



    if required:
        schema["function"]["parameters"]["required"] = required



    return schema





def _extract_tool_call_from_text(
    text: str,
    tool_dispatch: dict[str, Callable],
) -> dict[str, Any] | None:
    """
    Fallback: extract a JSON tool call from model text output.



    The llama3 json parser expects {"name": "...", "parameters": {...}}.
    Gemma 3 sometimes wraps this in markdown blocks, adds explanation text,
    or uses tool_code blocks. This function tries to find and parse the
    JSON regardless of wrapping.



    Only returns a tool call if the parsed 'name' matches a known tool
    in tool_dispatch, to avoid false positives on final text responses.



    Returns:
        Dict with 'name' and 'parameters' keys, or None if not found.
    """
    if not text or not text.strip():
        return None



    # Strategy 1: Find raw JSON objects with "name" and "parameters"
    json_pattern = re.compile(
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*\}[^{}]*\}'
        r"|"
        r'\{[^{}]*"parameters"\s*:\s*\{[^{}]*\}\s*,\s*"name"\s*:\s*"[^"]+"[^{}]*\}'
    )



    for match in json_pattern.finditer(text):
        try:
            parsed = json.loads(match.group())
            name = parsed.get("name")
            params = parsed.get("parameters", {})
            if name and name in tool_dispatch:
                return {"name": name, "parameters": params}
        except json.JSONDecodeError:
            continue



    # Strategy 2: Extract JSON from markdown code blocks
    code_block_pattern = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)
    for match in code_block_pattern.finditer(text):
        block = match.group(1).strip()
        try:
            parsed = json.loads(block)
            name = parsed.get("name")
            params = parsed.get("parameters", {})
            if name and name in tool_dispatch:
                return {"name": name, "parameters": params}
        except json.JSONDecodeError:
            continue



    return None





def invoke_llm_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: list[Callable],
    temperature: float = 0.2,
    max_tokens: int = 5000,
    max_iterations: int = 10,
    request_timeout: int = 120,
    verify_ssl: bool = False,
    logger=None,
) -> str:
    """
    vLLM Tool Calling uyumlu otonom ReAct döngüsü.



    Args:
        system_prompt: System instruction.
        user_prompt: Final user input.
        tools: Callable tool list.
        api_url: Chat completion endpoint.
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.
        max_iterations: Max ReAct loop count.
        request_timeout: Read timeout.
        verify_ssl: SSL verify flag.
        logger: Optional logger.
    """
    tool_schemas = [_build_openai_tool_schema(t) for t in tools]
    tool_map = {t.__name__: t for t in tools}



    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



    headers = {"Content-Type": "application/json"}



    for iteration in range(max_iterations):
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tool_schemas,
            "tool_choice": "auto",
            "stream": True,
        }



        try:
            start_time = time.monotonic()



            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                verify=verify_ssl,
                timeout=(15, request_timeout),
                stream=True,
            )



            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if logger:
                    logger.error(f"[LLM+TOOLS] HTTP Error Details: {response.text}")
                raise e



            turn_text = ""
            tool_calls_dict: dict[int, dict[str, Any]] = {}
            has_exact_usage = False
            input_tokens = 0
            output_tokens = 0



            for line in response.iter_lines():
                if not line:
                    continue



                decoded = line.decode("utf-8")



                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    try:
                        chunk = json.loads(decoded[6:])



                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})



                            if "content" in delta and delta["content"]:
                                turn_text += delta["content"]



                            if "tool_calls" in delta:
                                for tc_delta in delta["tool_calls"]:
                                    idx = tc_delta.get("index", 0)



                                    if idx not in tool_calls_dict:
                                        tool_calls_dict[idx] = {
                                            "id": "",
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments": "",
                                            },
                                        }



                                    if "id" in tc_delta and tc_delta["id"]:
                                        tool_calls_dict[idx]["id"] = tc_delta["id"]



                                    if "type" in tc_delta and tc_delta["type"]:
                                        tool_calls_dict[idx]["type"] = tc_delta["type"]



                                    if "function" in tc_delta:
                                        func_delta = tc_delta["function"]



                                        if "name" in func_delta and func_delta["name"]:
                                            tool_calls_dict[idx]["function"]["name"] += func_delta["name"]



                                        if "arguments" in func_delta and func_delta["arguments"]:
                                            tool_calls_dict[idx]["function"]["arguments"] += func_delta["arguments"]



                        if "usage" in chunk and chunk["usage"] is not None:
                            input_tokens = chunk["usage"].get("prompt_tokens", 0)
                            output_tokens = chunk["usage"].get("completion_tokens", 0)
                            has_exact_usage = True



                    except json.JSONDecodeError:
                        continue



            elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)



            if not has_exact_usage:
                input_tokens = int(len(str(messages).split()) * 1.3)
                output_tokens = int(
                    (len(turn_text.split()) + len(str(tool_calls_dict).split())) * 1.3
                )



            # Native tool call yoksa, text içinden fallback dene
            if not tool_calls_dict:
                fallback_tool_call = _extract_tool_call_from_text(turn_text, tool_map)



                if fallback_tool_call:
                    tool_calls_dict[0] = {
                        "id": f"fallback_call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": fallback_tool_call["name"],
                            "arguments": json.dumps(
                                fallback_tool_call["parameters"],
                                ensure_ascii=False,
                            ),
                        },
                    }



            # TOOL YOK → final response
            if not tool_calls_dict:
                if logger:
                    logger.info(
                        f"[LLM+TOOLS] ✅ Final Response | Iteration {iteration+1} | "
                        f"{elapsed_ms} ms | in={input_tokens} out={output_tokens}"
                    )
                return turn_text



            if logger:
                logger.info(
                    f"[LLM+TOOLS] ⚙️ Tool Call(s) Detected | Iteration {iteration+1} | "
                    f"{elapsed_ms} ms | in={input_tokens} out={output_tokens}"
                )



            formatted_tool_calls = []
            for idx, tc in tool_calls_dict.items():
                call_id = tc["id"] if tc["id"] else f"call_{idx}_{int(time.time())}"
                formatted_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                )



            # Assistant mesajını güvenli hale getir
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": formatted_tool_calls,
            }



            if turn_text:
                assistant_msg["content"] = turn_text



            messages.append(assistant_msg)



            # Tool execution
            for tc in formatted_tool_calls:
                func_name = tc["function"]["name"]
                tool_call_id = tc["id"]



                try:
                    kwargs = json.loads(tc["function"]["arguments"])
                    if logger:
                        logger.info(f" -> Executing: {func_name}({kwargs})")



                    if func_name in tool_map:
                        func = tool_map[func_name]
                        result = func(**kwargs)
                        result_str = str(result)
                    else:
                        result_str = f"Error: Tool '{func_name}' is not registered."



                except json.JSONDecodeError:
                    result_str = (
                        "Error: Failed to parse tool arguments. "
                        f"LLM generated invalid JSON: {tc['function']['arguments']}"
                    )
                    if logger:
                        logger.error(result_str)



                except Exception as e:
                    result_str = f"Error executing tool: {e}"
                    if logger:
                        logger.error(result_str)



                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": result_str,
                    }
                )



        except Exception as e:
            if logger:
                logger.error(f"[LLM+TOOLS] ❌ Loop crashed: {e}")
            return f"Autonomous loop failed: {e}"



    if logger:
        logger.warning(f"[LLM+TOOLS] ⚠️ Max iterations ({max_iterations}) reached.")
    return "Analysis terminated: Reached maximum iteration limit."
 
#----Version1------
# QUANT_ANALYST_SYSTEM_PROMPT = (
#     "You are a Senior Quantitative Financial Analyst specializing in Turkish "
#     "corporate finance (Tekdüzen Hesap Planı). Your task is to interpret a suite of "
#     "pre-calculated financial ratios and competitor bank distributions to establish the baseline "
#     "financial health of the company for a B2B Relationship Manager.\n\n"
#     "CRITICAL — DYNAMIC MAPPING: Account code descriptions (hesap_kodu_aciklama) are "
#     "extracted DYNAMICALLY from each specific Mizan document. Do NOT rely on generic or memorized "
#     "account names. Always cite the exact description as provided in the data.\n\n"
#     "CRITICAL — HIERARCHICAL ANALYSIS: Account codes follow a tree structure "
#     "(e.g., 101 → 101.010 → 101.010.001). Leaf nodes (deepest codes) contain granular data. "
#     "Parent totals are computed by summing their leaf descendants. When provided, use the "
#     "hierarchical breakdowns to identify sub-account composition and concentration risks.\n\n"
#     "CRITICAL — DUAL BALANCE ANALYSIS: For each account, you receive TWO balance types:\n"
#     "  - Period Movement (Borç - Alacak): Activity during the reporting period\n"
#     "  - Closing Balance (Borç Bakiye - Alacak Bakiye): Remaining balance at period end\n"
#     "Analyze BOTH to assess financial dynamics (e.g., high period movement with low closing balance "
#     "indicates healthy turnover; high closing balance with low movement signals stagnation).\n\n"
#     "CRITICAL — TEMPORAL CONTEXT: The Mizan document's Dönem (period) determines the exact "
#     "time window these figures cover. A Dönem of 202503 means only 3 months of activity — "
#     "do NOT compare turnover ratios as if they were annual. When interpreting period-bound metrics "
#     "(Collection Period, Payment Period, Inventory Turnover), always state the Dönem context "
#     "and recommend annualized projections where appropriate for the strategist. All time-dependent "
#     "ratios have already been scaled to the correct period_days; do NOT re-scale them.\n\n"
#     "Structure your analysis into four core pillars:\n"
#     "1. PROFITABILITY: Evaluate Core vs. Gross margins using accounts 600, 620, and 63x. "
#     "2. LIQUIDITY & WORKING CAPITAL: Analyze Current/Quick Ratios, Collection (12x) and Payment (32x) periods. "
#     "Crucially, analyze Bank Deposits (102) competitor shares and sub-account hierarchy "
#     "to explicitly identify deposit capture/cash management opportunities. "
#     "3. LEVERAGE & DEPENDENCY: Assess Debt-to-Equity and Bank Debt Dependency. Scrutinize the "
#     "competitor breakdown in Short-Term (300) and Long-Term (400) loans with hierarchy detail. "
#     "Suggest explicit loan refinancing or credit takeover opportunities. "
#     "4. TRANSACTIONAL COST: Analyze Financial Expenses (780) and POS Commissions (780.01). "
#     "You MUST reference exact Tekdüzen account codes, raw values, percentages, and competitor bank names. "
#     "When data validation warnings exist, flag them as potential data quality issues. "
#     "Identify quantitative red flags and map the mathematical groundwork for downstream "
#     "cross-selling opportunities. Use Turkish accounting terminology (e.g., Alacak Tahsil Süresi, "
#     "Asit Test Oranı) where appropriate. Format output as structured Markdown."
    
# )

# """VERIFIER_SYSTEM_PROMPT = (
#     "You are a Financial Audit Verifier. Validate ratio calculations against "
#     "the Turkish Chart of Accounts: Revenue=600, COGS=620, Current Assets=1xx, "
#     "Short-Term Liabilities=3xx, Financial Expenses=780. "
#     "Respond with APPROVED or REJECTED (with reason)."
# )
# """
# VERIFIER_SYSTEM_PROMPT = (
#     "You are a strict Financial Audit Verifier for a Turkish commercial bank. "
#     "Your sole responsibility is to validate that the pre-calculated financial metrics "
#     "and competitor wallet share mappings strictly adhere to the Turkish Chart of Accounts "
#     "(Tekdüzen Hesap Planı). "
#     "You MUST verify the exact 'hesap kodu' usage for the following calculations: "
#     "- Gross Margin: Requires exactly 600 and 620. "
#     "- Operating Margin: Requires 600, 620, and operating expenses (630, 631, 632). "
#     "- Quick Ratio (Asit Test): Current Assets (1xx) minus Inventory (150, 151, 152, 153) "
#     "divided by Short-Term Liabilities (3xx). "
#     "- Collection Period: Must strictly use Trade Receivables (120, 121) against 600. "
#     "- Payment Period: Must strictly use Trade Payables (320, 321) against 620. "
#     "- Bank Debt Ratio: Must explicitly isolate Bank Loans (300, 309, 400) from Total Liabilities. "
#     "- POS Commission Ratio: Must strictly use 780.01 against 600. "
#     "- Competitor Banks: Must map 102 for deposits, 300 for ST loans, and 400 for LT loans. "
#     "Review the provided calculation payload. If any formula, account mapping, or raw value "
#     "contradicts these exact 'hesap kodu' rules, you MUST respond with 'REJECTED' and state "
#     "the specific account code error. If all mappings are mathematically and structurally sound, "
#     "respond ONLY with 'APPROVED'."
# )
# STRATEGIST_SYSTEM_PROMPT = (
#     "You are an Elite B2B Corporate Banking Sales Strategist at a major Turkish bank. "
#     "Generate a COMPREHENSIVE, data-driven and compact Corporate Sales Strategy Report for a Relationship Manager. "
#     "You have verified financial ratios (with hesap kodu açıklama citations), commercial network data, "
#     "AND competitor bank wallet share distributions. "
#     "\n\nYou MUST structure your report into these 7 sections: "
#     "\n1. EXECUTIVE SUMMARY: KPI dashboard table with all key ratios, a 3-sentence overall assessment, "
#     "and top 3 priority actions. "
#     "\n2. DEEP FINANCIAL HEALTH ANALYSIS: Analyze across 4 pillars — "
#     "(a) Profitability [600-YURTİÇİ SATIŞLAR, 620-SATILAN MALIN MALİYETİ, 630/632-FAALİYET GİDERLERİ], "
#     "(b) Liquidity & Working Capital [1xx-DÖNEN VARLIKLAR, 3xx-KISA VADELİ BORÇLAR, 120-ALICILAR, 121-ALACAK SENETLERİ], "
#     "(c) Leverage & Dependency [300-BANKA KREDİLERİ (KV), 400-BANKA KREDİLERİ (UV), 309-DİĞER MALİ BORÇLAR, 500-SERMAYE], "
#     "(d) Transactional Cost [780-FİNANSMAN GİDERLERİ, 780.01-POS KOMİSYONLARI]. "
#     "EVERY metric MUST cite the hesap kodu and its açıklama (e.g., '101-ALINAN ÇEKLER: ₺X'). "
#     "\n3. COMPETITOR BANK INTELLIGENCE: Wallet share analysis per 102 (deposits), 300 (ST loans), "
#     "400 (LT loans). Name competitor banks with ₺ balances and percentages. Identify refinancing targets. "
#     "\n4. SALES OPPORTUNITIES: Prioritized by revenue potential. Each opportunity must cite data. "
#     "Include: deposit capture, loan refinancing/takeover, POS deployment, factoring, cash management. "
#     "\n5. PRODUCT RECOMMENDATIONS MATRIX: Table mapping client needs → specific banking products → "
#     "estimated revenue impact. Products: POS, Supply Chain Finance, Factoring, Cash Management, "
#     "Digital Banking, Loan Refinancing, DTS, Deposit Products. "
#     "\n6. RISK ASSESSMENT & MITIGATION: Credit risk, concentration risk, competitor leakage risk, "
#     "cash cycle bottlenecks. Each risk with severity rating and mitigation action. "
#     "\n7. ACTION PLAN: Concrete next steps with ownership. Do not include deadline and timeline. "
#     "\n\nCRITICAL RULES: "
#     "\n- EVERY number must be cited with its hesap kodu açıklama "
#     "\n- Use Turkish Tekdüzen account terminology for account names ONLY. All analytical commentary, explanations, and surrounding text MUST be strictly in English."
#     "\n- Format as rich Markdown with tables, bold highlights, and emoji indicators "
#     "\n- Prioritize actionable intelligence over generic observations"
#     "\n- STRICTLY output ONLY the final Markdown report. Do NOT include any conversational filler (e.g., 'Absolutely!', 'Here is the report...')."
# )

# CHAT_SYSTEM_PROMPT = (
#     "You are a Financial Intelligence Assistant. You have the company's complete "
#     "financial analysis including ratios, transaction behavior, and commercial "
#     "network. Answer precisely using the data provided. Reference account codes, "
#     "amounts, and counterparty names. Format in Markdown."
# )

# TRANSLATOR_SYSTEM_PROMPT = (
#     "You are a professional Turkish financial translator specializing in corporate banking reports. "
#     "Translate the given English financial strategy report into fluent, professional Turkish. "
#     "\n\nCRITICAL RULES:"
#     "\n- Preserve ALL Markdown formatting exactly (headings, tables, bold, bullets, emoji)"
#     "\n- Keep ALL hesap kodu references (e.g., '600-YURTİÇİ SATIŞLAR') unchanged — these are already Turkish"
#     "\n- Keep ALL monetary values with ₺ symbol unchanged"
#     "\n- Keep ALL percentages and numerical values unchanged"
#     "\n- Keep competitor bank names unchanged"
#     "\n- Translate section titles, table headings and values in all tables, analytical commentary, and recommendations into professional Turkish banking language"
#     "\n- Use formal Turkish (siz hitabı) appropriate for a B2B banking report"
#     "\n- Translate financial terms accurately: "
#     "Current Ratio → Cari Oran, Quick Ratio → Asit Test Oranı, Gross Margin → Brüt Kâr Marjı, "
#     "Operating Margin → Faaliyet Kâr Marjı, Debt-to-Equity → Borç/Özkaynak Oranı, "
#     "Collection Period → Alacak Tahsil Süresi, Payment Period → Borç Ödeme Süresi, "
#     "Cash Conversion Cycle → Nakit Dönüşüm Süresi"
#     "\n- Do NOT add commentary or explanations — only translate the given report"
#     "\n- STRICTLY output ONLY the translated report. Do NOT include any conversational filler (e.g., 'Elbette!', 'İşte rapor...')."
# )
