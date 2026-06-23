"""
Sector Analysis — TCMB Sector Balance Sheet Benchmarks
=======================================================
Coding foundation for sector-level benchmarking based on the structure of
TCMB "Sektör Bilançoları" (Real Sector Company Accounts) statistics:
https://www.tcmb.gov.tr → İstatistikler → Reel Sektör İstatistikleri → Sektör Bilançoları

Responsibilities:
1. Load sector benchmark ratios from data/tcmb_sector_benchmarks.json
   (values are indicative placeholders until refreshed from TCMB releases).
2. Fuzzy-match the LLM-predicted sector string (English or Turkish, free
   text) to a benchmark sector.
3. Build a company-vs-sector comparison (dict + markdown) from the
   quant_analyst financial_ratios output, for injection into the
   strategist prompt.
4. Provide `fetch_tcmb_sector_data()` as the refresh entry point for
   wiring up the official TCMB Excel releases / EVDS API.
"""

import json
import logging
import os
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger("swarm.sector_analysis")

BENCHMARKS_PATH = Path(__file__).parent / "data" / "tcmb_sector_benchmarks.json"
NACE_MAPPING_PATH = Path(__file__).parent / "data" / "nace_sector_mapping.json"


def is_sector_analysis_enabled() -> bool:
    """
    Whether the TCMB sector-benchmark comparison should be injected into the
    strategist report. OPTIONAL by design: the benchmark JSON holds indicative
    placeholders that may be stale, and a comparison against out-of-date
    sector figures can mislead the RM. Controlled by ENABLE_SECTOR_ANALYSIS:

      - unset / "auto" (default): ENABLED only when the benchmark dataset is
        marked fresh (metadata.needs_refresh == false); DISABLED while the
        data still carries placeholder values.
      - "1"/"true"/"on": force ENABLED (use the data as-is, even if stale).
      - "0"/"false"/"off": force DISABLED (skip sector analysis entirely).

    Read per-call so deployments/tests switch via the environment.
    """
    flag = os.environ.get("ENABLE_SECTOR_ANALYSIS", "auto").strip().lower()
    if flag in ("1", "true", "on", "yes", "enabled"):
        return True
    if flag in ("0", "false", "off", "no", "disabled"):
        return False
    # "auto": follow the freshness of the benchmark dataset
    needs_refresh = load_benchmarks().get("metadata", {}).get("needs_refresh", True)
    return not needs_refresh

# Keyword → benchmark sector mapping. Sector strings come from an LLM
# (predict_sector) so matching must tolerate EN/TR free text.
# ORDER MATTERS: specific industries come BEFORE generic ones so that
# multi-word predictions like 'Textile Manufacturing + Export' match
# Textile, not Manufacturing.
_SECTOR_KEYWORDS = {
    "Textile": ["textile", "tekstil", "apparel", "garment", "konfeksiyon", "hazır giyim", "iplik"],
    "Food & Beverage": ["food", "gıda", "gida", "beverage", "içecek", "icecek", "dairy", "süt", "agro-food"],
    "Automotive": ["automotive", "otomotiv", "vehicle", "araç", "oto", "spare part", "yedek parça"],
    "Chemicals": ["chemical", "kimya", "plastic", "plastik", "pharma", "ilaç", "ilac", "boya", "paint"],
    "Energy": ["energy", "enerji", "power", "elektrik", "solar", "güneş", "petrol", "gas", "doğalgaz"],
    "Tourism": ["tourism", "turizm", "hotel", "otel", "hospitality", "konaklama", "restaurant", "restoran"],
    "Agriculture": ["agricultur", "tarım", "tarim", "farm", "çiftlik", "hayvancılık", "seed", "tohum"],
    "Technology": ["technology", "teknoloji", "software", "yazılım", "IT", "bilişim", "telecom", "iletişim", "e-commerce", "e-ticaret"],
    "Transportation & Logistics": ["logistic", "lojistik", "transport", "taşıma", "nakliye", "kargo", "shipping", "freight", "depolama"],
    "Construction": ["construction", "inşaat", "insaat", "contract", "müteahhit", "infrastructure", "yapı"],
    "Retail": ["retail", "perakende", "market", "store", "mağaza"],
    "Trading": ["trading", "trade", "ticaret", "wholesale", "toptan", "distribution", "dağıtım"],
    "Manufacturing": ["manufactur", "üretim", "imalat", "industrial", "sanayi", "machinery", "metal", "steel", "demir", "çelik"],
    "Services": ["service", "hizmet", "consult", "danışman"],
}

# Cross-sector modifiers (not TCMB balance-sheet sectors, but they shift
# product priorities — e.g., 'Textile Manufacturing + Export').
_MODIFIER_KEYWORDS = {
    "Export": ["export", "ihracat", "exporter"],
    "Import": ["import", "ithalat", "importer"],
}

# Maps benchmark metric → company-value sources + judgement direction:
#   (bench_key, quant_ratio_key, db_metric_en, unit, direction)
# - quant_ratio_key: key in quant_analyst's financial_ratios (Mizan-derived,
#   PRIMARY source — computed from the analyzed document itself)
# - db_metric_en:   en_description in local_db's bank-core metrics
#   (TR_EN_METRIC_MAP values) — FALLBACK source filling Mizan gaps
# - direction: "higher_better", "lower_better", "context" (no judgement)
#
# DB-name alignment with local_db.TR_EN_METRIC_MAP (TR originals):
#   Current Ratio ← 'Cari Oran'
#   Acid-Test Ratio (Quick Ratio) ← 'Asit Test Oranı'
#   Total Liabilities / Total Assets ← 'Toplam Yükümlülük / Toplam Aktif'
#   Gross Profit Margin (%) ← 'Brüt Kar marjı (%)'
#   Net Operating Profit (EBIT) Margin (%) ← 'Net Faaliyet (FVÖK) Karı (%)'
#   Net Profit Margin (%) ← 'Net kar marjı (%)'
#   Receivables Collection Period (Days) ← 'Alacak Devir Süresi (Gün)'
#   Payables Period (Days) ← 'Borç Devir Süresi (Gün)'
#   Inventory Period (Days) ← 'Stok Devir Süresi (Gün)'
#   Total Financial Debt / Total Liabilities ← 'Toplam Finansal Borç / Toplam Yükümlülükler'
# NOTE: debt_to_equity intentionally has NO DB binding — the DB only carries
# 'Toplam Net Borç / Özkaynak' (NET debt / equity), a different definition
# than the benchmark's total-liabilities-based debt-to-equity.
_RATIO_BINDINGS = [
    ("current_ratio", "current_ratio", "Current Ratio", "x", "higher_better"),
    ("acid_test_ratio", "quick_ratio", "Acid-Test Ratio (Quick Ratio)", "x", "higher_better"),
    ("debt_to_equity", "debt_to_equity", None, "x", "lower_better"),
    ("leverage_ratio_pct", None, "Total Liabilities / Total Assets", "%", "lower_better"),
    ("gross_margin_pct", "gross_margin", "Gross Profit Margin (%)", "%", "higher_better"),
    ("operating_margin_pct", "operating_margin", "Net Operating Profit (EBIT) Margin (%)", "%", "higher_better"),
    ("net_margin_pct", None, "Net Profit Margin (%)", "%", "higher_better"),
    ("collection_period_days", "collection_period", "Receivables Collection Period (Days)", "days", "lower_better"),
    ("payment_period_days", "payment_period", "Payables Period (Days)", "days", "context"),
    ("inventory_period_days", "inventory_period", "Inventory Period (Days)", "days", "lower_better"),
    ("bank_loans_to_liabilities_pct", "bank_debt_ratio", "Total Financial Debt / Total Liabilities", "%", "context"),
]


# EDW storage convention (CONFIRMED): DB metrics WITHOUT a '(%)' suffix
# that map to %-based TCMB benchmarks are stored as plain ratios
# (e.g. 'Toplam Yükümlülük / Toplam Aktif' = 0.72) → fixed ×100 conversion.
# '(%)'-suffixed metrics (Gross/EBIT/Net margins) are stored as percents
# already and need no scaling.
_DB_METRIC_SCALE = {
    "Total Liabilities / Total Assets": 100.0,
    "Total Financial Debt / Total Liabilities": 100.0,
}

_benchmarks_cache = None


def load_benchmarks(path: Path = BENCHMARKS_PATH) -> dict:
    """Load (and cache) the TCMB benchmark dataset."""
    global _benchmarks_cache
    if _benchmarks_cache is None:
        try:
            with open(path, encoding="utf-8") as f:
                _benchmarks_cache = json.load(f)
            if _benchmarks_cache.get("metadata", {}).get("needs_refresh"):
                logger.warning(
                    "TCMB sector benchmarks are INDICATIVE placeholders — refresh "
                    "from official TCMB Sektör Bilançoları releases "
                    "(see fetch_tcmb_sector_data)."
                )
        except Exception as e:
            logger.error(f"Could not load sector benchmarks: {e}")
            _benchmarks_cache = {"metadata": {}, "sectors": {}}
    return _benchmarks_cache


def _norm(text: str) -> str:
    """Lowercase + strip Turkish diacritics so 'İnşaat' matches 'insaat'."""
    s = (text or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


_nace_cache = None


def _load_nace_mapping(path: Path = NACE_MAPPING_PATH) -> dict:
    """Load (and cache) the NACE → sector mapping built from the İTO list."""
    global _nace_cache
    if _nace_cache is None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            code_to_sector = {}
            for sec, v in data.get("sectors", {}).items():
                for code in v.get("nace_codes", []):
                    code_to_sector[code] = sec
            _nace_cache = {
                "codes": code_to_sector,
                "divisions": data.get("division_fallback", {}),
            }
        except Exception as e:
            logger.error(f"Could not load NACE sector mapping: {e}")
            _nace_cache = {"codes": {}, "divisions": {}}
    return _nace_cache


def normalize_nace_code(nace_code: str) -> str:
    """Normalize NACE code formats: '471101' / '47 11 01' / '47.11.1' → '47.11.01'."""
    digits = re.sub(r"\D", "", str(nace_code or ""))
    if len(digits) >= 6:
        return f"{digits[0:2]}.{digits[2:4]}.{digits[4:6]}"
    if len(digits) == 5:  # e.g. '47111' → assume single-digit subclass
        return f"{digits[0:2]}.{digits[2:4]}.0{digits[4]}"
    if len(digits) == 4:
        return f"{digits[0:2]}.{digits[2:4]}"
    if len(digits) == 2:
        return digits
    return str(nace_code or "").strip()


def sector_from_nace(nace_code: str) -> str:
    """
    Map a customer's NACE code (from the local bank DB) to a TCMB benchmark
    sector. Exact 6-digit İTO match first, then 2-digit division fallback.
    Returns None when the code is absent or unknown.
    """
    if not nace_code:
        return None
    mapping = _load_nace_mapping()
    code = normalize_nace_code(nace_code)
    sector = mapping["codes"].get(code)
    if sector:
        return sector
    division = code.split(".")[0][:2]
    sector = mapping["divisions"].get(division)
    if sector:
        logger.info(
            f"NACE {code} not in İTO list — resolved via division {division} → {sector}"
        )
        return sector
    logger.warning(f"⚠️ NACE code '{nace_code}' could not be mapped to any sector")
    return None


def match_sectors(sector: str) -> dict:
    """
    Multi-sector aware matching of a free-text prediction (EN/TR) to TCMB
    benchmark sectors. Handles combined predictions like 'Retail + Export'
    or 'Textile Manufacturing + Export'.

    Returns:
      {
        "primary": str,            # benchmark key (fallback: 'General')
        "secondary": [str, ...],   # additional matched benchmark sectors
        "modifiers": [str, ...],   # 'Export' / 'Import' cross-sector flags
        "is_fallback": bool,       # True = prediction absent or unmatched
      }
    """
    result = {"primary": "General", "secondary": [], "modifiers": [], "is_fallback": True}
    s = _norm(sector)
    if not s:
        logger.warning(
            "⚠️ Sector prediction absent (empty) — TCMB benchmark falls back to 'General'"
        )
        return result

    sectors = load_benchmarks().get("sectors", {})
    tokens = [t.strip() for t in re.split(r"[+/,;]", s) if t.strip()]

    matched = []
    for tok in tokens:
        # Exact key match first, then keyword match (specific-first order)
        hit = next((k for k in sectors if _norm(k) == tok), None)
        if hit is None:
            hit = next(
                (k for k, kws in _SECTOR_KEYWORDS.items()
                 if k in sectors and any(_norm(kw) in tok for kw in kws)),
                None,
            )
        if hit and hit != "General" and hit not in matched:
            matched.append(hit)

    for modifier, kws in _MODIFIER_KEYWORDS.items():
        if any(_norm(kw) in s for kw in kws):
            result["modifiers"].append(modifier)

    if matched:
        result["primary"] = matched[0]
        result["secondary"] = matched[1:]
        result["is_fallback"] = False
    elif "general" in s:
        # 'General' is the explicit pipeline fallback, not a real prediction
        logger.warning(
            "⚠️ Sector prediction absent ('General') — TCMB benchmark uses all-sector aggregate"
        )
    else:
        logger.warning(
            f"⚠️ Sector '{sector}' not matched to any TCMB benchmark sector — "
            f"falling back to 'General' (report will lack sector-specific benchmark)"
        )
    return result


def match_sector(sector: str) -> str:
    """Map a free-text sector string (EN/TR) to a single benchmark sector key."""
    return match_sectors(sector)["primary"]


def get_sector_benchmark(sector: str, already_matched: bool = False) -> dict:
    """
    Return the benchmark dict for a sector, with metadata.
    Pass already_matched=True when `sector` is an exact benchmark key
    (skips re-matching and its fallback logging).
    """
    data = load_benchmarks()
    if already_matched and sector in data.get("sectors", {}):
        key = sector
    else:
        key = match_sector(sector)
    benchmark = dict(data.get("sectors", {}).get(key, {}))
    benchmark["matched_sector"] = key
    benchmark["source"] = data.get("metadata", {}).get("source", "TCMB Sektör Bilançoları")
    benchmark["reference_period"] = data.get("metadata", {}).get("reference_period", "")
    benchmark["needs_refresh"] = data.get("metadata", {}).get("needs_refresh", True)
    return benchmark


def compare_company_to_sector(financial_ratios: dict, sector: str,
                              db_metrics: dict = None) -> dict:
    """
    Compare company metrics against the matched TCMB sector benchmark.

    Company values are resolved per binding with this priority:
      1. quant_analyst's Mizan-derived ratio (source "mizan") — computed
         from the analyzed document, so it wins when both exist;
      2. the bank-core DB metric from local_db (source "bank_db",
         English en_description keys) — fills gaps the Mizan can't
         provide (net margin, leverage ratio, ...).

    Returns:
      {
        "enabled": bool,                   # False → sector analysis turned off
        "matched_sector": str,
        "secondary_sectors": [str, ...],   # multi-sector predictions
        "modifiers": [str, ...],           # Export / Import flags
        "is_fallback": bool,               # sector prediction absent/unmatched
        "rows": [{metric, company, sector, unit, deviation_pct,
                  assessment, source}],
        "banking_notes": str,
        "markdown": str   # ready-to-inject prompt section ("" when disabled)
      }

    When sector analysis is disabled (see is_sector_analysis_enabled), this
    returns an inert result with enabled=False and empty markdown so the
    strategist simply omits the benchmark section.
    """
    if not is_sector_analysis_enabled():
        logger.info(
            "[Benchmark] Sector analysis DISABLED (ENABLE_SECTOR_ANALYSIS / stale "
            "benchmark data) — skipping TCMB comparison in the report"
        )
        return {
            "enabled": False, "matched_sector": None, "secondary_sectors": [],
            "modifiers": [], "is_fallback": True, "rows": [],
            "banking_notes": "", "markdown": "",
        }

    match_info = match_sectors(sector)
    benchmark = get_sector_benchmark(match_info["primary"], already_matched=True)
    matched = benchmark.get("matched_sector", "General")
    db_metrics = db_metrics or {}
    rows = []

    for bench_key, ratio_key, db_metric_en, unit, direction in _RATIO_BINDINGS:
        bench_val = benchmark.get(bench_key)
        if bench_val is None:
            continue

        # 1. Mizan-derived ratio (primary), 2. bank DB metric (fallback)
        company_val = None
        source = None
        if ratio_key:
            company_val = (financial_ratios or {}).get(ratio_key, {}).get("value")
            if company_val is not None:
                source = "mizan"
        if company_val is None and db_metric_en and db_metric_en in db_metrics:
            company_val = db_metrics[db_metric_en]
            source = "bank_db"
        if company_val is None:
            continue

        try:
            company_val = float(company_val)
            bench_val = float(bench_val)
        except (TypeError, ValueError):
            continue
        if source == "bank_db":
            company_val *= _DB_METRIC_SCALE.get(db_metric_en, 1.0)

        deviation_pct = ((company_val - bench_val) / bench_val * 100) if bench_val else 0.0

        if direction == "context":
            assessment = "context"
        elif abs(deviation_pct) <= 15:
            assessment = "in line with sector"
        else:
            better = (deviation_pct > 0) == (direction == "higher_better")
            assessment = "better than sector" if better else "worse than sector"

        rows.append({
            # Label by the source actually used, so the table never shows a
            # Mizan ratio name for a value that came from the bank DB
            "metric": db_metric_en if source == "bank_db" else ratio_key,
            "company": round(company_val, 2),
            "sector": round(bench_val, 2),
            "unit": unit,
            "deviation_pct": round(deviation_pct, 1),
            "assessment": assessment,
            "source": source,
        })

    # ── Markdown rendering for LLM prompt injection ──
    if rows:
        table_lines = [
            "| Metric | Company | Sector Benchmark | Deviation | Assessment | Source |",
            "|--------|---------|------------------|-----------|------------|--------|",
        ]
        for r in rows:
            source_label = "Mizan" if r["source"] == "mizan" else "Bank DB (latest)"
            table_lines.append(
                f"| {r['metric']} | {r['company']}{r['unit']} | "
                f"{r['sector']}{r['unit']} | {r['deviation_pct']:+.1f}% | "
                f"{r['assessment']} | {source_label} |"
            )
        table_md = "\n".join(table_lines)
    else:
        table_md = "_No comparable benchmark metrics available._"

    refresh_note = (
        " (indicative values — refresh from official TCMB release)"
        if benchmark.get("needs_refresh") else ""
    )
    extra_lines = []
    if match_info["is_fallback"]:
        extra_lines.append(
            "⚠️ **Sector prediction absent or unmatched** — the all-sector "
            "General aggregate is used; treat deviations as indicative only."
        )
    if match_info["secondary"]:
        extra_lines.append(
            f"**Also matches sectors:** {', '.join(match_info['secondary'])} — the company "
            f"spans multiple sectors; blend product priorities accordingly."
        )
    if match_info["modifiers"]:
        extra_lines.append(
            f"**Cross-border profile:** {' & '.join(match_info['modifiers'])}-oriented — "
            f"weight trade finance, FX hedging, SWIFT and "
            f"{'export factoring' if 'Export' in match_info['modifiers'] else 'import LC/customs guarantee'} higher."
        )
    extra_md = ("\n" + "\n".join(extra_lines) + "\n") if extra_lines else ""

    markdown = (
        f"### 📐 TCMB SECTOR BENCHMARK COMPARISON — {matched} "
        f"({benchmark.get('tr_name', '')})\n"
        f"Source: {benchmark.get('source', 'TCMB Sektör Bilançoları')}, "
        f"{benchmark.get('reference_period', '')}{refresh_note}\n\n"
        f"{table_md}\n{extra_md}\n"
        f"**Sector banking profile:** {benchmark.get('banking_notes', 'n/a')}\n"
    )

    return {
        "enabled": True,
        "matched_sector": matched,
        "secondary_sectors": match_info["secondary"],
        "modifiers": match_info["modifiers"],
        "is_fallback": match_info["is_fallback"],
        "rows": rows,
        "banking_notes": benchmark.get("banking_notes", ""),
        "markdown": markdown,
    }


def fetch_tcmb_sector_data(output_path: Path = BENCHMARKS_PATH):
    """
    Refresh entry point for official TCMB sector balance sheet data.

    NOT IMPLEMENTED YET — wiring requires either:
    1. TCMB Sektör Bilançoları Excel releases (annual, ~3-digit NACE
       breakdown): download the XLSX from the TCMB statistics portal and
       map rows to the ratio keys in tcmb_sector_benchmarks.json, or
    2. EVDS API (https://evds2.tcmb.gov.tr) with an API key, querying the
       real-sector company-accounts series.

    Until implemented, edit data/tcmb_sector_benchmarks.json manually from
    the published TCMB tables and set metadata.needs_refresh = false.
    """
    raise NotImplementedError(
        "Wire this to the TCMB Sektör Bilançoları XLSX release or the EVDS API; "
        "see docstring for the two supported refresh paths."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    demo_ratios = {
        "current_ratio": {"value": 1.1},
        "quick_ratio": {"value": 0.7},
        "debt_to_equity": {"value": 2.4},
        "gross_margin": {"value": 16.0},
        "operating_margin": {"value": 6.2},
        "collection_period": {"value": 95},
        "payment_period": {"value": 60},
        "inventory_period": {"value": 80},
        "bank_debt_ratio": {"value": 41.0},
    }
    result = compare_company_to_sector(demo_ratios, "Tekstil / Hazır Giyim")
    print(result["markdown"])
