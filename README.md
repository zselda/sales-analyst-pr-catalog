# Financial Intelligence Platform V1
## Corporate Sales Strategy Generator 

An advanced AI-powered pipeline that transforms raw Mizan (Turkish uniform chart of accounts) spreadsheet data into actionable, ING-branded corporate sales strategies.

This platform uses a multi-agent **LangGraph** workflow powered by **Google Gemini** to extract data, map B2B commercial relationships, analyze financial health, and automatically generate comprehensive B2B sales strategy reports in **English and Turkish**.

---

## Architecture & Multi-Agent Pipeline

The core engine is built using [LangGraph](https://python.langchain.com/docs/langgraph) and operates via 5 specialized AI agents working sequentially:

1. **Extractor Agent**: Reads and standardizes raw Mizan data (Excel/CSV format), standardizing account codes and extracting hierarchical balances.
2. **Analyzer Agent**: Computes critical financial ratios (Gross Margin, Operating Margin, Current Ratio) and identifies top competitors in the banking relationships based on bank deposit accounts.
3. **Network Mapper Agent**: Maps commercial dependencies by parsing `120` (Alacaklı / Receivable Customers) and `320` (Borçlu / Payable Suppliers) accounts to build a relational layout of the company ecosystem.
4. **Strategist Agent**: Synthesizes the data into a structured 7-part Markdown sales strategy report (Company Profiling, Financial Health, Relationship Mapping, Wallet Share, Pitch Opportunities, Risk Assessment, and Action Plan).
5. **Translator Agent (Optional)**: If dual-language mode is enabled, perfectly translates the English B2B banking strategy report into native Turkish corporate banking terminology.

Finally, the localized Markdown reports are formatted into an **ING Bank Branded HTML Report** and exported cleanly to **PDF**.

---

## Outputs

When executed, the system generates the following artifacts in the `output/` directory:

1. **`{Company}_Report_EN.pdf`**: The synthesized English Corporate Strategy Report.
2. **`{Company}_Report_TR.pdf`**: The translated Turkish Corporate Strategy Report (if `--turkish` flag is passed).
3. **`{Company}_Report.html`**: A highly styled, ING-banded interactive HTML version of the report, containing inline styles, tables, and corporate typography.

---

## Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/zselda/financial-intelligence-platformV1.git
cd financial-intelligence-platformV1/backend

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy the sample environment file and add your Google Gemini API key:

```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

*(Note: The `llm_config.py` uses the Gemini 2.5 Pro experimental model by default)*

---

## Usage

You can run the full analysis pipeline directly from the command line on any standard Mizan format document.

### 1. Standard Execution (English Only)
```bash
python3 run_pipeline.py --file your_mizan_sample.xlsx --output-dir ./output
```

### 2. Dual-Language Execution (English + Turkish)
Append the `--turkish` flag to trigger the Translator Agent for a secondary localized PDF report.
```bash
python3 run_pipeline.py --file your_mizan_sample.xlsx --turkish --output-dir ./output
```

---

## Technology Stack

- **LLM**: Google Gemini (`gemini-2.5-pro-exp` via `google-genai` native SDK)
- **Orchestration**: LangGraph (State Graph Management)
- **Data Engineering**: Pandas, OpenPyXL (Excel parsing)
- **Document Generation**: Markdown-to-HTML, WeasyPrint / PDFKit for PDF generation
- **Styling**: Native CSS implementation for ING Identity branding
