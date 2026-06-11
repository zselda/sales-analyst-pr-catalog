"""
Agent: Translator — Turkish Financial Report Translation
 
Translates the English strategy report into professional Turkish banking language.
Preserves all formatting, account codes, monetary values, and structural elements.
 
This agent is CONDITIONALLY executed based on the GENERATE_TURKISH flag.
"""
 
import logging
from agents.base import BaseAgent
from llm_config import invoke_llm, TRANSLATOR_SYSTEM_PROMPT
 
logger = logging.getLogger("swarm.agents.translator")
 
 
class TranslatorAgent(BaseAgent):
    name = "translator"
    description = "Translate strategy report from English to professional Turkish"
    required_inputs = ["strategy_report"]
    output_keys = ["translated_report"]
 
    def execute(self, state: dict) -> dict:
        report = state.get("strategy_report", "")
 
        if not report or not isinstance(report, str):
            logger.warning("No strategy report found for translation")
            return {"translated_report": ""}
 
        logger.info(f"Translating report ({len(report)} chars) to Turkish...")
 
        prompt = (
            "Translate the following Corporate Sales Strategy Report into professional Turkish.\n\n"
            "--- BEGIN REPORT ---\n"
            f"{report}\n"
            "--- END REPORT ---"
        )
 
        try:
            translated = invoke_llm(
                TRANSLATOR_SYSTEM_PROMPT,
                prompt,
                temperature=0.2,
                max_tokens=4096,
            )
            self.metrics.record_llm_call(tokens=len(translated.split()))
            logger.info(f"✅ Turkish translation complete: {len(translated)} chars")
            return {"translated_report": translated}
        except Exception as e:
            logger.warning(f"⚠️ Translation failed ({e}), returning empty translation")
            return {
                "translated_report": (
                    f"# ⚠️ Çeviri Başarısız\n\n"
                    f"LLM hizmeti çeviri sırasında kullanılamadı.\n"
                    f"**Hata:** {e}\n\n"
                    f"Lütfen LLM hizmeti hazır olduğunda tekrar deneyiniz.\n"
                )
            }
 
 
# Module-level callable for LangGraph
translator_agent = TranslatorAgent()
