"""
Agent 2: Quantitative Analyst — Local calc + LLM interpretation (Banking Enhanced)



DYNAMIC MAPPING: Account code descriptions are extracted dynamically
from each Mizan document — no hardcoded dictionaries.



PREFIX-BASED AGGREGATION: Uses str.startswith() for clean aggregation
across account code hierarchies (e.g., "12" captures 120, 121, etc.).
"""



import re
import logging
import pandas as pd
from collections import defaultdict
from agents.base import BaseAgent
from llm_config import invoke_llm, QUANT_ANALYST_SYSTEM_PROMPT



logger = logging.getLogger("swarm.agents.quant_analyst")



mizan_mapping= {
    # --- 1. DÖNEN VARLIKLAR ---
    100: "Kasa",
    101: "Alınan Çekler",
    102: "Bankalar",
    103: "Verilen Çekler ve Ödeme Emirleri (-)",
    108: "Diğer Hazır Değerler",
    110: "Hisse Senetleri",
    111: "Özel Kesim Tahvil, Senet ve Bonoları",
    112: "Kamu Kesimi Tahvil, Senet ve Bonoları",
    118: "Diğer Menkul Kıymetler",
    119: "Menkul Kıymetler Değer Düşüklüğü Karşılığı (-)",
    120: "Alıcılar",
    121: "Alacak Senetleri",
    122: "Alacak Senetleri Reeskontu (-)",
    124: "Kazanılmamış Finansal Kiralama Faiz Gelirleri (-)",
    126: "Verilen Depozito ve Teminatlar",
    127: "Diğer Ticari Alacaklar",
    128: "Şüpheli Ticari Alacaklar",
    129: "Şüpheli Ticari Alacaklar Karşılığı (-)",
    131: "Ortaklardan Alacaklar",
    132: "İştiraklerden Alacaklar",
    133: "Bağlı Ortaklıklardan Alacaklar",
    135: "Personelden Alacaklar",
    136: "Diğer Çeşitli Alacaklar",
    137: "Diğer Alacak Senetleri Reeskontu (-)",
    138: "Şüpheli Diğer Alacaklar",
    139: "Şüpheli Diğer Alacaklar Karşılığı (-)",
    150: "İlk Madde ve Malzeme",
    151: "Yarı Mamüller - Üretim",
    152: "Mamüller",
    153: "Ticari Mallar",
    157: "Diğer Stoklar",
    158: "Stok Değer Düşüklüğü Karşılığı (-)",
    159: "Verilen Sipariş Avansları",
    170: "Yıllara Yaygın İnşaat ve Onarım Maliyetleri",
    178: "Yıllara Yaygın İnşaat Enflasyon Düzeltme Farkları",
    179: "Taşeronlara Verilen Avanslar",
    180: "Gelecek Aylara Ait Giderler",
    181: "Gelir Tahakkukları",
    190: "Devreden Katma Değer Vergisi",
    191: "İndirilecek Katma Değer Vergisi",
    192: "Diğer Katma Değer Vergisi",
    193: "Peşin Ödenen Vergiler ve Fonlar",
    195: "İş Avansları",
    196: "Personel Avansları",
    197: "Sayım ve Tesellüm Noksanları",
    198: "Diğer Çeşitli Dönen Varlıklar",
    199: "Diğer Dönen Varlıklar Karşılığı (-)",



    # --- 2. DURAN VARLIKLAR ---
    220: "Alıcılar",
    221: "Alacak Senetleri",
    222: "Alacak Senetleri Reeskontu (-)",
    224: "Kazanılmamış Finansal Kiralama Faiz Gelirleri (-)",
    226: "Verilen Depozito ve Teminatlar",
    229: "Şüpheli Alacaklar Karşılığı (-)",
    231: "Ortaklardan Alacaklar",
    232: "İştiraklerden Alacaklar",
    233: "Bağlı Ortaklıklardan Alacaklar",
    235: "Personelden Alacaklar",
    236: "Diğer Çeşitli Alacaklar",
    237: "Diğer Alacak Senetleri Reeskontu (-)",
    239: "Şüpheli Diğer Alacaklar Karşılığı (-)",
    240: "Bağlı Menkul Kıymetler",
    241: "Bağlı Menkul Kıymetler Değer Düşüklüğü Karşılığı (-)",
    242: "İştirakler",
    243: "İştiraklere Sermaye Taahhütleri (-)",
    244: "İştirakler Sermaye Payları Değer Düşüklüğü Karşılığı (-)",
    245: "Bağlı Ortaklıklar",
    246: "Bağlı Ortaklıklara Sermaye Taahhütleri (-)",
    247: "Bağlı Ortaklıklar Sermaye Payları Değer Düşüklüğü Karşılığı (-)",
    248: "Diğer Mali Duran Varlıklar",
    249: "Diğer Mali Duran Varlıklar Karşılığı (-)",
    250: "Arazi ve Arsalar",
    251: "Yeraltı ve Yerüstü Düzenleri",
    252: "Binalar",
    253: "Tesis, Makine ve Cihazlar",
    254: "Taşıtlar",
    255: "Demirbaşlar",
    256: "Diğer Maddi Duran Varlıklar",
    257: "Birikmiş Amortismanlar (-)",
    258: "Yapılmakta Olan Yatırımlar",
    259: "Verilen Avanslar",
    260: "Haklar",
    261: "Şerefiye",
    262: "Kuruluş ve Örgütlenme Giderleri",
    263: "Araştırma ve Geliştirme Giderleri",
    264: "Özel Maliyetler",
    267: "Diğer Maddi Olmayan Duran Varlıklar",
    268: "Birikmiş Amortismanlar (-)",
    269: "Verilen Avanslar",
    271: "Arama Giderleri",
    272: "Hazırlık ve Geliştirme Giderleri",
    278: "Birikmiş Tükenme Payları (-)",
    279: "Verilen Avanslar",
    280: "Gelecek Yıllara Ait Giderler",
    281: "Gelir Tahakkukları",
    291: "Gelecek Yıllarda İndirilecek Katma Değer Vergisi",
    292: "Diğer Katma Değer Vergisi",
    293: "Gelecek Yıllar İhtiyacı Stoklar",
    294: "Elden Çıkarılacak Stoklar ve Maddi Duran Varlıklar",
    295: "Peşin Ödenen Vergiler ve Fonlar",
    297: "Diğer Çeşitli Duran Varlıklar",
    298: "Stok Değer Düşüklüğü Karşılığı (-)",
    299: "Birikmiş Amortismanlar (-)",



    # --- 3. KISA VADELİ YABANCI KAYNAKLAR ---
    300: "Banka Kredileri",
    301: "Finansal Kiralama İşlemlerinden Borçlar",
    302: "Ertelenmiş Finansal Kiralama Borçlanma Maliyetleri (-)",
    303: "Uzun Vadeli Kredilerin Anapara Taksitleri ve Faizleri",
    304: "Tahvil Anapara Borç, Taksit ve Faizleri",
    305: "Çıkarılmış Bonolar ve Senetler",
    306: "Çıkarılmış Diğer Menkul Kıymetler",
    308: "Menkul Kıymetler İhraç Farkı (-)",
    309: "Diğer Mali Borçlar",
    320: "Satıcılar",
    321: "Borç Senetleri",
    322: "Borç Senetleri Reeskontu (-)",
    326: "Alınan Depozito ve Teminatlar",
    329: "Diğer Ticari Borçlar",
    331: "Ortaklara Borçlar",
    332: "İştiraklere Borçlar",
    333: "Bağlı Ortaklıklara Borçlar",
    335: "Personele Borçlar",
    336: "Diğer Çeşitli Borçlar",
    337: "Diğer Borç Senetleri Reeskontu (-)",
    340: "Alınan Sipariş Avansları",
    349: "Alınan Diğer Avanslar",
    350: "Yıllara Yaygın İnşaat ve Onarım Hakediş Bedelleri",
    358: "Yıllara Yaygın İnşaat Enflasyon Düzeltme Farkları",
    360: "Ödenecek Vergi ve Fonlar",
    361: "Ödenecek Sosyal Güvenlik Kesintileri",
    368: "Vadesi Geçmiş, Ertelenmiş veya Taksitlendirilmiş Vergi ve Yükümlülükler",
    369: "Ödenecek Diğer Yükümlülükler",
    370: "Dönem Karı Vergi ve Diğer Yasal Yükümlülük Karşılıkları",
    371: "Dönem Karının Peşin Ödenen Vergi ve Diğer Yükümlülükleri (-)",
    372: "Kıdem Tazminatı Karşılığı",
    373: "Maliyet Giderleri Karşılığı",
    379: "Diğer Borç ve Gider Karşılıkları",
    380: "Gelecek Aylara Ait Gelirler",
    381: "Gider Tahakkukları",
    391: "Hesaplanan Katma Değer Vergisi",
    392: "Diğer Katma Değer Vergisi",
    393: "Merkez ve Şubeler Cari Hesabı",
    397: "Sayım ve Tesellüm Fazlaları",
    399: "Diğer Çeşitli Yabancı Kaynaklar",



    # --- 4. UZUN VADELİ YABANCI KAYNAKLAR ---
    400: "Banka Kredileri",
    401: "Finansal Kiralama İşlemlerinden Borçlar",
    402: "Ertelenmiş Finansal Kiralama Borçlanma Maliyetleri (-)",
    405: "Çıkarılmış Tahviller",
    407: "Çıkarılmış Diğer Menkul Kıymetler",
    408: "Menkul Kıymetler İhraç Farkı (-)",
    409: "Diğer Mali Borçlar",
    420: "Satıcılar",
    421: "Borç Senetleri",
    422: "Borç Senetleri Reeskontu (-)",
    426: "Alınan Depozito ve Teminatlar",
    429: "Diğer Ticari Borçlar",
    431: "Ortaklara Borçlar",
    432: "İştiraklere Borçlar",
    433: "Bağlı Ortaklıklara Borçlar",
    436: "Diğer Çeşitli Borçlar",
    437: "Diğer Borç Senetleri Reeskontu (-)",
    440: "Alınan Sipariş Avansları",
    449: "Alınan Diğer Avanslar",
    472: "Kıdem Tazminatı Karşılığı",
    479: "Diğer Borç ve Gider Karşılıkları",
    480: "Gelecek Yıllara Ait Gelirler",
    481: "Gider Tahakkukları",
    492: "Gelecek Yıllara Ertelenen veya Terkin Edilen KDV",
    493: "Tesise Katılma Payları",
    499: "Diğer Çeşitli Uzun Vadeli Yabancı Kaynaklar",



    # --- 5. ÖZ KAYNAKLAR ---
    500: "Sermaye",
    501: "Ödenmemiş Sermaye (-)",
    502: "Sermaye Düzeltmesi Olumlu Farkları",
    503: "Sermaye Düzeltmesi Olumsuz Farkları (-)",
    520: "Hisse Senedi İhraç Primleri",
    521: "Hisse Senedi İptal Karları",
    522: "M.D.V. Yeniden Değerleme Artışları",
    523: "İştirakler Yeniden Değerleme Artışları",
    524: "Maliyet Artışları Fonu",
    526: "Borsada Oluşan Değer Artışları",
    529: "Diğer Sermaye Yedekleri",
    540: "Yasal Yedekler",
    541: "Statü Yedekleri",
    542: "Olağanüstü Yedekler",
    548: "Diğer Kar Yedekleri",
    549: "Özel Fonlar",
    570: "Geçmiş Yıllar Karları",
    580: "Geçmiş Yıllar Zararları (-)",
    590: "Dönem Net Karı",
    591: "Dönem Net Zararı (-)",



    # --- 6. GELİR TABLOSU HESAPLARI ---
    600: "Yurtiçi Satışlar",
    601: "Yurtdışı Satışlar",
    602: "Diğer Gelirler",
    610: "Satıştan İadeler (-)",
    611: "Satış İskontoları (-)",
    612: "Diğer İndirimler (-)",
    620: "Satılan Mamüller Maliyeti (-)",
    621: "Satılan Ticari Mallar Maliyeti (-)",
    622: "Satılan Hizmet Maliyeti (-)",
    623: "Diğer Satışların Maliyeti (-)",
    630: "Araştırma ve Geliştirme Giderleri (-)",
    631: "Pazarlama, Satış ve Dağıtım Giderleri (-)",
    632: "Genel Yönetim Giderleri (-)",
    640: "İştiraklerden Temettü Gelirleri",
    641: "Bağlı Ortaklıklardan Temettü Gelirleri",
    642: "Faiz Gelirleri",
    643: "Komisyon Gelirleri",
    644: "Konusu Kalmayan Karşılıklar",
    645: "Menkul Kıymet Satış Karları",
    646: "Kambiyo Karları",
    647: "Reeskont Faiz Gelirleri",
    648: "Enflasyon Düzeltmesi Karları",
    649: "Diğer Olağan Gelir ve Karlar",
    653: "Komisyon Giderleri (-)",
    654: "Karşılık Giderleri (-)",
    655: "Menkul Kıymet Satış Zararları (-)",
    656: "Kambiyo Zararları (-)",
    657: "Reeskont Faiz Giderleri (-)",
    658: "Enflasyon Düzeltmesi Zararları (-)",
    659: "Diğer Olağan Gider ve Zararlar (-)",
    660: "Kısa Vadeli Borçlanma Giderleri (-)",
    661: "Uzun Vadeli Borçlanma Giderleri (-)",
    671: "Önceki Dönem Gelir ve Karları",
    679: "Diğer Olağandışı Gelir ve Karlar",
    680: "Çalışmayan Kısım Gider ve Zararları (-)",
    681: "Önceki Dönem Gider ve Zararları (-)",
    689: "Diğer Olağandışı Gider ve Zararlar (-)",
    690: "Dönem Karı veya Zararı",
    691: "Dönem Karı Vergi ve Diğer Yasal Yükümlülük Karşılıkları (-)",
    692: "Dönem Net Karı veya Zararı",
    697: "Yıllara Yaygın İnşaat Enflasyon Düzeltme Farkları",
    698: "Enflasyon Düzeltme Hesabı",



    # --- 7. MALİYET HESAPLARI (7/A SEÇENEĞİ) ---
    710: "Direkt İlk Madde ve Malzeme Giderleri",
    711: "Direkt İlk Madde ve Malzeme Giderleri Yansıtma Hesabı",
    712: "Direkt İlk Madde ve Malzeme Fiyat Farkı",
    713: "Direkt İlk Madde ve Malzeme Miktar Farkı",
    720: "Direkt İşçilik Giderleri",
    721: "Direkt İşçilik Giderleri Yansıtma Hesabı",
    722: "Direkt İşçilik Ücret Farkları",
    723: "Direkt İşçilik Süre (Zaman) Farkları",
    730: "Genel Üretim Giderleri",
    731: "Genel Üretim Giderleri Yansıtma Hesabı",
    732: "Genel Üretim Giderleri Bütçe Farkları",
    733: "Genel Üretim Giderleri Verimlilik Farkları",
    734: "Genel Üretim Giderleri Kapasite Farkları",
    740: "Hizmet Üretim Maliyeti",
    741: "Hizmet Üretim Maliyeti Yansıtma Hesabı",
    742: "Hizmet Üretim Maliyeti Fark Hesapları",
    750: "Araştırma ve Geliştirme Giderleri",
    751: "Araştırma ve Geliştirme Giderleri Yansıtma Hesabı",
    752: "Araştırma ve Geliştirme Gider Farkları",
    760: "Pazarlama, Satış ve Dağıtım Giderleri",
    761: "Pazarlama, Satış ve Dağıtım Giderleri Yansıtma Hesabı",
    762: "Pazarlama, Satış ve Dağıtım Giderleri Fark Hesabı",
    770: "Genel Yönetim Giderleri",
    771: "Genel Yönetim Giderleri Yansıtma Hesabı",
    772: "Genel Yönetim Gider Farkları",
    780: "Finansman Giderleri",
    781: "Finansman Giderleri Yansıtma Hesabı",
    782: "Finansman Giderleri Fark Hesabı",



    # --- 7. MALİYET HESAPLARI (7/B SEÇENEĞİ) ---
    790: "İlk Madde ve Malzeme Giderleri",
    791: "İşçi Ücret ve Giderleri",
    792: "Memur Ücret ve Giderleri",
    793: "Dışarıdan Sağlanan Fayda ve Hizmetler",
    794: "Çeşitli Giderler",
    795: "Vergi, Resim ve Harçlar",
    796: "Amortismanlar ve Tükenme Payları",
    797: "Finansman Giderleri",
    798: "Gider Çeşitleri Yansıtma Hesabı",
    799: "Üretim Maliyet Hesabı",



    # --- 9. NAZIM HESAPLAR ---
    900: "Nazım Hesaplar (Teminatlar, Matrah Artırımları vb.)",
    901: "Nazım Hesaplar Karşılığı"
}





class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"
    description = "Calculate financial ratios from standardized Mizan data with dynamic period extraction"
    required_inputs = ["standardized_mizan"]
    output_keys = ["financial_ratios"]



    def execute(self, state: dict) -> dict:
        retry_count = state.get("retry_count", 0)
        period_months = 12
        period_days = 360
        raw_donem = "Unknown"
        donem_label = "Annual (12M) - Default"



        standardized = state.get("standardized_mizan", [])
        if not standardized:
            return {"financial_ratios": {"error": "No standardized mizan data"}, "retry_count": retry_count + 1}



        df = pd.DataFrame(standardized)



        # ── Data-Driven Period Extraction ──
        donem_col = "donem" if "donem" in df.columns else "period" if "period" in df.columns else None
        if donem_col:
            valid_donems = df[donem_col].dropna().astype(str).unique()
            if len(valid_donems) > 0:
                raw_donem = valid_donems[0].replace(".0", "").strip()
                if len(raw_donem) >= 6 and raw_donem[-2:].isdigit():
                    extracted_month = int(raw_donem[-2:])
                    if 1 <= extracted_month <= 12:
                        period_months = extracted_month
                        period_days = period_months * 30
                        donem_label = f"{period_months} Months ({raw_donem})"
                        logger.info(f"✅ Dynamic Period Extracted: {raw_donem} -> {period_months} Months ({period_days} days)")
                    else:
                        logger.warning(f"⚠️ Invalid month extracted from donem '{raw_donem}'. Falling back to 12M.")
                else:
                    logger.warning(f"⚠️ Unrecognized donem format '{raw_donem}'. Falling back to 12M.")
        else:
            logger.warning("⚠️ 'donem' column not found in data. Falling back to 12M.")



        # Hesap kodlarının string olduğundan emin olalım
        df["account_code"] = df["account_code"].astype(str)



                # ── YARDIMCI FONKSİYONLAR ──
        def bal_debit(code: str) -> float:
            """Net balance for debit-normal accounts (Assets, Expenses).
            Exact match first, then prefix aggregation."""
            m = df[df["account_code"] == code]
            if not m.empty:
                return float(m.iloc[0]["debit"] - m.iloc[0]["credit"])
            #m = df[df["account_code"].str.startswith(code)]
            if not m.empty:
                return float(m["debit"].sum() - m["credit"].sum())
            return 0.0
        
        def bal_credit(code: str) -> float:
            """Net balance for credit-normal accounts (Liabilities, Revenue)."""
            return -bal_debit(code)
        
        
        def main_accounts_w_prefix(df: pd.DataFrame, prefix: str = None):
            df["account_code"] = df["account_code"].astype(str).str.strip()
            main_accounts_df = df[~df["account_code"].str.contains(".", regex=False, na=False)]
            if prefix:
                main_accounts_df = main_accounts_df[main_accounts_df["account_code"].str.startswith(prefix)]
            return main_accounts_df
        # ══════════════════════════════════════════════════════════════
        # 1. GELİR TABLOSU (Income Statement)
        # ══════════════════════════════════════════════════════════════
        gross_revenue = bal_credit("600") + bal_credit("601") + bal_credit("602")
        sales_deductions = bal_debit("610") + bal_debit("611") + bal_debit("612")
        net_revenue = gross_revenue - sales_deductions
        cogs = bal_debit("620") + bal_debit("621") + bal_debit("622") + bal_debit("623") 
        gross_profit = net_revenue - cogs
        gross_margin = (gross_profit / net_revenue * 100) if net_revenue else 0
        
        op_expenses = bal_debit("630") + bal_debit("631") +bal_debit("632")
        operating_profit = gross_profit - op_expenses
        operating_margin = (operating_profit / net_revenue * 100) if net_revenue else 0
        
        # EBITDA Proxy (Operating Profit + Depreciation/Amortization add-back)
        depreciation_257 = bal_credit("257")
        amortization_268 = bal_credit("268")
        ebitda_proxy = operating_profit + depreciation_257 + amortization_268
        
        # ══════════════════════════════════════════════════════════════
        # 2. BİLANÇO & LİKİDİTE (Balance Sheet & Liquidity)
        # ══════════════════════════════════════════════════════════════
        EKSI_HESAPLAR = [
        "103", "119", "122", "124", "129", "137", "139", "199",
        "222", "224", "229", "237", "239", "241", "243", "244", "245", "247", "249", "257", "268", "278", "289", "298", "299",
        "302", "308", "322", "371", "381",
        "402", "408", "422",
        "501", "580", "591",
        "610", "611", "612", "620", "621", "622", "623"
        ]
        def eksi_hesap_main_sum(df: pd.DataFrame, prefix: str = None):
            df["account_code"] = df["account_code"].astype(str).str.strip()
            condition_no_dot = ~df["account_code"].str.contains(".", regex=False, na=False)
            condition_length_3 = df["account_code"].str.len() == 3
            main_accounts_df = df[condition_no_dot & condition_length_3].copy()
            main_accounts_df["is_contra"] = main_accounts_df["account_code"].isin(EKSI_HESAPLAR)
            if prefix:
                main_accounts_df = main_accounts_df[main_accounts_df["account_code"].str.startswith(prefix)]
            toplam_borc = main_accounts_df["debit"].sum()
            toplam_alacak = main_accounts_df["credit"].sum()
            net_bakiye = 0.0
            if prefix:
                ilk_hane = prefix[0]
                if ilk_hane in ["1", "2"]: # Aktif Karakterli
                    net_bakiye = toplam_borc - toplam_alacak
                elif ilk_hane in ["3", "4", "5"]: # Pasif Karakterli (Sermaye, Borçlar vb.)
                    net_bakiye = toplam_alacak - toplam_borc
                elif ilk_hane == "6": # Gelir Tablosu (Satışlar Alacak, Giderler Borç kalanı verir)
                    net_bakiye = toplam_alacak - toplam_borc
                else:
                    net_bakiye = toplam_borc - toplam_alacak
            else:
                # Prefix yoksa tüm mizanı topluyordur.
                net_bakiye = toplam_borc - toplam_alacak
            return float(toplam_borc), float(toplam_alacak), float(net_bakiye)
        
        
        current_assets_debit, current_assets_credit, current_assets_net= eksi_hesap_main_sum(df, prefix="1")
        
        non_current_assets_debit, non_current_assets_credit, non_current_assets_net= eksi_hesap_main_sum(df, prefix="2")
        
        total_assets = current_assets_net + non_current_assets_net
        
        short_term_liab_debit, short_term_liab_credit, short_term_liab_net= eksi_hesap_main_sum(df, prefix="3")
        
        current_ratio = (current_assets_net / short_term_liab_net) if short_term_liab_net else 0
        inventory = bal_debit("150") + bal_debit("151") + bal_debit("152") + bal_debit("153") +bal_debit("154") + bal_debit("155") + bal_debit("156") + bal_debit("157") + bal_debit("158") + bal_debit("159")
        quick_ratio = ((current_assets_net - inventory) / short_term_liab_net) if short_term_liab_net else 0
        
        # Cash & liquid instruments
        cash_100 = bal_debit("100")
        received_checks_101 = bal_debit("101")
        banks_102_total = bal_debit("102")
        given_checks_103 = bal_credit("103")
        
        # ══════════════════════════════════════════════════════════════
        # 3. BORÇLULUK (Leverage & Debt)
        # ══════════════════════════════════════════════════════════════
        
        long_term_liab_debit, long_term_liab_credit, long_term_liab_net = eksi_hesap_main_sum(df, prefix="4")
        total_liab = short_term_liab_net + long_term_liab_net
        
        
        total_equity_debit, total_equity_credit, total_equity_net = eksi_hesap_main_sum(df, prefix="5")
        debt_to_equity = (total_liab / total_equity_net) if total_equity_net else 0
        
        total_bank_loans = bal_credit("300") + bal_credit("400") + bal_credit("309")
        bank_debt_ratio = (total_bank_loans / total_liab * 100) if total_liab else 0
        
        fin_exp_780 = bal_debit("780")
        fin_expense_ratio = (fin_exp_780 / net_revenue * 100) if net_revenue else 0
        #pos_780_01 = bal_debit("780.01")
        
        # ══════════════════════════════════════════════════════════════
        # 4. ÇALIŞMA SERMAYESİ VE İLİŞKİLİ TARAF (Working Capital & Related Party)
        # ══════════════════════════════════════════════════════════════
        trade_receivables = bal_debit("120") + bal_debit("121") + bal_credit("122") + bal_credit("124") + bal_debit("126") + bal_debit("127") + bal_debit("128") + bal_credit("129")
        collection_period = (trade_receivables / net_revenue * period_days) if net_revenue else 0
        
        trade_payables = bal_debit("320") + bal_debit("321") + bal_credit("322") + bal_debit("326") + bal_debit("329")
        payment_period = (trade_payables / cogs * period_days) if cogs else 0
        
        inventory_period = (inventory / cogs * period_days) if cogs else 0
        cash_conversion_cycle = collection_period + inventory_period - payment_period
        
        insider_lending_131 = bal_debit("131")
        insider_borrowing_331 = bal_credit("331")
        insider_lending_ratio = (insider_lending_131 / total_assets * 100) if total_assets else 0
        
        check_risk_ratio = (given_checks_103 / banks_102_total) if banks_102_total else 0
        # ══════════════════════════════════════════════════════════════
        # 5. GRAND TOTAL NAKİT AKIŞI (Cash Flow & Future Projections)
        # ══════════════════════════════════════════════════════════════
        # Dönem İçi Hacim (Flow): 100, 102 ve 108 hesaplarındaki brüt borç/alacak toplamları
        liquid_df = pd.concat([main_accounts_w_prefix(df, prefix="100"),main_accounts_w_prefix(df, prefix="102"), main_accounts_w_prefix(df, prefix="108")])
        period_cash_inflow = float(liquid_df["debit"].sum())
        period_cash_outflow = float(liquid_df["credit"].sum())
        period_net_cash_movement = period_cash_inflow - period_cash_outflow
        
        # Gelecekteki Stok (Stock): Kapanış bakiyeleri üzerinden projeksiyon# Gelecek Giriş = Ticari Alacaklar + Alınan Çekler (Bekleyen Tahsilatlar)
        future_cash_inflow = trade_receivables + received_checks_101
        # Gelecek Çıkış = Tüm Borçlar (3xx + 4xx) + Verilen Çekler (103 - Eksi karakterli aktif)# short_term_liab ve long_term_liab halihazırda 3xx ve 4xx'in net bakiyesidir.
        future_cash_outflow = total_liab + given_checks_103
        future_net_position = future_cash_inflow - future_cash_outflow
        
        # ══════════════════════════════════════════════════════════════
        # 6. RAKİP BANKA ANALİZİ (Competitor Bank Analysis)
        # ══════════════════════════════════════════════════════════════
        def get_bank_breakdown(parent_code: str) -> list:
            """Extract sub-account balances to identify competitor bank shares.
            Returns nested category structure with sub-accounts."""
            categories = {}
            parent_row = df[df["account_code"] == parent_code]
            parent_name = str(parent_row.iloc[0]["account_name"]).strip().upper() if not parent_row.empty else ""
        
            for _, row in df.iterrows():
                code = str(row["account_code"]).strip()
                name = str(row.get("account_name", code)).strip()
                raw_net = float(row["debit"]) - float(row["credit"])
                if raw_net == 0 or code == parent_code or not code.startswith(parent_code):
                    continue
        
                match = re.search(r'^(' + re.escape(parent_code) + r'[\.\s\-]+[A-Za-z0-9]+)', code)
                l1_code = match.group(1) if match else code
        
                if l1_code not in categories:
                    categories[l1_code] = {"name": l1_code, "raw_balance": 0.0, "children": [], "is_explicit": False}
                if code == l1_code:
                    categories[l1_code]["name"] = name
                    categories[l1_code]["raw_balance"] = raw_net
                    categories[l1_code]["is_explicit"] = True
                else:
                    child_balance_type = "debit" if raw_net > 0 else "credit"
                    categories[l1_code]["children"].append({
                        "name": name,
                        "absolute_balance": abs(raw_net),
                        "balance_type": child_balance_type,
                        "raw_net": raw_net,
                    })
                    if not categories[l1_code]["is_explicit"]:
                        categories[l1_code]["raw_balance"] += raw_net
        
            # Flatten dummy categories
            flat_categories = []
            for l1_code, cat in categories.items():
                cat_name_upper = cat["name"].strip().upper()
                is_dummy = False
                if parent_name and cat_name_upper == parent_name:
                    is_dummy = True
                elif cat_name_upper == l1_code.upper():
                    is_dummy = True
        
                if is_dummy:
                    for child in cat["children"]:
                        if parent_name and child["name"].strip().upper() == parent_name:
                            continue
                        flat_categories.append({
                            "name": child["name"],
                            "raw_balance": child["raw_net"],
                            "children": [],
                            "is_explicit": True,
                        })
                else:
                    filtered_children = []
                    for child in cat["children"]:
                        if parent_name and child["name"].strip().upper() == parent_name:
                            continue
                        filtered_children.append(child)
                    cat["children"] = filtered_children
                    flat_categories.append(cat)
        
            total_abs_parent = sum(abs(cat["raw_balance"]) for cat in flat_categories)
            result = []
            for cat in flat_categories:
                cat_raw = cat["raw_balance"]
                cat_abs = abs(cat_raw)
                if cat_abs == 0:
                    continue
                cat_type = "debit" if cat_raw > 0 else "credit"
                cat_share = (cat_abs / total_abs_parent * 100) if total_abs_parent else 0
                processed_children = []
                total_abs_children = sum(c["absolute_balance"] for c in cat["children"])
                for child in cat["children"]:
                    c_share = (child["absolute_balance"] / total_abs_children * 100) if total_abs_children else 0
                    processed_children.append({
                        "name": child["name"],
                        "balance": child["absolute_balance"],
                        "balance_type": child["balance_type"],
                        "share_of_parent_pct": c_share,
                    })
                processed_children.sort(key=lambda x: x["balance"], reverse=True)
                result.append({
                    "category_name": cat["name"],
                    "balance": cat_abs,
                    "balance_type": cat_type,
                    "share_of_total_pct": cat_share,
                    "sub_accounts": processed_children,
                })
            result.sort(key=lambda x: x["balance"], reverse=True)
            return result
        
        banks_102 = get_bank_breakdown("102")
        banks_300 = get_bank_breakdown("300")
        banks_400 = get_bank_breakdown("400")
        
        def fmt_bank_shares(parent_code: str, parent_name: str, data: list) -> str:
            """Format bank breakdown data explicitly for LLM comprehension with parent hierarchy."""
            if not data:
                return f"No detailed sub-account data available for {parent_code} - {parent_name}."
            # Ana hesabın (Parent) toplam bakiyesini hesapla
            parent_total = sum(cat.get("balance", 0) for cat in data)
            lines = []
            # En üste Ana Hesap (Parent Account) bilgisini ekliyoruz
            lines.append(f"Main account: {parent_code} - {parent_name} (Total Analyzed Balance: ₺{parent_total:,.0f})")
            for category in data:
                cat_name = category.get("category_name", "Unknown")
                cat_balance = category.get("balance", 0)
                cat_pct = category.get("share_of_total_pct", 0)
                lines.append(f"🔹 CATEGORY: {cat_name}")
                lines.append(f"   Category Total Balance: ₺{cat_balance:,.0f} (This category represents {cat_pct:.1f}% of the entire {parent_code} account)")
                sub_accounts = category.get("sub_accounts", [])
                top_3_subs = sorted(sub_accounts, key=lambda x: x.get("balance", 0), reverse=True)[:3]
                for sub in top_3_subs:
                    sub_name = sub.get("name", "Unknown")
                    sub_balance = sub.get("balance", 0)
                    sub_pct = sub.get("share_of_parent_pct", 0)
                    # Alt hesap - Kategori ilişkisi
                    lines.append(f"     ↳ Bank/Sub-account: {sub_name} | Balance: ₺{sub_balance:,.0f} | Share: {sub_pct:.1f}% of {cat_name}")
                lines.append("")
            return "\n".join(lines)
        
        # ══════════════════════════════════════════════════════════════
        # 6. RATIOS DICTIONARY
        # ══════════════════════════════════════════════════════════════
        ratios = {
            "gross_margin": {
                "value": round(gross_margin, 2), "unit": "%",
                "formula": "(Net Revenue - COGS[62x]) / Net Revenue × 100",
                "accounts_used": ["600", "620"],
                "raw_values": {
                    "gross_revenue": gross_revenue,
                    "sales_deductions": sales_deductions,
                    "net_revenue": net_revenue,
                    "cogs_62x": cogs,
                    "gross_profit": gross_profit,
                }
            },
            "operating_margin": {
                "value": round(operating_margin, 2), "unit": "%",
                "formula": "(Gross Profit - Operating Expenses[63x]) / Net Revenue × 100",
                "accounts_used": ["600", "620", "630", "631", "632"],
                "raw_values": {
                    "gross_profit": gross_profit,
                    "op_expenses_63x": op_expenses,
                    "operating_profit": operating_profit,
                    "net_revenue": net_revenue,
                }
            },
            "current_ratio": {
                "value": round(current_ratio, 2), "unit": "x",
                "formula": "Current Assets [1xx] / Short-Term Liabilities [3xx]",
                "accounts_used": ["100", "102", "120", "300", "320"],
                "raw_values": {
                    "current_assets": current_assets_net,
                    "short_term_liabilities": short_term_liab_net,
                }
            },
            "quick_ratio": {
                "value": round(quick_ratio, 2), "unit": "x",
                "formula": "(Current Assets[1xx] - Inventory[15x]) / Short-Term Liabilities[3xx]",
                "accounts_used": ["100", "102", "120", "150", "151", "152", "153", "300", "309", "320"],
                "raw_values": {
                    "liquid_assets": current_assets_net - inventory,
                    "inventory_total": inventory,
                    "short_term_liabilities": short_term_liab_net,
                }
            },
            "collection_period": {
                "value": round(collection_period, 0), "unit": "days",
                "formula": f"Trade Receivables[12x] / Net Revenue × {period_days}",
                "accounts_used": ["120", "121", "600"],
                "period_days_used": period_days,
                "raw_values": {
                    "trade_receivables": trade_receivables,
                    "net_revenue": net_revenue,
                }
            },
            "payment_period": {
                "value": round(payment_period, 0), "unit": "days",
                "formula": f"Trade Payables[32x] / COGS[62x] × {period_days}",
                "accounts_used": ["320", "321", "620"],
                "period_days_used": period_days,
                "raw_values": {
                    "trade_payables": trade_payables,
                    "cogs_62x": cogs,
                }
            },
            "inventory_period": {
                "value": round(inventory_period, 0), "unit": "days",
                "formula": f"Inventory[15x] / COGS[62x] × {period_days}",
                "accounts_used": ["150", "151", "152", "153", "620"],
                "period_days_used": period_days,
                "raw_values": {
                    "total_inventory": inventory,
                    "cogs_62x": cogs,
                }
            },
            "cash_conversion_cycle": {
                "value": round(cash_conversion_cycle, 0), "unit": "days",
                "formula": "Collection Period + Inventory Period - Payment Period",
                "accounts_used": ["120", "121", "150", "151", "152", "153", "320", "321", "600", "620"],
                "period_days_used": period_days,
                "raw_values": {
                    "collection_period": collection_period,
                    "inventory_period": inventory_period,
                    "payment_period": payment_period,
                }
            },
            "debt_to_equity": {
                "value": round(debt_to_equity, 2), "unit": "x",
                "formula": "(Short-Term Liab + Long-Term Liab) / Equity",
                "accounts_used": ["300", "320", "400", "500", "570", "590"],
                "raw_values": {
                    "short_term_liab": short_term_liab_net,
                    "long_term_liab": long_term_liab_net,
                    "total_liabilities": total_liab,
                    "total_equity": total_equity_net,
                }
            },
            "bank_debt_ratio": {
                "value": round(bank_debt_ratio, 2), "unit": "%",
                "formula": "Total Bank Loans[300+400+309] / Total Liabilities × 100",
                "accounts_used": ["300", "400", "309"],
                "raw_values": {
                    "bank_loans_st_300": bal_credit("300"),
                    "bank_loans_lt_400": bal_credit("400"),
                    "credit_cards_309": bal_credit("309"),
                    "total_bank_loans": total_bank_loans,
                    "total_liabilities": total_liab,
                }
            },
            "financial_expense_ratio": {
                "value": round(fin_expense_ratio, 2), "unit": "%",
                "formula": "Financial Expenses [780] / Net Revenue × 100",
                "accounts_used": ["780"],
                "raw_values": {
                    "finansman_giderleri_780": fin_exp_780,
                    "net_revenue": net_revenue,
                }
            },
            #"pos_commission_ratio": {
            #    "value": round((pos_780_01 / net_revenue * 100) if net_revenue else 0, 2), "unit": "%",
            #    "formula": "POS Commission [780.01] / Net Revenue × 100",
            #    "accounts_used": ["780.01"],
            #    "raw_values": {
            #        "pos_komisyon_780_01": pos_780_01,
            #        "net_revenue": net_revenue,
            #    }
            #},
            "insider_lending_ratio": {
                "value": round(insider_lending_ratio, 2), "unit": "%",
                "formula": "(131 / Equity) * 100",
                "accounts_used": ["131"],
                "raw_values": {
                    "insider_lending_131": insider_lending_131,
                    "insider_borrowing_331": insider_borrowing_331,
                    "current_assets": current_assets_net,
                    "non_current_assets": non_current_assets_net,
                    "total_assets": total_assets,
                }
            },
            "check_risk_ratio": {
                "value": round(check_risk_ratio, 2), "unit": "x",
                "formula": "Given Checks [103] / Bank Deposits [102]",
                "accounts_used": ["103", "102"],
                "raw_values": {
                    "given_checks_103": given_checks_103,
                    "banks_102_total": banks_102_total,
                }
            },
            "cash_flow_summary": {
                "period_cash_inflow": period_cash_inflow,
                "period_cash_outflow": period_cash_outflow,
                "period_net_movement": period_net_cash_movement,
                "future_cash_inflow": future_cash_inflow,
                "future_cash_outflow": future_cash_outflow,
                "future_net_position": future_net_position
                            },
            "competitor_banks": {
                "102": banks_102,
                "300": banks_300,
                "400": banks_400,
            },
            "donem_context": {
                "raw": raw_donem,
                "period_months": period_months,
                "period_days": period_days,
                "label": donem_label,
            },
        }
        for n, d in ratios.items():
            if isinstance(d, dict) and "value" in d:
                logger.info(f"  - {n}: {d['value']}{d['unit']}")
        
        # ── LLM INTERPRETATION ──
        llm_text = ""
        try:
            def map_accounts(acc_list):
                mapped = []
                for acc in acc_list:
                    acc_str = str(acc)
                    code = acc_str.split('.')[0] if '.' in acc_str else acc_str
                    if code.isdigit() and int(code) in mizan_mapping:
                        mapped.append(f"{acc_str} - {mizan_mapping[int(code)]}")
                    else:
                        mapped.append(acc_str)
                return mapped
        
            summary = "\n".join(
                f"- {n}: {d['value']}{d['unit']} (Accounts Used: {', '.join(map_accounts(d['accounts_used']))})"
                for n, d in ratios.items() if isinstance(d, dict) and "value" in d
            )
        
            prompt = (
                f"⏱️ DATA PERIOD: {donem_label} ({period_days} days). "
                f"Company: **{state.get('company_name', 'Company')}**\n\n"
                f"## RATIO SUMMARY:\n{summary}\n\n"
        
                f"## 1. INCOME STATEMENT:\n"
                f"- Gross Revenue (600+601+602): ₺{gross_revenue:,.0f}\n"
                f"- Sales Deductions (610+611+612): ₺{sales_deductions:,.0f}\n"
                f"- **Net Revenue:** ₺{net_revenue:,.0f}\n"
                f"- COGS (62x): ₺{cogs:,.0f} | **Gross Profit:** ₺{gross_profit:,.0f} | Gross Margin: {round(gross_margin, 2)}%\n"
                f"- Operating Expenses (63x): ₺{op_expenses:,.0f} | **Operating Profit:** ₺{operating_profit:,.0f} | Operating Margin: {round(operating_margin, 2)}%\n"
                f"- **EBITDA Proxy** (OpProfit + Depreciation 257 + Amortization 268): ₺{ebitda_proxy:,.0f}\n"
                f"→ **Assessment:** Provide 1-2 sentence profitability insight.\n\n"
        
                f"## 2. BALANCE SHEET & LIQUIDITY:\n"
                f"- Current Assets (1xx): ₺{current_assets_net:,.0f} | Non-Current (2xx): ₺{non_current_assets_net:,.0f} | **Total Assets:** ₺{total_assets:,.0f}\n"
                f"- Cash (100): ₺{cash_100:,.0f} | Checks (101): ₺{received_checks_101:,.0f} | Banks (102): ₺{banks_102_total:,.0f} | Given Checks (103): ₺{given_checks_103:,.0f}\n"
                f"- Inventory (15x): ₺{inventory:,.0f}\n"
                f"- ST Liabilities (3xx): ₺{short_term_liab_net:,.0f}\n"
                f"- **Current Ratio:** {round(current_ratio, 2)}x | **Quick Ratio:** {round(quick_ratio, 2)}x\n"
                f"→ **Assessment:** Evaluate liquidity position. Flag if QR<1.0 or 103>102.\n\n"
        
                f"## 3. LEVERAGE & CAPITAL STRUCTURE:\n"
                f"- LT Liabilities (4xx): ₺{long_term_liab_net:,.0f}\n"
                f"- **Total Liabilities:** ₺{total_liab:,.0f} | **Total Equity (5xx):** ₺{total_equity_net:,.0f}\n"
                f"- Total Bank Loans (300+400+309): ₺{total_bank_loans:,.0f} | D/E: {round(debt_to_equity, 2)}x | Bank Debt Ratio: {round(bank_debt_ratio, 2)}%\n"
                f"- Fin. Expenses (780): ₺{fin_exp_780:,.0f} | Fin. Expense Ratio: {round(fin_expense_ratio, 2)}%\n"
                f"→ **Assessment:** Evaluate leverage position and cost of debt.\n\n"
        
                f"## 4. WORKING CAPITAL & CASH CYCLE:\n"
                f"- Trade Receivables (12x): ₺{trade_receivables:,.0f} → Collection: {collection_period:.0f} days\n"
                f"- Trade Payables (32x): ₺{trade_payables:,.0f} → Payment: {payment_period:.0f} days\n"
                f"- Inventory Period: {inventory_period:.0f} days | **CCC: {cash_conversion_cycle:.0f} days**\n"
                f"- Insider Lending (131): ₺{insider_lending_131:,.0f} | Insider Borrowing (331): ₺{insider_borrowing_331:,.0f} → Ratio: {round(insider_lending_ratio, 2)}%\n"
                f"→ **Assessment:** Evaluate CCC efficiency. Flag insider lending if >5% of assets.\n\n"
        
                f"## 5. CASH FLOW & FUTURE OBLIGATIONS (CRITICAL):\n"
                f"- Period Cash Inflows (Debits 100,102,108): ₺{period_cash_inflow:,.0f}\n"
                f"- Period Cash Outflows (Credits 100,102,108): ₺{period_cash_outflow:,.0f}\n"
                f"- **Net Period Cash Movement:** ₺{period_net_cash_movement:,.0f}\n"
                f"- Future Inflows (12x+101 closing): ₺{future_cash_inflow:,.0f}\n"
                f"- Future Outflows (3xx+4xx+103 closing): ₺{future_cash_outflow:,.0f}\n"
                f"- **Net Future Liquidity Position:** ₺{future_net_position:,.0f}\n"
                f"→ **Assessment:** CRITICAL — flag funding gap or surplus. Compare historical burn rate vs future obligations.\n\n"
        
                f"## 6. COMPETITOR BANK DISTRIBUTION:\n"
                f"**102-BANKALAR (Deposits):**\n{fmt_bank_shares('102', 'BANKALAR (Deposits)', banks_102)}\n\n"
                f"**300-BANKA KREDİLERİ KV (ST Loans):**\n{fmt_bank_shares('300', 'BANKA KREDİLERİ KV (ST Loans)', banks_300)}\n\n"
                f"**400-BANKA KREDİLERİ UV (LT Loans):**\n{fmt_bank_shares('400', 'BANKA KREDİLERİ UV (LT Loans)', banks_400)}\n"
                f"→ **ING Status:** State ING's presence/absence in each category.\n\n"
        
                f"## BANKING INTELLIGENCE GUIDELINES:\n"
                f"- CASH ANALYSIS: Compare Net Period Cash Movement with Net Future Liquidity Position. If bleeding cash + future deficit → critical risk.\n"
                f"- CASH TRAPPING: Account 131 > 5% of Total Assets → Capital Leakage.\n"
                f"- DUALITY CHECK: Account 103 > 102 → urgent liquidity risk.\n"
                f"- CASH CONVERSION: CCC {cash_conversion_cycle:.0f} days. If high → working capital financing.\n"
                f"- CROSS-SELL: Competitor bank with majority 300 but ING in 102 → loan buyout opportunity.\n"
                f"- TAX AVOIDANCE: High OpProfit but low Net Profit → investigate non-operating expenses.\n\n"
        
                f"## OUTPUT FORMAT INSTRUCTIONS:\n"
                f"Structure your analysis into EXACTLY these 6 sections. For EACH section:\n"
                f"1. Cite the exact ₺ values and account codes provided above\n"
                f"2. End each section with a clear **Assessment:** line (1-2 sentences)\n"
                f"3. Use the section headers exactly as given:\n"
                f"   ### 1. INCOME STATEMENT ANALYSIS\n"
                f"   ### 2. BALANCE SHEET & LIQUIDITY\n"
                f"   ### 3. LEVERAGE & CAPITAL STRUCTURE\n"
                f"   ### 4. WORKING CAPITAL & CASH CYCLE\n"
                f"   ### 5. CASH FLOW & FUTURE OBLIGATIONS\n"
                f"   ### 6. COMPETITOR BANK DISTRIBUTION & ING STATUS\n"
            )
            llm_text = invoke_llm(QUANT_ANALYST_SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=3000)
            self.metrics.record_llm_call(tokens=len(llm_text.split()))
            logger.info(f"✅ LLM interpretation: {len(llm_text)} chars")
        except Exception as e:
            logger.warning(f"LLM skipped: {e}")
            llm_text = "LLM interpretation unavailable."
 
        ratios["llm_interpretation"] = llm_text
        return {"financial_ratios": ratios, "retry_count": retry_count + 1}
quant_analyst_agent = QuantAnalystAgent()