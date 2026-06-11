#!/usr/bin/env python3
"""
Build data/nace_sector_mapping.json from the İTO meslek grupları PDF
=====================================================================
Source: https://www.ito.org.tr/documents/Uye_Sicil/Dokumanlar/meslek-gruplari.pdf
(Istanbul Chamber of Commerce professional groups — 6-digit NACE Rev.2
codes with Turkish activity descriptions, grouped into ~76 profession groups.)

Each İTO profession group is mapped to one of the TCMB benchmark sectors
used across the pipeline (see data/tcmb_sector_benchmarks.json). The output
JSON maps  sector → sorted list of 6-digit NACE codes, and also keeps the
İTO group provenance plus a 2-digit NACE division fallback for codes that
do not appear in the İTO list.

Usage:
    pip install pypdf
    python scripts/build_nace_mapping.py <path-to-meslek-gruplari.pdf>
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

from pypdf import PdfReader

OUT_PATH = Path(__file__).parent.parent / "data" / "nace_sector_mapping.json"

# İTO profession group id → TCMB benchmark sector.
# Judgment calls for banking-product purposes are noted inline.
GROUP_TO_SECTOR = {
    1: "Retail",                        # PERAKENDE TİCARET
    2: "Trading",                       # TOPTAN VE DIŞ TİCARET
    3: "Trading",                       # BİJUTERİ, OYUNCAK VE HEDİYELİK EŞYA (mostly trade)
    4: "Trading",                       # KUYUMCULUK (jewelry trade + small manufacture)
    5: "Technology",                    # BİLGİ TEKNOLOJİLERİ
    6: "Manufacturing",                 # KAĞIT, KIRTASİYE VE AMBALAJ
    7: "Chemicals",                     # KOZMETİK (20.42 etc.)
    8: "Chemicals",                     # İLAÇ VE TIBBİ CİHAZ (pharma)
    9: "Food & Beverage",               # TOPTAN GIDA VE TEMİZLİK ÜRÜNLERİ
    10: "Manufacturing",                # CAM VE CAM ÜRÜNLERİ
    11: "Agriculture",                  # PEYZAJ VE ÇİÇEKÇİLİK
    12: "Food & Beverage",              # EKMEK, UN VE UNLU MAMÜLLER
    13: "Food & Beverage",              # MEYVE VE SEBZE
    14: "Food & Beverage",              # HAYVANSAL GIDA ÜRÜNLERİ
    15: "Services",                     # EĞİTİM
    16: "Tourism",                      # OTELLER
    17: "Tourism",                      # RESTORAN VE YİYECEK İÇECEK HİZMETLERİ (55-56)
    18: "Services",                     # FİNANS KURULUŞLARI
    19: "Services",                     # MALİ MÜŞAVİRLİK
    20: "Services",                     # SİGORTACILIK
    21: "Services",                     # GAYRİMENKUL HİZMETLERİ
    22: "Transportation & Logistics",   # ŞEHİRİÇİ YOLCU TAŞIMACILIĞI
    24: "Transportation & Logistics",   # LOJİSTİK HİZMETLERİ
    25: "Transportation & Logistics",   # GÜMRÜK MÜŞAVİRLİĞİ
    26: "Transportation & Logistics",   # TAŞIT KİRALAMA VE İLGİLİ HİZMETLER
    27: "Energy",                       # AKARYAKIT
    28: "Services",                     # İŞLETME DESTEK HİZMETLERİ
    29: "Services",                     # MİMARLIK VE MÜHENDİSLİK
    30: "Technology",                   # BİLGİ, İLETİŞİM VE MEDYA
    31: "Services",                     # KÜLTÜR VE SPOR
    32: "Manufacturing",                # BASIM-YAYIN (printing 18.x)
    33: "Services",                     # SAĞLIK HİZMETLERİ
    34: "Textile",                      # DERİ, KÜRK VE SARACİYE (leather 15.x)
    35: "Textile",                      # İPLİK VE ELYAF ÜRÜNLERİ
    36: "Textile",                      # ÖRME KUMAŞ, ÇORAP VE TRİKOTAJ
    37: "Textile",                      # KUMAŞ
    38: "Textile",                      # HAZIR GİYİM VE KONFEKSİYON
    39: "Textile",                      # İÇ GİYİM VE AKSESUARLARI
    40: "Textile",                      # EV TEKSTİLİ
    41: "Textile",                      # HALI-KİLİM VE YER KAPLAMALARI
    42: "Textile",                      # TEKSTİL YAN SANAYİ ÜRÜNLERİ
    43: "Textile",                      # TEKSTİL TERBİYE
    44: "Construction",                 # ALTYAPI İNŞAATI
    45: "Construction",                 # KONUT İNŞAATI
    46: "Construction",                 # İNŞAAT TAAHHÜT
    47: "Construction",                 # RESTORASYON VE İZOLASYON
    48: "Construction",                 # İNŞAAT MALZEMELERİ
    49: "Manufacturing",                # TOPRAK ÜRÜNLERİ (ceramics 23.x)
    50: "Construction",                 # MEKANİK TESİSAT VE DOĞALGAZ TESİSATI (43.22)
    51: "Textile",                      # AYAKKABI VE AYAKKABI YAN SANAYİ (footwear 15.20)
    53: "Automotive",                   # MOTORLU TAŞIT SATIŞ VE SERVİSİ
    55: "Manufacturing",                # DEMİR ÇELİK
    56: "Manufacturing",                # DEMİR DIŞI METALLER
    57: "Manufacturing",                # DÖKÜM VE METAL İŞLEME
    58: "Manufacturing",                # METAL ÜRÜNLER VE MUTFAK EKİPMANLARI
    59: "Manufacturing",                # MAKİNA VE EKİPMANLARI
    60: "Manufacturing",                # TAKIM TEZGAHLARI VE OTOMASYON
    61: "Trading",                      # TEKNİK HIRDAVAT (hardware trade)
    62: "Manufacturing",                # MERMERCİLİK VE MADENCİLİK
    63: "Energy",                       # ENERJİ
    64: "Manufacturing",                # ELEKTRİK EKİPMANLARI
    65: "Manufacturing",                # TEKNİK VE DEKORATİF AYDINLATMA
    66: "Manufacturing",                # ELEKTRİKLİ EV ALETLERİ
    67: "Technology",                   # TELEKOMÜNİKASYON
    68: "Chemicals",                    # PLASTİK VE KAUÇUK
    69: "Chemicals",                    # KİMYEVİ MADDE
    70: "Manufacturing",                # ORMAN ÜRÜNLERİ (wood 16.x)
    71: "Manufacturing",                # MOBİLYA
    72: "Food & Beverage",              # BAKLAVA, PASTA VE ŞEKERLİ MAMÜLLER
    73: "Retail",                       # GÖZLÜKÇÜLÜK VE SAATÇİLİK (47.78 optics retail)
    74: "Food & Beverage",              # ET VE ET ÜRÜNLERİ
    75: "Transportation & Logistics",   # KARGO, POSTA VE DEPOLAMA
    77: "Textile",                      # DOKUMA
    79: "Services",                     # FOTOĞRAFÇILIK
    80: "Retail",                       # ZÜCCACİYE (houseware retail)
    81: "Energy",                       # DOĞAL VE İŞLENMİŞ KATI YAKIT
}

# 2-digit NACE Rev.2 division → sector fallback, used when a customer's
# 6-digit code is not in the İTO list (standard NACE structure).
DIVISION_FALLBACK = {
    "01": "Agriculture", "02": "Agriculture", "03": "Agriculture",
    "05": "Energy", "06": "Energy", "07": "Manufacturing", "08": "Manufacturing", "09": "Energy",
    "10": "Food & Beverage", "11": "Food & Beverage", "12": "Food & Beverage",
    "13": "Textile", "14": "Textile", "15": "Textile",
    "16": "Manufacturing", "17": "Manufacturing", "18": "Manufacturing",
    "19": "Chemicals", "20": "Chemicals", "21": "Chemicals", "22": "Chemicals",
    "23": "Manufacturing", "24": "Manufacturing", "25": "Manufacturing",
    "26": "Technology", "27": "Manufacturing", "28": "Manufacturing",
    "29": "Automotive", "30": "Automotive",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "35": "Energy", "36": "Services", "37": "Services", "38": "Services", "39": "Services",
    "41": "Construction", "42": "Construction", "43": "Construction",
    "45": "Automotive", "46": "Trading", "47": "Retail",
    "49": "Transportation & Logistics", "50": "Transportation & Logistics",
    "51": "Transportation & Logistics", "52": "Transportation & Logistics",
    "53": "Transportation & Logistics",
    "55": "Tourism", "56": "Tourism",
    "58": "Technology", "59": "Technology", "60": "Technology",
    "61": "Technology", "62": "Technology", "63": "Technology",
    "64": "Services", "65": "Services", "66": "Services", "68": "Services",
    "69": "Services", "70": "Services", "71": "Services", "72": "Technology",
    "73": "Services", "74": "Services", "75": "Services", "77": "Services",
    "78": "Services", "79": "Tourism", "80": "Services", "81": "Services",
    "82": "Services", "85": "Services", "86": "Services", "87": "Services",
    "88": "Services", "90": "Services", "91": "Services", "92": "Services",
    "93": "Services", "94": "Services", "95": "Services", "96": "Services",
}

ROW_PATTERN = re.compile(
    r"(\d{1,3})\s*-\s*([A-ZÇĞİÖŞÜ0-9 ,\.\-&/()']+?)\s+(\d{2}\.\d{2}\.\d{2})\s"
)


def build(pdf_path: str):
    reader = PdfReader(pdf_path)
    full_text = "\n".join((p.extract_text() or "") for p in reader.pages)

    sectors = {}
    group_names = {}
    unmapped_groups = set()
    total_rows = 0

    for m in ROW_PATTERN.finditer(full_text):
        gid, gname, code = int(m.group(1)), m.group(2).strip(), m.group(3)
        total_rows += 1
        sector = GROUP_TO_SECTOR.get(gid)
        if sector is None:
            unmapped_groups.add((gid, gname))
            continue
        entry = sectors.setdefault(sector, {"ito_groups": set(), "nace_codes": set()})
        entry["ito_groups"].add(f"{gid} - {gname}")
        entry["nace_codes"].add(code)
        group_names[gid] = gname

    payload = {
        "metadata": {
            "source": "İTO Meslek Grupları — https://www.ito.org.tr/documents/Uye_Sicil/Dokumanlar/meslek-gruplari.pdf",
            "description": "6-digit NACE Rev.2 codes per TCMB benchmark sector, extracted from the Istanbul Chamber of Commerce professional groups list. division_fallback maps 2-digit NACE divisions for codes not present in the İTO list.",
            "generated_on": date.today().isoformat(),
            "total_codes": sum(len(v["nace_codes"]) for v in sectors.values()),
            "total_rows_parsed": total_rows,
        },
        "sectors": {
            k: {
                "ito_groups": sorted(v["ito_groups"]),
                "nace_codes": sorted(v["nace_codes"]),
            }
            for k, v in sorted(sectors.items())
        },
        "division_fallback": DIVISION_FALLBACK,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {OUT_PATH}")
    print(f"Rows parsed: {total_rows} | codes mapped: {payload['metadata']['total_codes']}")
    for sec, v in payload["sectors"].items():
        print(f"  {sec:28s} {len(v['nace_codes']):4d} codes from {len(v['ito_groups'])} İTO groups")
    if unmapped_groups:
        print("UNMAPPED GROUPS (add to GROUP_TO_SECTOR):")
        for gid, gname in sorted(unmapped_groups):
            print(f"  {gid} - {gname}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1])
