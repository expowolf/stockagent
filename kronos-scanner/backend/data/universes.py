# data/universes.py
"""
Ticker universes.

The Episodic Pivot lives in small and mid caps. A mega-cap essentially never
gaps 10% on 10x volume after months of being ignored — it is too widely held
and too closely covered for the market to "completely reassess" it overnight.
Scanning mega-caps for EPs is looking in the one place the setup cannot form.

SMALL_MID is therefore the default: liquid US names roughly $2-60, weighted
toward the sectors where hard catalysts actually land — biotech (FDA, trial
readouts), high-beta tech, energy, uranium/lithium, and crypto miners.

MEGA_CAP is kept for regime context and for Setup B (the consolidation
breakout), which does work on larger names.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Small / mid cap — the EP hunting ground
# --------------------------------------------------------------------------- #
SMALL_MID = [
    # Biotech / pharma — FDA and trial catalysts are the classic EP trigger
    "SRPT", "IONS", "EXEL", "HALO", "ARWR", "BEAM", "NTLA", "CRSP", "EDIT",
    "FATE", "SANA", "VERV", "RXRX", "RCKT", "KRYS", "APLS", "IRWD", "PTCT",
    "FOLD", "RARE", "VCYT", "NVAX", "OCGN", "VXRT", "CYTK", "MDGL", "AXSM",
    "ACAD", "SUPN", "HRMY", "PCRX", "COLL", "ITCI", "TEVA", "AMRX", "VTRS",
    "INSM", "AGIO", "BCRX", "ARQT", "IMVT", "TGTX", "DYN", "SWTX", "ARDX",
    "ANAB", "CELC", "OABI", "ELVN", "NUVL", "KYMR", "RVMD", "ZNTL", "ERAS",
    # High-beta tech / software
    "AI", "BBAI", "SOUN", "PATH", "ASAN", "FROG", "PD", "DOMO", "YEXT",
    "BIGC", "FVRR", "UPWK", "ETSY", "PINS", "SNAP", "RBLX", "U", "DKNG",
    "GENI", "OPEN", "RDFN", "COMP", "LMND", "ROOT", "HIPO", "AFRM", "UPST",
    "SOFI", "LC", "OPRT", "ENVA", "EVER", "CARS", "TRUE", "SSTK", "EB",
    "TDOC", "HIMS", "GDRX", "PGNY", "ACCD", "AMWL", "OSCR", "CLOV", "ALHC",
    # Semis / hardware / networking
    "AMBA", "SITM", "POWI", "SMTC", "LSCC", "RMBS", "ALGM", "INDI", "NVTS",
    "AOSL", "UCTT", "ICHR", "ACLS", "COHU", "FORM", "ONTO", "CAMT", "PLAB",
    "WOLF", "AXTI", "LITE", "AAOI", "INFN", "CIEN", "VIAV", "EXTR", "NTGR",
    "DGII", "CALX", "CRUS", "SLAB", "SYNA", "MTSI", "DIOD", "VSH", "MXL",
    # Energy — E&P, services, refiners
    "AR", "RRC", "SWN", "CHRD", "MTDR", "MUR", "SM", "CRC", "CIVI", "PR",
    "VTLE", "TALO", "GPOR", "CRK", "CNX", "NOG", "CPE", "ESTE", "REPX",
    "WTTR", "PUMP", "NINE", "OIS", "RES", "PTEN", "NBR", "HP", "WHD",
    # Clean energy / EV / hydrogen — high beta, catalyst-rich
    "ARRY", "SHLS", "NOVA", "RUN", "ENPH", "SEDG", "CSIQ", "JKS", "DQ",
    "MAXN", "FSLR", "PLUG", "BE", "BLDP", "FCEL", "CHPT", "BLNK", "EVGO",
    "GTLS", "FLNC", "STEM", "AMPS", "NEP",
    # Uranium / nuclear — heavily news-driven
    "UEC", "DNN", "NXE", "UUUU", "LEU", "SMR", "OKLO", "URG", "LTBR", "ASPI",
    # Materials / mining / lithium
    "X", "CLF", "AA", "MP", "TMC", "LAC", "SGML", "PLL", "CENX", "KALU",
    "HL", "CDE", "AG", "EXK", "FSM", "GATO", "USAS", "SVM", "MUX", "AUMN",
    # Crypto miners — among the most frequent EP printers
    "MARA", "RIOT", "CLSK", "HUT", "BITF", "HIVE", "CIFR", "WULF", "IREN",
    "CORZ", "BTBT", "BTDR", "CAN", "SOS", "BTCS", "GREE",
    # Consumer / retail / restaurants
    "GME", "AMC", "KSS", "M", "JWN", "GPS", "ANF", "AEO", "URBN", "BOOT",
    "TLYS", "ZUMZ", "PLCE", "CRI", "HIBB", "ASO", "SPWH", "BGFV",
    "WING", "SHAK", "CAVA", "SG", "PTLO", "FWRG", "BROS", "DNUT", "KRUS",
    "LOCO", "JACK", "WEN", "PZZA", "NDLS", "PBPB", "BJRI", "CBRL", "DENN",
    # Travel / transport / industrials
    "JBLU", "SAVE", "ALGT", "SNCY", "MESA", "SKYW", "HA", "RYAAY",
    "CAR", "HTZ", "LYFT", "WISH", "CHGG", "COUR", "UDMY", "LRN", "ATGE",
    # Financials — regional banks and specialty lenders
    "VLY", "ZION", "CMA", "KEY", "HBAN", "RF", "FITB", "WAL", "BANC",
    "FCFS", "WRLD", "EZPW", "PRAA", "ECPG", "NAVI", "SLM",
    # Misc high-beta / meme-adjacent
    "SPCE", "RKLB", "ASTS", "PL", "BKSY", "RDW", "LUNR", "ACHR", "JOBY",
    "IONQ", "RGTI", "QBTS", "QUBT", "ARQQ",
]

# --------------------------------------------------------------------------- #
# Mega / large cap — regime context and Setup B breakouts
# --------------------------------------------------------------------------- #
MEGA_CAP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO",
    "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN", "CSCO", "IBM",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "UNH", "JNJ", "LLY", "ABBV",
    "MRK", "PFE", "WMT", "COST", "HD", "KO", "PEP", "PG", "DIS", "XOM",
    "CVX", "CAT", "BA", "GE", "HON", "LMT", "RTX", "NEE", "LIN",
]

# Liquid ETFs for regime context.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "SMH", "ARKK", "GLD", "TLT", "XBI", "IBB", "URA", "LIT",
]


def dedupe(seq):
    """Order-preserving dedupe."""
    seen = set()
    out = []
    for t in seq:
        u = t.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# Default: EP hunting ground plus a little context.
DEFAULT_UNIVERSE = dedupe(SMALL_MID + ETFS)

# Everything, when breadth matters more than focus.
FULL_UNIVERSE = dedupe(SMALL_MID + MEGA_CAP + ETFS)
