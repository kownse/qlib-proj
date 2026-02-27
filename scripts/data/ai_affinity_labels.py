"""
Manually curated AI affinity scores for S&P 500 stocks.

Scoring:
    2  = Strong AI beneficiary (direct AI revenue: chips, cloud AI, AI platforms)
    1  = Moderate AI beneficiary (AI infrastructure demand, genuine AI-driven growth)
    0  = AI neutral (traditional businesses, minimal AI impact, or mixed impact)
   -1  = Moderate AI disruption risk (core business partially threatened by AI)
   -2  = Strong AI disruption risk (core business being directly replaced by AI)

Key distinction:
  - Positive scores: Companies that "sell AI" or whose demand grows BECAUSE of AI buildout
  - Negative scores: Companies whose addressable market is shrinking due to AI,
    even if they adopt AI defensively (e.g. adding "AI features" to a threatened product)
  - Using AI in your product does NOT make you an AI beneficiary if AI shrinks your TAM

Validated against 2025 stock performance:
  - Score 2/1 stocks should show AI-driven revenue growth or stock outperformance
  - Score -1/-2 stocks should show valuation compression from AI disruption fears

Unlabeled stocks default to 0 (neutral).
"""

# fmt: off
AI_AFFINITY_SCORES = {
    # =========================================================================
    # === STRONG AI BENEFICIARIES (2) ===
    # Direct AI revenue. Core business IS selling AI products/services.
    # 2025 validation: these stocks significantly outperformed on AI narrative.
    # =========================================================================

    # --- AI Chips / Semiconductor ---
    "NVDA": 2,   # GPU monopoly for AI training/inference
    "AMD": 2,    # MI300 AI accelerators, data center GPUs
    "AVGO": 2,   # Custom AI ASICs (Google TPU, Meta MTIA), networking
    "ARM": 2,    # AI chip architecture licensing (datacenter + mobile)
    "MRVL": 2,   # Custom AI silicon, DPUs, electro-optics
    "TSM": 2,    # Manufactures ALL leading AI chips (TSMC)
    "MU": 2,     # HBM memory critical for AI GPUs; +239% in 2025

    # --- AI Cloud / Hyperscalers ---
    "MSFT": 2,   # Azure AI, OpenAI partnership, Copilot, $80B+ AI capex
    "GOOGL": 2,  # Gemini, DeepMind, TPUs, AI search overhaul
    "GOOG": 2,   # Same as GOOGL
    "AMZN": 2,   # AWS Bedrock, Trainium/Inferentia chips, $100B+ AI capex
    "META": 2,   # LLaMA open-source, AI-driven ads, $65B+ AI capex

    # --- AI Software Platforms ---
    "PLTR": 2,   # AI/ML analytics platform (AIP); +150% in 2025

    # --- AI Data Center Networking ---
    "ANET": 2,   # 400G/800G switches for AI clusters; sole-source for hyperscalers

    # =========================================================================
    # === MODERATE AI BENEFICIARIES (1) ===
    # AI buildout directly drives demand for their products/services.
    # They don't "sell AI" but AI creates NEW demand for what they sell.
    # =========================================================================

    # --- Semiconductor Supply Chain (AI chip demand → their demand) ---
    "QCOM": 1,   # On-device AI (Snapdragon), AI PC chips
    "AMAT": 1,   # Semiconductor equipment for AI chip fabs
    "LRCX": 1,   # Semiconductor equipment (etch/deposition)
    "KLAC": 1,   # Semiconductor inspection/metrology
    "INTC": 1,   # AI PC chips, foundry for AI chip makers
    "TXN": 1,    # Analog chips for AI edge/power delivery
    "SNPS": 1,   # EDA tools for AI chip design; AI-assisted design
    "CDNS": 1,   # EDA tools for AI chip design; AI-assisted design
    "ON": 1,     # Power semiconductors for AI servers
    "MCHP": 1,   # MCUs for AI edge devices
    "ADI": 1,    # Analog/mixed-signal for AI data centers
    "NXPI": 1,   # AI chips for automotive (ADAS, autonomy)
    "MPWR": 1,   # Power management ICs for AI GPU modules
    "TER": 1,    # Semiconductor test equipment (more chips = more testing)
    "KEYS": 1,   # Test & measurement for AI chip validation

    # --- AI Servers / Data Center Hardware ---
    "DELL": 1,   # AI server revenue surging (PowerEdge for AI)
    "SMCI": 1,   # AI-optimized server racks (liquid cooling)
    "VRT": 1,    # Data center power/cooling infrastructure (Vertiv)
    "HPE": 1,    # AI servers (ProLiant, Cray for HPC/AI)
    "JNPR": 1,   # AI data center networking (acquired by HPE)
    "CRWV": 1,   # CoreWeave - GPU cloud for AI workloads
    "NBIS": 1,   # Nebius - AI cloud infrastructure

    # --- Data Storage for AI ---
    "SNDK": 1,   # NAND flash for AI data centers; +559% in 2025 (top SP500)
    "WDC": 1,    # HDD/SSD for AI data storage; +282% in 2025
    "STX": 1,    # HDD storage for AI data centers; tripled in 2025
    "NTAP": 1,   # Enterprise data storage for AI workloads

    # --- Data Center Components ---
    "APH": 1,    # High-speed connectors for AI server clusters
    "GLW": 1,    # Fiber optic cables for AI data center interconnects
    "TEL": 1,    # Connectors for data center / AI infrastructure

    # --- Data Center Power / Utilities ---
    "VST": 1,    # Vistra Energy - nuclear/gas power for data centers
    "CEG": 1,    # Constellation Energy - nuclear PPAs with MSFT for AI
    "NRG": 1,    # NRG Energy - data center power contracts
    "GEV": 1,    # GE Vernova - gas turbines for AI data centers
    "ETN": 1,    # Eaton - power distribution for data centers
    "PWR": 1,    # Quanta Services - builds power infrastructure for DCs

    # --- Data Center REITs ---
    "DLR": 1,    # Digital Realty - data center REIT, record AI leasing
    "EQIX": 1,   # Equinix - data center REIT, AI inference positioning

    # --- AI-Powered Platforms (AI is the core value driver) ---
    "APP": 1,    # AppLovin - AI ad optimization engine; explosive 2025 growth
    "SNOW": 1,   # AI data cloud (Cortex), $200M Anthropic deal; +60% in 2025
    "DDOG": 1,   # Observability for AI infrastructure; growing with cloud AI
    "MDB": 1,    # Vector search/RAG for AI applications
    "NET": 1,    # Workers AI - edge inference platform
    "ORCL": 1,   # OCI AI cloud, GPU clusters for AI training
    "AXON": 1,   # AI for public safety (Draft One); +39% revenue growth

    # --- Cybersecurity (AI increases threat landscape → more demand) ---
    "PANW": 1,   # AI-powered threat detection; AI creates more attacks
    "CRWD": 1,   # AI-driven endpoint protection; AI-era security leader

    # --- AI-Driven Products ---
    "TSLA": 1,   # FSD autonomous driving AI, Dojo supercomputer, Optimus
    "CSCO": 1,   # AI networking infrastructure (Nexus for AI clusters)

    # =========================================================================
    # === AI NEUTRAL (0) ===
    # Explicitly labeled stocks where AI impact is marginal, mixed, or unclear.
    # All unlabeled SP500 stocks also default to 0.
    # =========================================================================

    # --- Tech companies where AI is a feature, not a growth driver ---
    "AAPL": 0,   # Apple Intelligence shipped but hasn't driven iPhone growth
    "FTNT": 0,   # Network security; hardware moat (FortiGate), 500+ AI patents, but faces platform consolidation
    "ANSS": 0,   # Simulation software; acquired by SNPS; AI supplements but doesn't drive revenue
    "FICO": 0,   # Credit scoring; data moat matters more than AI
    "MSCI": 0,   # Financial indexes/analytics; AI impact marginal
    "FFIV": 0,   # Application delivery; not a clear AI play

    # --- Consumer platforms where AI is table stakes ---
    "NFLX": 0,   # Streaming; AI recommendations are table stakes, not differentiator
    "SPOT": 0,   # Music streaming; AI recs are table stakes
    "ABNB": 0,   # Travel marketplace; AI pricing is operational efficiency
    "PYPL": 0,   # Payments; AI fraud detection is operational
    "UBER": 0,   # Ride-hailing; autonomous vehicles are uncertain risk/reward
    "SHOP": 0,   # E-commerce platform; AI features are add-ons
    "BKNG": 0,   # Travel booking; AI search is incremental
    "EBAY": 0,   # Marketplace; AI listing tools are marginal
    "ETSY": 0,   # Marketplace; AI search is incremental

    # --- Payroll/HR (entrenched, slow disruption) ---
    "ADP": 0,    # Payroll giant; deeply entrenched, slow AI disruption
    "PAYX": 0,   # Payroll; similar to ADP, regulatory moat

    # Default 0 for all unlabeled stocks:
    # Consumer Staples: KO, PEP, PG, CL, KHC, COST, WMT, etc.
    # Healthcare: JNJ, UNH, PFE, MRK, LLY, ABBV, TMO, ISRG, etc.
    # Financials: JPM, BAC, GS, V, MA, BLK, SCHW, etc.
    # Industrials: CAT, GE, HON, UNP, DE, RTX, LMT, etc.
    # Energy: XOM, CVX, COP, SLB, etc.
    # REITs (non-DC): AMT, SPG, O, etc.
    # Materials: LIN, APD, FCX, NUE, etc.
    # Telecom: VZ, T, TMUS
    # Utilities (non-DC): NEE, DUK, SO, AEP, etc.

    # =========================================================================
    # === MODERATE AI DISRUPTION RISK (-1) ===
    # Core business partially threatened by AI. These companies may be
    # "adding AI features" but their addressable market is shrinking.
    # 2025-2026 validation: significant stock underperformance, valuation compression.
    # =========================================================================

    # --- SaaS / Enterprise Software (seat-based model directly at risk) ---
    # "If AI agents do the work of 100 people, you need 10 software seats, not 100"
    "CRM": -1,   # Salesforce - AI agents reduce CRM seat count; -38% in 2026 SaaSpocalypse
    "NOW": -1,   # ServiceNow - AI handles IT tickets; -44% from peak
    "ADBE": -1,  # Creative tools threatened by Midjourney/DALL-E/Sora; -43% from peak
    "INTU": -1,  # TurboTax/QuickBooks threatened by AI assistants; -45% from peak
    "WDAY": -1,  # HR/finance workflows automated by AI agents
    "PATH": -1,  # UiPath RPA directly replaced by AI agents
    "ADSK": -1,  # AutoCAD/Revit threatened by AI design generation
    "PAYC": -1,  # Paycom HR/payroll - AI automates HR workflows
    "DAY": -1,   # Dayforce HCM - AI automates HR tasks
    "PTC": -1,   # PLM/CAD software; AI-generated designs threaten
    # NOTE: CDNS/SNPS are BENEFICIARIES (score 1) - AI chip design drives their demand.

    # --- Cybersecurity SaaS (per-seat model + platform consolidation) ---
    "ZS": -1,    # Zscaler - zero trust SaaS per-seat pricing; AI reduces headcount → fewer seats; -28% in Feb 2026 cybersecurity bloodbath

    # --- Web/SMB Services (AI replaces core product) ---
    "GDDY": -1,  # GoDaddy - AI website builders replace web hosting/design; -60% from peak

    # --- IT Services / Consulting (AI reduces billable hours) ---
    "ACN": -1,   # Accenture - consulting being automated by AI coding/agents
    "IBM": -1,   # Legacy IT services; AI investments haven't reversed decline
    "CTSH": -1,  # Cognizant - IT outsourcing; AI deflation in services
    "IT": -1,    # Gartner - research/consulting; AI provides alternatives
    "EPAM": -1,  # Software dev outsourcing; AI coding assistants reduce demand
    "LDOS": -1,  # Leidos - government IT services; AI automates tasks

    # --- Financial Data/Analytics (AI provides cheaper alternatives) ---
    "FDS": -1,   # FactSet - financial data; AI headwinds in 2025
    "GPN": -1,   # Global Payments - payment processing; AI/fintech alternatives

    # --- Creative / Content (generative AI creates content directly) ---
    "PARA": -1,  # Paramount - traditional media; AI content generation
    "WBD": -1,   # Warner Bros Discovery - same as PARA
    "DIS": -1,   # Disney - AI content generation risk for media segment
    "NWSA": -1,  # News Corp - AI replacing journalism
    "FOX": -1,   # Fox Corp - traditional news/media
    "FOXA": -1,  # Fox Corp (class A)
    "IPG": -1,   # Interpublic - ad agency; AI generates creative
    "OMC": -1,   # Omnicom - ad agency; AI automates creative/media buying

    # =========================================================================
    # === STRONG AI DISRUPTION RISK (-2) ===
    # Core business model directly and immediately replaced by AI.
    # =========================================================================
    "CHGG": -2,  # Chegg - homework help replaced by ChatGPT; stock collapsed
}
# fmt: on


def get_ai_affinity(symbol: str) -> int:
    """Get AI affinity score for a stock. Returns 0 (neutral) if not labeled."""
    return AI_AFFINITY_SCORES.get(symbol, 0)


def get_all_labeled_symbols() -> dict:
    """Return the full AI affinity mapping."""
    return dict(AI_AFFINITY_SCORES)
