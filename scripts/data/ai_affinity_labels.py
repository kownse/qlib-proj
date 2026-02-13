"""
Manually curated AI affinity scores for stocks.

Scoring:
    2  = Strong AI beneficiary (direct AI revenue: chips, cloud AI, AI platforms)
    1  = Moderate AI beneficiary (AI-adjacent infrastructure, data centers, AI adoption upside)
    0  = AI neutral (traditional businesses, minimal AI impact)
   -1  = Moderate AI disruption risk (traditional IT services, basic SaaS, some media)
   -2  = Strong AI disruption risk (core business being directly replaced by AI)

Unlabeled stocks default to 0 (neutral).
"""

# fmt: off
AI_AFFINITY_SCORES = {
    # === Strong AI Beneficiaries (2) ===
    # AI Chips / Semiconductor
    "NVDA": 2,   # GPU monopoly for AI training/inference
    "AMD": 2,    # MI300 AI accelerators
    "AVGO": 2,   # Custom AI chips (Google TPU), networking
    "ARM": 2,    # AI chip architecture (mobile + datacenter)
    "MRVL": 2,   # Custom AI silicon, DPUs
    "TSM": 2,    # Manufactures all AI chips (TSMC)
    # AI Cloud / Hyperscalers
    "MSFT": 2,   # Azure AI, OpenAI partnership, Copilot
    "GOOGL": 2,  # Gemini, DeepMind, TPUs, search AI
    "GOOG": 2,   # Same as GOOGL
    "AMZN": 2,   # AWS AI/ML, Bedrock, Trainium chips
    "META": 2,   # LLaMA, AI-driven ads, Reality Labs
    # AI Software Platforms
    "PLTR": 2,   # AI/ML analytics platform (AIP)
    "SNOW": 2,   # AI data cloud, Cortex
    "CRM": 2,    # Einstein AI, Agentforce
    "NOW": 2,    # Now Assist AI, workflow automation
    "ADBE": 2,   # Firefly generative AI, Sensei
    # AI Infrastructure / Networking
    "ANET": 2,   # AI data center networking (400G/800G)

    # === Moderate AI Beneficiaries (1) ===
    # AI-Adjacent Semiconductor
    "QCOM": 1,   # On-device AI, Snapdragon
    "MU": 1,     # HBM memory for AI chips
    "AMAT": 1,   # Semiconductor equipment for AI chip fabs
    "LRCX": 1,   # Semiconductor equipment
    "KLAC": 1,   # Semiconductor equipment
    "INTC": 1,   # AI PC chips, foundry aspirations
    "TXN": 1,    # Analog chips, AI edge devices
    "SNPS": 1,   # EDA for AI chip design
    "CDNS": 1,   # EDA for AI chip design
    "ON": 1,     # Power management for AI
    "MCHP": 1,   # Microcontrollers for AI edge
    "ADI": 1,    # Analog/mixed-signal for AI
    "NXPI": 1,   # Auto AI chips
    # AI Infrastructure / Servers
    "DELL": 1,   # AI server sales
    "SMCI": 1,   # AI server racks (Super Micro)
    "VRT": 1,    # Data center power/cooling for AI
    "APP": 1,    # AI-driven mobile advertising
    "CRWV": 1,   # CoreWeave AI cloud
    "NBIS": 1,   # Nebius AI cloud
    # Cloud/Data/SaaS Benefiting from AI
    "ORCL": 1,   # OCI AI cloud, database AI
    "WDAY": 1,   # AI in HR/finance workflows
    "DDOG": 1,   # AI observability
    "MDB": 1,    # Vector search for AI apps
    "PATH": 1,   # AI + RPA automation
    "NET": 1,    # AI inference at the edge (Workers AI)
    "PANW": 1,   # AI-powered cybersecurity
    "CRWD": 1,   # AI-driven threat detection
    "ZS": 1,     # Zero trust + AI security
    # AI-Adjacent Tech
    "NFLX": 1,   # AI recommendation engine
    "UBER": 1,   # AI routing, autonomous vehicle option
    "SHOP": 1,   # AI for merchants (Shopify Magic)
    "SPOT": 1,   # AI recommendations, DJ AI
    "ABNB": 1,   # AI for pricing/matching
    "PYPL": 1,   # AI fraud detection
    # Utilities Powering Data Centers
    "VST": 1,    # Vistra Energy - data center power
    "CEG": 1,    # Constellation Energy - nuclear for data centers
    "NRG": 1,    # NRG Energy - data center power
    "TSLA": 1,   # FSD AI, Dojo, Optimus robot
    # Enterprise Software AI Adopters
    "INTU": 1,   # TurboTax AI, QuickBooks AI
    "ANSS": 1,   # AI simulation
    "FTNT": 1,   # AI-driven network security
    "FICO": 1,   # AI credit scoring
    "MSCI": 1,   # AI in financial analytics
    "CSCO": 1,   # AI networking

    # === AI Neutral (0) ===
    # Default for most traditional companies. Only listing exceptions below.
    # Consumer Staples: KO, PEP, PG, CL, KHC, etc. → 0
    # Healthcare: JNJ, UNH, PFE, MRK, LLY, etc. → 0
    # Financials: JPM, BAC, GS, V, MA, etc. → 0
    # Industrials: CAT, GE, HON, UNP, etc. → 0
    # Energy: XOM, CVX, COP, etc. → 0
    # REITs: AMT, SPG, etc. → 0
    # Materials: LIN, APD, etc. → 0
    # Communication (non-AI): VZ, T, TMUS → 0

    # === Moderate AI Disruption Risk (-1) ===
    # Traditional IT Consulting / Services
    "ACN": -1,   # Consulting being automated by AI
    "IBM": -1,   # Legacy IT services, though investing in AI
    # Traditional Software Being Disrupted
    "CTSH": -1,  # Cognizant - IT outsourcing
    "IT": -1,    # Gartner - research/consulting
    "EPAM": -1,  # Software development outsourcing
    "GPN": -1,   # Payment processing (AI alternatives)
    "FSLR": -1,  # (Included for illustration; solar not AI-disrupted per se, but example)
    # Media/Content at Risk
    "PARA": -1,  # Traditional media disrupted by AI content
    "WBD": -1,   # Warner Bros Discovery
    "DIS": -1,   # Disney - AI content generation risk
    "NWSA": -1,  # News Corp - AI replacing journalism

    # === Strong AI Disruption Risk (-2) ===
    # Companies whose core business model is most directly threatened
    "CHGG": -2,  # Chegg - homework help replaced by ChatGPT
}
# fmt: on


def get_ai_affinity(symbol: str) -> int:
    """Get AI affinity score for a stock. Returns 0 (neutral) if not labeled."""
    return AI_AFFINITY_SCORES.get(symbol, 0)


def get_all_labeled_symbols() -> dict:
    """Return the full AI affinity mapping."""
    return dict(AI_AFFINITY_SCORES)
