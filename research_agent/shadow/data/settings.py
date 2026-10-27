"""The one pinned configuration every upstream cache run uses."""

from copy import deepcopy

from shadow.data import alpaca_news
from shadow.data.paths import warehouse_dir
from tradingagents.default_config import DEFAULT_CONFIG

# Bump when anything that changes cached outputs changes (prompts, vendors,
# graph shape), so old and new cache files can be told apart.
CACHE_VERSION = 2   # 2: Reddit and StockTwits dropped

ANALYSTS = ("market", "social", "news", "fundamentals")

# Upstream cache runs only (the proposal allows 0 to 0.2). At 0.0 DeepSeek's
# hidden reasoning looped until the token cap and returned empty reports.
UPSTREAM_TEMPERATURE = 0.2
# One retry at this temperature when an agent still loops to the token cap
# and returns nothing; the run records the temperature it used.
RETRY_TEMPERATURE = 0.6

TOOL_VENDORS = {
    **alpaca_news.CONFIG_OVERRIDES["tool_vendors"],
    # Statements by filing date from EDGAR; Yahoo only if EDGAR has nothing.
    "get_balance_sheet": "sec_edgar,yfinance",
    "get_cashflow": "sec_edgar,yfinance",
    "get_income_statement": "sec_edgar,yfinance",
}


def shadow_config(temperature: float = UPSTREAM_TEMPERATURE) -> dict:
    """DEFAULT_CONFIG (with the .env overrides) plus the project's fixed choices."""
    config = deepcopy(DEFAULT_CONFIG)
    config["temperature"] = temperature
    config["tool_vendors"] = {**config.get("tool_vendors", {}), **TOOL_VENDORS}
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["checkpoint_enabled"] = False
    # Upstream outputs must depend only on (ticker, date): no memory log.
    config["memory_log_path"] = None
    # TradingAgentsGraph creates results_dir (nothing is written there by the
    # cache runner); keep it out of ~/.tradingagents and out of the cache folders.
    config["results_dir"] = str(warehouse_dir() / "_tradingagents_results")
    return config


def model_settings(config: dict) -> dict:
    return {
        "provider": config["llm_provider"],
        "deep_think_llm": config["deep_think_llm"],
        "quick_think_llm": config["quick_think_llm"],
        "backend_url": config.get("backend_url"),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
        "llm_max_retries": config.get("llm_max_retries"),
    }
