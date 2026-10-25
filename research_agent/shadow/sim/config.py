"""Every number the simulator needs, and the rating-to-position mapping.

The mapping comes from the proposal (Section 3): Buy = 100% of the per-ticker
allocation, Overweight = 50%, Hold = unchanged, Underweight = halve, Sell =
exit, long-only. The money figures are placeholders chosen by the simulator
owner, not decisions the group has made. They live here so changing one is a
single edit and no logic moves.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ratings the Portfolio Manager can emit (proposal, Section 3).
PM_RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")

#: Ratings that set an absolute position, as a multiple of the per-ticker allocation.
TARGET_SIZE = {"Buy": 1.0, "Overweight": 0.5, "Sell": 0.0}

#: Ratings that scale the position currently held.
KEEP_FRACTION = {"Underweight": 0.5}

#: "Hold" places no order: the position carries forward unchanged.

#: The Trader only proposes Buy, Hold or Sell; the shadow book reads them as
#: the same-named ratings (Hold proposals are never logged as decisions).
TRADER_ACTIONS = ("Buy", "Hold", "Sell")


@dataclass(frozen=True)
class SimConfig:
    """One book's trading rules. The same instance is used for both books.

    Using one config for both books is deliberate: the decision and shadow
    books must differ *only* in which rating they follow, or the gap between
    them stops measuring the risk team and starts measuring a difference in
    accounting.
    """

    #: Cash each book starts with. Not stated anywhere in the project
    #: documents; the proposal only says "the same fixed notional cash
    #: balance". Any value works because every reported figure is a ratio.
    starting_cash: float = 100_000.0

    #: Number of tickers in the universe (AAPL, GOOGL, MSFT, NVDA, AMZN,
    #: JPM, XOM, JNJ, TSLA, PG). Used only to derive the allocation.
    universe_size: int = 10

    #: Cost per trade as a fraction of notional, charged on both buys and
    #: sells. 10 bps comes from an earlier proposal draft; the submitted
    #: version says only "the same prices and trading costs".
    transaction_cost: float = 0.0010

    #: Largest position in one ticker, as a multiple of the per-ticker
    #: allocation. The proposal says "per-ticker cap" without a number.
    #: 1.0 means a full-size Buy is the most the book can hold in one name.
    max_position: float = 1.0

    @property
    def per_ticker_allocation(self) -> float:
        """Dollar value of a full-size (size 1.0) position in one ticker."""
        return self.starting_cash / self.universe_size

    def __post_init__(self) -> None:
        problems = []
        if self.starting_cash <= 0:
            problems.append(f"starting_cash must be > 0, got {self.starting_cash!r}")
        if self.universe_size <= 0:
            problems.append(f"universe_size must be > 0, got {self.universe_size!r}")
        if not 0 <= self.transaction_cost < 1:
            problems.append(f"transaction_cost must be in [0, 1), got {self.transaction_cost!r}")
        if self.max_position <= 0:
            problems.append(f"max_position must be > 0, got {self.max_position!r}")
        if problems:
            raise ValueError("invalid SimConfig: " + "; ".join(problems))


DEFAULT = SimConfig()
