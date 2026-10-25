"""One book: cash, positions, and the rules for moving between them.

Pure Python. No database, no network, no dates beyond what the caller passes
in, so the whole trading rule set can be tested without a warehouse or a
price feed.

The decision book and the shadow book are the same object with the same
config; they differ only in which rating the caller turns into orders (the
Portfolio Manager's rating for the decision book, the Trader's proposed action
for the shadow book). That is the whole of the difference between the two
books, which is what lets their divergence be read as the risk team's effect.

Order of operations in one trading day, matching the proposal's "trades fill
at the next trading day's open; positions are marked daily to close":

1. ``apply_fills`` at that day's OPEN, for decisions made before that day.
2. ``value`` at that day's CLOSE, giving the value written to the snapshot.

Sells settle before buys, so cash freed by an exit is available to the same
day's entries. If the buys still want more cash than the book holds, every
buy that day is scaled by the same factor rather than filling some orders
and dropping others, which would make the fill order significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shadow.sim.config import DEFAULT, KEEP_FRACTION, PM_RATINGS, TARGET_SIZE, SimConfig


@dataclass(frozen=True)
class Order:
    """Where the book should move one ticker on one fill. Set exactly one field.

    ``target_size``: an absolute position, as a multiple of the per-ticker
    allocation (Buy 1.0, Overweight 0.5, Sell 0.0). Long-only, so never negative.
    ``keep_fraction``: keep this share of what is held now (Underweight 0.5).
    """

    ticker: str
    target_size: float | None = None
    keep_fraction: float | None = None

    def __post_init__(self) -> None:
        if (self.target_size is None) == (self.keep_fraction is None):
            raise ValueError(f"{self.ticker}: an order sets exactly one of target_size and keep_fraction")
        if self.keep_fraction is not None and not 0 <= self.keep_fraction <= 1:
            raise ValueError(f"{self.ticker}: keep_fraction must be in [0, 1], got {self.keep_fraction!r}")


def order_for_rating(ticker: str, rating: str) -> Order | None:
    """The order a five-tier rating places; None for Hold (the position carries forward)."""
    if rating not in PM_RATINGS:
        raise ValueError(f"{ticker}: unknown rating {rating!r}; expected one of {PM_RATINGS}")
    if rating in TARGET_SIZE:
        return Order(ticker, target_size=TARGET_SIZE[rating])
    if rating in KEEP_FRACTION:
        return Order(ticker, keep_fraction=KEEP_FRACTION[rating])
    return None


@dataclass(frozen=True)
class Fill:
    """What actually happened, after capping, cash scaling and costs."""

    ticker: str
    shares: float   # signed: positive bought, negative sold
    price: float
    cost: float     # transaction cost paid, always >= 0


@dataclass
class Book:
    """Cash plus share counts, moved by orders and valued at a price map."""

    config: SimConfig = DEFAULT
    cash: float = None  # type: ignore[assignment]  # defaults to config.starting_cash
    positions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.config.starting_cash

    # -- valuation ------------------------------------------------------------

    def value(self, prices: dict[str, float]) -> float:
        """Cash plus every position marked at ``prices``.

        Every held ticker must have a price: silently skipping one would
        quietly shrink the book and show up later as a fake drawdown.
        """
        missing = [t for t in self.positions if t not in prices]
        if missing:
            raise KeyError(f"no price to mark {', '.join(sorted(missing))}")
        return self.cash + sum(sh * prices[t] for t, sh in self.positions.items())

    # -- trading --------------------------------------------------------------

    def _target_shares(self, order: Order, price: float) -> float:
        if order.keep_fraction is not None:
            return self.positions.get(order.ticker, 0.0) * order.keep_fraction
        size = min(max(order.target_size, 0.0), self.config.max_position)
        return size * self.config.per_ticker_allocation / price

    def apply_fills(self, orders: list[Order], prices: dict[str, float]) -> list[Fill]:
        """Move to the targets in ``orders`` at ``prices``; return what filled.

        Tickers with no order keep their current position, which is how a
        Hold carries forward: a Hold rating places no order, and the decision
        logger writes no row for a Hold proposal.
        """
        if not orders:
            return []
        seen = set()
        for o in orders:
            if o.ticker in seen:
                raise ValueError(f"two orders for {o.ticker} on one fill")
            seen.add(o.ticker)
        missing = [o.ticker for o in orders if o.ticker not in prices]
        if missing:
            raise KeyError(f"no fill price for {', '.join(sorted(missing))}")

        deltas: dict[str, float] = {}
        for o in orders:
            price = prices[o.ticker]
            if price <= 0:
                raise ValueError(f"{o.ticker}: fill price must be > 0, got {price!r}")
            deltas[o.ticker] = self._target_shares(o, price) - self.positions.get(o.ticker, 0.0)

        fee = self.config.transaction_cost
        fills: list[Fill] = []

        # Sells first: they free cash the same day's buys may need.
        for ticker, delta in deltas.items():
            if delta >= 0:
                continue
            price = prices[ticker]
            shares = max(delta, -self.positions.get(ticker, 0.0))  # never go short
            if shares == 0:
                continue
            gross = -shares * price
            cost = gross * fee
            self.cash += gross - cost
            self._move(ticker, shares)
            fills.append(Fill(ticker, shares, price, cost))

        # Buys, scaled together if the book cannot afford all of them.
        wanted = {t: d for t, d in deltas.items() if d > 0}
        if wanted:
            need = sum(d * prices[t] * (1 + fee) for t, d in wanted.items())
            scale = 1.0 if need <= self.cash else (self.cash / need if need > 0 else 0.0)
            for ticker, delta in wanted.items():
                shares = delta * scale
                if shares <= 0:
                    continue
                price = prices[ticker]
                gross = shares * price
                cost = gross * fee
                self.cash -= gross + cost
                self._move(ticker, shares)
                fills.append(Fill(ticker, shares, price, cost))

        # Float error can leave cash at -1e-13; clamp rather than fail the
        # snapshot's "cash >= 0" check on a rounding artefact.
        if -1e-6 < self.cash < 0:
            self.cash = 0.0
        return fills

    def _move(self, ticker: str, shares: float) -> None:
        held = self.positions.get(ticker, 0.0) + shares
        if abs(held) < 1e-9:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = held
