"""Trading rules for one book. Pure: no database, no prices fetched."""

from __future__ import annotations

import pytest

from shadow.sim.book import Book, Order, order_for_rating
from shadow.sim.config import SimConfig

# 10 tickers, $100k, so a full-size position is $10,000.
CONFIG = SimConfig(starting_cash=100_000.0, universe_size=10, transaction_cost=0.0010)
FREE = SimConfig(starting_cash=100_000.0, universe_size=10, transaction_cost=0.0)


def test_starts_flat_with_full_cash():
    book = Book(config=CONFIG)
    assert book.cash == 100_000.0
    assert book.positions == {}
    assert book.value({}) == 100_000.0


def test_full_size_buy_takes_one_allocation():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    assert book.positions["AAPL"] == pytest.approx(100.0)   # $10,000 / $100
    assert book.cash == pytest.approx(90_000.0)
    assert book.value({"AAPL": 100.0}) == pytest.approx(100_000.0)


def test_half_size_buy_takes_half_an_allocation():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 0.5)], {"AAPL": 100.0})
    assert book.positions["AAPL"] == pytest.approx(50.0)
    assert book.cash == pytest.approx(95_000.0)


def test_transaction_cost_is_charged_on_buys_and_sells():
    book = Book(config=CONFIG)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    assert book.cash == pytest.approx(100_000.0 - 10_000.0 - 10.0)   # 10 bps of $10k
    book.apply_fills([Order("AAPL", 0.0)], {"AAPL": 100.0})
    assert book.positions == {}
    assert book.cash == pytest.approx(100_000.0 - 10.0 - 10.0)       # charged again on exit


def test_target_zero_exits_and_never_goes_short():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    fills = book.apply_fills([Order("AAPL", 0.0)], {"AAPL": 110.0})
    assert book.positions == {}
    assert fills[0].shares == pytest.approx(-100.0)
    # Selling with nothing held does nothing at all.
    assert book.apply_fills([Order("AAPL", 0.0)], {"AAPL": 110.0}) == []
    assert book.positions == {}


def test_order_moves_to_a_target_not_by_an_amount():
    """A second Buy at the same size is a no-op, not a doubling."""
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    assert book.positions["AAPL"] == pytest.approx(100.0)


def test_rebalances_down_when_the_target_shrinks():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    book.apply_fills([Order("AAPL", 0.5)], {"AAPL": 100.0})
    assert book.positions["AAPL"] == pytest.approx(50.0)
    assert book.cash == pytest.approx(95_000.0)


def test_keep_fraction_halves_what_is_held_whatever_the_price():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    book.apply_fills([Order("AAPL", keep_fraction=0.5)], {"AAPL": 150.0})
    assert book.positions["AAPL"] == pytest.approx(50.0)
    assert book.cash == pytest.approx(90_000.0 + 50 * 150.0)


def test_keep_fraction_with_nothing_held_does_nothing():
    book = Book(config=FREE)
    assert book.apply_fills([Order("AAPL", keep_fraction=0.5)], {"AAPL": 100.0}) == []
    assert book.positions == {}


def test_an_order_sets_exactly_one_kind_of_target():
    with pytest.raises(ValueError, match="exactly one"):
        Order("AAPL")
    with pytest.raises(ValueError, match="exactly one"):
        Order("AAPL", target_size=1.0, keep_fraction=0.5)
    with pytest.raises(ValueError, match="keep_fraction"):
        Order("AAPL", keep_fraction=1.5)


@pytest.mark.parametrize("rating, target, keep", [
    ("Buy", 1.0, None), ("Overweight", 0.5, None), ("Sell", 0.0, None), ("Underweight", None, 0.5),
])
def test_ratings_map_to_the_proposals_positions(rating, target, keep):
    order = order_for_rating("AAPL", rating)
    assert (order.target_size, order.keep_fraction) == (target, keep)


def test_hold_places_no_order():
    assert order_for_rating("AAPL", "Hold") is None


def test_unknown_rating_is_rejected():
    with pytest.raises(ValueError, match="unknown rating"):
        order_for_rating("AAPL", "Strong Buy")


def test_untouched_tickers_carry_forward():
    """A ticker with no order (a Hold) keeps its position."""
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0), Order("MSFT", 1.0)], {"AAPL": 100.0, "MSFT": 50.0})
    book.apply_fills([Order("AAPL", 0.0)], {"AAPL": 100.0})
    assert "MSFT" in book.positions
    assert book.positions["MSFT"] == pytest.approx(200.0)


def test_position_cap_limits_a_single_name():
    capped = SimConfig(starting_cash=100_000.0, universe_size=10, transaction_cost=0.0, max_position=1.0)
    book = Book(config=capped)
    book.apply_fills([Order("AAPL", 3.0)], {"AAPL": 100.0})   # asks for 3x
    assert book.positions["AAPL"] == pytest.approx(100.0)     # gets 1x


def test_sells_settle_before_buys_so_freed_cash_is_usable():
    poor = SimConfig(starting_cash=10_000.0, universe_size=1, transaction_cost=0.0)
    book = Book(config=poor, cash=0.0, positions={"AAPL": 100.0})   # fully invested
    book.apply_fills([Order("AAPL", 0.0), Order("MSFT", 1.0)], {"AAPL": 100.0, "MSFT": 100.0})
    assert book.positions == {"MSFT": pytest.approx(100.0)}
    assert book.cash == pytest.approx(0.0)


def test_buys_scale_together_when_cash_runs_short():
    """Every buy is scaled by one factor, so fill order does not matter."""
    poor = SimConfig(starting_cash=10_000.0, universe_size=1, transaction_cost=0.0)
    book = Book(config=poor, cash=5_000.0)   # wants $20k of stock, holds $5k
    book.apply_fills([Order("AAPL", 1.0), Order("MSFT", 1.0)], {"AAPL": 100.0, "MSFT": 100.0})
    assert book.positions["AAPL"] == pytest.approx(25.0)   # half of 50 shares
    assert book.positions["MSFT"] == pytest.approx(25.0)
    assert book.cash == pytest.approx(0.0)


def test_cash_never_goes_negative():
    poor = SimConfig(starting_cash=1_000.0, universe_size=1, transaction_cost=0.0010)
    book = Book(config=poor)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 7.0})
    assert book.cash >= 0.0


def test_value_marks_positions_at_the_given_prices():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    assert book.value({"AAPL": 120.0}) == pytest.approx(90_000.0 + 12_000.0)


def test_value_refuses_to_mark_a_position_with_no_price():
    book = Book(config=FREE)
    book.apply_fills([Order("AAPL", 1.0)], {"AAPL": 100.0})
    with pytest.raises(KeyError, match="AAPL"):
        book.value({})


def test_fill_without_a_price_is_an_error_not_a_skip():
    book = Book(config=FREE)
    with pytest.raises(KeyError, match="AAPL"):
        book.apply_fills([Order("AAPL", 1.0)], {})


def test_two_orders_for_one_ticker_is_rejected():
    book = Book(config=FREE)
    with pytest.raises(ValueError, match="two orders"):
        book.apply_fills([Order("AAPL", 1.0), Order("AAPL", 0.5)], {"AAPL": 100.0})


def test_both_books_share_one_config():
    """The books differ only in the orders they are given, never in accounting."""
    decision, shadow = Book(config=CONFIG), Book(config=CONFIG)
    decision.apply_fills([order_for_rating("AAPL", "Overweight")], {"AAPL": 100.0})   # reduced
    shadow.apply_fills([order_for_rating("AAPL", "Buy")], {"AAPL": 100.0})            # full proposal
    assert shadow.positions["AAPL"] == pytest.approx(2 * decision.positions["AAPL"])
    assert decision.value({"AAPL": 100.0}) == pytest.approx(100_000.0 - 5.0)
    assert shadow.value({"AAPL": 100.0}) == pytest.approx(100_000.0 - 10.0)


def test_bad_config_is_rejected():
    with pytest.raises(ValueError, match="transaction_cost"):
        SimConfig(transaction_cost=1.5)
    with pytest.raises(ValueError, match="starting_cash"):
        SimConfig(starting_cash=0.0)
