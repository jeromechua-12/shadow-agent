"""Calibration Agent query functions on fake data, centred on the point-in-time rule."""

import json
import re

import pytest

from shadow.data import queries
from shadow.data.db import init_db
from shadow.data.queries import CalibrationTools, beta_stats
from shadow.tests import fake_db
from shadow.tests.fake_db import AS_OF, VERSION

VISIBLE = {"D1", "D2", "D3a", "D3r", "D4", "D8"}
HIDDEN = {"D5", "D6", "D7"}


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    # Built once: the query functions only read.
    con = init_db(tmp_path_factory.mktemp("q") / "q.duckdb")
    fake_db.build(con)
    yield con
    con.close()


@pytest.fixture
def tools(con):
    return CalibrationTools(con, AS_OF, VERSION)


def all_outputs(tools):
    """Every tool's output, including a read of every visible trade."""
    outs = [tools.cycle_summary(), tools.search_trades(limit=100), tools.subgroup_stats(split_by="month"),
            tools.past_memos(limit=12)]
    outs += [tools.read_trade(d) for d in VISIBLE]
    return outs


# ---- the point-in-time rule ---------------------------------------------------------

def test_future_outcomes_are_never_returned(tools):
    trades = {t["decision_id"] for t in tools.search_trades(limit=100)["trades"]}
    assert trades == VISIBLE
    assert tools.subgroup_stats()["overall"]["n"] == len(VISIBLE)
    assert set(tools.subgroup_stats()["overall"]["decision_ids"]) == VISIBLE
    # D5 was decided before AS_OF but resolves after it; D6 was decided after it.
    for hidden in HIDDEN:
        with pytest.raises(LookupError) as e:
            tools.read_trade(hidden)
        # Same answer as for an id that never existed: nothing leaks about the future.
        assert str(e.value) == f"no resolved trade {hidden!r} as of {AS_OF}"
    text = json.dumps(all_outputs(tools))
    for hidden in HIDDEN:
        assert f'"{hidden}"' not in text
    assert "FUTURE MEMO" not in text


def test_no_returned_date_is_after_as_of(tools):
    """Scan every value of every tool output: no date later than the calibration date."""
    text = json.dumps(all_outputs(tools))
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    assert dates and max(dates) <= AS_OF


@pytest.mark.parametrize("as_of, expected", [
    ("2026-09-15", set()),
    ("2026-09-16", {"D1"}),
    ("2026-09-17", {"D1", "D2", "D3a", "D3r"}),
    ("2026-10-04", VISIBLE),
    ("2026-10-05", VISIBLE | {"D5"}),
    ("2026-10-14", VISIBLE | {"D5", "D6"}),
])
def test_visibility_moves_with_as_of(con, as_of, expected):
    tools = CalibrationTools(con, as_of, VERSION)
    got = tools.search_trades(limit=100)["trades"]
    assert {t["decision_id"] for t in got} == expected
    assert all(t["resolved_on"] <= as_of and t["trade_date"] <= as_of for t in got)


def test_pending_trades_are_counted_not_shown(tools):
    # D5 (decided 09-20, resolves 10-05) is pending at AS_OF; D6 is after AS_OF and not counted.
    assert tools.cycle_summary()["decisions_awaiting_outcome"] == 1


def test_future_memo_and_hypothesis_check_are_ignored(con, tools):
    memos = tools.past_memos()
    assert [m["memo_id"] for m in memos["memos"]] == ["M1"]
    (h1,) = memos["hypotheses"]
    assert (h1["hypothesis_id"], h1["status"], h1["last_checked_in"]) == ("H1", "pending", None)
    assert tools.cycle_summary()["current_memo"] == {"memo_id": "M1", "cycle_date": "2026-08-31", "risk_level": 3}

    later = CalibrationTools(con, "2026-11-01", VERSION).past_memos()
    assert [m["memo_id"] for m in later["memos"]] == ["M2", "M1"]
    assert later["hypotheses"][0]["status"] == "failed"


def test_memo_of_the_cycle_itself_is_not_past(con):
    # A memo dated on the calibration date is the one being written, not a past one.
    assert CalibrationTools(con, "2026-08-31", VERSION).past_memos()["memos"] == []


def test_books_use_snapshots_up_to_as_of(tools):
    books = tools.cycle_summary()["books"]
    assert books["decision"]["value"] == 101.0 and books["shadow"]["value"] == 106.0
    assert books["decision"]["return_this_cycle"] == pytest.approx(0.01)
    assert books["shadow"]["return_this_cycle"] == pytest.approx(0.06)
    assert books["decision_minus_shadow_return_this_cycle"] == pytest.approx(-0.05)


def test_other_experiment_versions_are_separate(con):
    tools = CalibrationTools(con, AS_OF, "rules_template")
    assert [t["decision_id"] for t in tools.search_trades()["trades"]] == ["D7"]


# ---- statistics ---------------------------------------------------------------

def test_cycle_summary_counts(tools):
    s = tools.cycle_summary()
    assert s["this_cycle"]["n"] == 6
    assert s["this_cycle"]["labels"]["noise"] == 1
    assert (s["this_cycle"]["n_scored"], s["this_cycle"]["successes"]) == (5, 3)
    event = s["by_category"]["event_risk"]["this_cycle"]
    assert event["labels"]["missed_gain"] == 1 and event["labels"]["correct_rejection"] == 1
    assert event["thin_evidence"] is True


def test_tool_results_carry_the_cusum_verdict_but_not_its_per_trade_trace(tools):
    assert '"trace"' not in json.dumps(all_outputs(tools))
    trend = tools.cycle_summary()["all_history"]["trend"]
    assert {"flagged", "direction", "flagged_at", "final_s_pos", "final_s_neg"} <= set(trend)


def test_beta_stats():
    even = beta_stats(5, 5)
    assert even["posterior_mean"] == 0.5 and even["p_rate_above_half"] == pytest.approx(0.5, abs=1e-3)
    s = beta_stats(7, 3)
    assert s["posterior_mean"] == pytest.approx(8 / 12, abs=1e-4)
    assert s["ci90"][0] < s["posterior_mean"] < s["ci90"][1]
    assert s["p_rate_above_half"] == pytest.approx(0.8867, abs=2e-3)  # P(Beta(8,4) > 0.5)


def test_subgroup_by_news_keyword_and_split(tools):
    fed = tools.subgroup_stats({"main_reason": "event_risk", "news_keyword": "FED"})
    assert fed["overall"]["decision_ids"] == ["D2"]
    split = tools.subgroup_stats({"main_reason": "event_risk"}, split_by="ticker")
    assert {k: v["n"] for k, v in split["groups"].items()} == {"AAPL": 1, "NVDA": 1}


def test_search_filters_and_order(tools):
    worst = tools.search_trades({"part": "approved_part"}, order="worst")["trades"]
    assert [t["decision_id"] for t in worst] == ["D3a", "D8", "D1"]
    calm = tools.search_trades({"vix": {"max": 16}})["trades"]
    assert {t["decision_id"] for t in calm} == {"D1", "D8"}
    assert tools.search_trades(limit=2)["truncated"] is True


def test_read_trade_contents(tools):
    t = tools.read_trade("D3a")
    assert t["trade"]["label"] == "approval_loss"
    assert [p["decision_id"] for p in t["other_parts_of_same_proposal"]] == ["D3r"]
    assert t["trader_proposal"]["entry_price"] == 100.0
    assert t["conditions"]["vix"] == 22.0
    assert len(t["reports"]["news"]) <= 1501
    assert tools.read_trade("D2")["news_seen"][0]["title"].startswith("Fed")


# ---- input safety ------------------------------------------------------------------

@pytest.mark.parametrize("filters, message", [
    ({"tickr": "NVDA"}, "unknown filter"),
    ({"verdict": "maybe"}, "unknown value"),
    ({"vix": 20}, '"min"'),
    ({"vix": {"min": "20"}}, "must be numbers"),
    ({"trade_date": {"min": "not a date"}}, "Invalid isoformat"),
])
def test_bad_filters_are_rejected(tools, filters, message):
    with pytest.raises(ValueError, match=message):
        tools.search_trades(filters)


def test_injection_text_is_just_a_value(con, tools):
    assert tools.search_trades({"ticker": "NVDA'; DROP TABLE runs; --"})["count"] == 0
    assert tools.subgroup_stats({"news_keyword": "%' OR 1=1 --"})["overall"]["n"] == 0
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == len(fake_db.RUNS)


def test_function_forms(con):
    assert queries.search_trades(con, AS_OF, VERSION, {"label": "noise"})["trades"][0]["decision_id"] == "D8"
    assert queries.cycle_summary(con, AS_OF, VERSION)["as_of"] == AS_OF
