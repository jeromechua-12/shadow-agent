"""CUSUM trend detection: pure function tests, independent of the database and
the two-book simulator. Rows are {"label", "resolved_on", "decision_id"} dicts,
the same shape ``group_stats`` passes in from the query functions.
"""

from datetime import date, timedelta

import pytest

from shadow.data.queries import CUSUM_H, cusum_trend, group_stats


def _rows(labels: list[str], start: str = "2026-07-01") -> list[dict]:
    day = date.fromisoformat(start)
    rows = []
    for i, label in enumerate(labels):
        rows.append({"decision_id": f"d{i}", "resolved_on": (day + timedelta(days=i)).isoformat(),
                     "label": label, "part": "approved_part", "score": 0.0, "trade_date": start})
    return rows


SUCCESS, FAILURE = "correct_approval", "approval_loss"


def test_stable_rate_does_not_flag():
    # Alternating success/failure: a constant ~50% rate, no drift.
    rows = _rows([SUCCESS, FAILURE] * 25)
    trend = cusum_trend(rows)
    assert trend["n"] == 50
    assert trend["flagged"] is False
    assert trend["direction"] is None


BASELINE = [SUCCESS, FAILURE] * 10  # 20 points, steady ~50% rate: never drifts on its own


def test_shift_from_baseline_to_all_success_flags_up():
    # A steady baseline, then a clear step up to all successes partway through.
    rows = _rows(BASELINE + [SUCCESS] * 15)
    trend = cusum_trend(rows, target=0.5)
    assert trend["flagged"] is True
    assert trend["direction"] == "up"
    # Flags during the shifted segment, not the steady baseline before it.
    flagged_index = next(i for i, r in enumerate(rows) if r["decision_id"] == trend["flagged_at"]["decision_id"])
    assert flagged_index >= len(BASELINE)


def test_shift_from_baseline_to_all_failure_flags_down():
    rows = _rows(BASELINE + [FAILURE] * 15)
    trend = cusum_trend(rows, target=0.5)
    assert trend["flagged"] is True
    assert trend["direction"] == "down"
    flagged_index = next(i for i, r in enumerate(rows) if r["decision_id"] == trend["flagged_at"]["decision_id"])
    assert flagged_index >= len(BASELINE)


def test_lopsided_split_flags_in_the_direction_of_the_first_segment():
    # With no steady baseline, an immediate run of one label is itself a deviation
    # from the 0.5 target and is flagged first, before any later shift back.
    rows = _rows([FAILURE] * 15 + [SUCCESS] * 15)
    trend = cusum_trend(rows, target=0.5)
    assert trend["flagged"] is True
    assert trend["direction"] == "down"
    flagged_index = next(i for i, r in enumerate(rows) if r["decision_id"] == trend["flagged_at"]["decision_id"])
    assert flagged_index < 15


def test_thin_evidence_below_ten_scored():
    rows = _rows([SUCCESS] * 4 + [FAILURE] * 3)
    trend = cusum_trend(rows)
    assert trend["n"] == 7
    assert trend["thin_evidence"] is True


def test_noise_labels_are_excluded_like_group_stats():
    rows = _rows([SUCCESS, "noise", FAILURE, "noise"] * 10)
    trend = cusum_trend(rows)
    assert trend["n"] == 20  # noise rows (half of the 40) are dropped


def test_all_same_label_has_no_variance_and_never_flags():
    # target defaults to 1.0 here: sigma is undefined (0 * (1-0)), z is always 0.
    rows = _rows([SUCCESS] * 20)
    trend = cusum_trend(rows)
    assert trend["target"] == 1.0
    assert trend["flagged"] is False
    assert trend["final_s_pos"] == 0.0 and trend["final_s_neg"] == 0.0


def test_empty_rows():
    trend = cusum_trend([])
    assert trend == {
        "n": 0, "target": 0.5, "k": 0.5, "h": CUSUM_H, "flagged": False, "direction": None,
        "flagged_at": None, "final_s_pos": 0.0, "final_s_neg": 0.0, "thin_evidence": True, "trace": [],
    }


def test_trace_is_chronological_and_matches_final_values():
    rows = _rows([FAILURE] * 10 + [SUCCESS] * 10)
    trend = cusum_trend(rows, target=0.5)
    dates = [t["resolved_on"] for t in trend["trace"]]
    assert dates == sorted(dates)
    assert trend["trace"][-1]["s_pos"] == trend["final_s_pos"]
    assert trend["trace"][-1]["s_neg"] == trend["final_s_neg"]


def test_explicit_target_overrides_self_referential_rate():
    # All successes, but held against a demanding target of 0.9: still some drift room.
    rows = _rows([SUCCESS] * 20)
    trend = cusum_trend(rows, target=0.9)
    assert trend["target"] == 0.9
    assert trend["final_s_pos"] >= 0.0  # a constant positive z accumulates upward


def test_group_stats_embeds_trend():
    # group_stats calls cusum_trend with its default (self-referential) target, so
    # this only checks the wiring/shape; direction is covered with an explicit
    # target in the tests above.
    rows = _rows(BASELINE + [SUCCESS] * 15)
    stats = group_stats(rows)
    assert stats["trend"]["n"] == stats["n_scored"]
    assert isinstance(stats["trend"]["flagged"], bool)
    assert stats["trend"]["target"] == stats["success_rate"]


@pytest.mark.parametrize("h", [1.0, 8.0])
def test_higher_threshold_is_harder_to_flag(h):
    rows = _rows([FAILURE] * 6 + [SUCCESS] * 6)
    trend = cusum_trend(rows, target=0.5, h=h)
    if h == 8.0:
        assert trend["flagged"] is False
    else:
        assert trend["flagged"] is True
