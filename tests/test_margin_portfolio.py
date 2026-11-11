"""The registered margin verdict must stay falsifiable once it is code.

The rule is four conjuncts; the test pins each one so a later change that flips a
comparison cannot quietly turn a failure into a pass.
"""
from __future__ import annotations

import pytest

from tools.margin_portfolio import verdict_for


def control(median=-0.02, p90=0.03, strategy=-0.01):
    return {"V": dict(random_median=median, random_p90=p90, strategy=strategy)}


def window(net=0.02, t=3.0):
    return dict(net_excess=net, t=t)


def test_a_clean_pass_requires_all_four_conjuncts():
    assert verdict_for(window(net=0.05, t=3.0), control())["passed"] is True


def test_negative_excess_fails_even_with_a_large_t():
    verdict = verdict_for(window(net=-0.01, t=9.0), control())
    assert verdict["passed"] is False and verdict["t_ok"] is True


def test_small_t_fails_even_with_positive_excess():
    assert verdict_for(window(net=0.05, t=2.4), control())["passed"] is False


def test_inside_the_random_distribution_fails():
    verdict = verdict_for(window(net=0.02, t=3.0), control(median=0.03, p90=0.04))
    assert verdict["above_random_median"] is False and verdict["passed"] is False


def test_passing_the_median_but_not_p90_fails():
    verdict = verdict_for(window(net=0.035, t=3.0), control(median=0.01, p90=0.05))
    assert verdict["above_random_median"] is True
    assert verdict["above_p90"] is False
    assert verdict["passed"] is False


def test_missing_random_values_are_not_silently_a_pass():
    verdict = verdict_for(window(net=0.05, t=3.0), control(median=None, p90=None))
    assert verdict["passed"] is False
