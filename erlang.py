"""Erlang C helpers used for contact-centre staffing maths.

Terms
-----
agents (n)   number of agents handling contacts in the interval
load (a)     offered load in Erlangs = contacts * AHT / interval length
AHT          average handle time in seconds
SL           service level = share of contacts answered within `threshold` seconds
ASA          average speed of answer in seconds
"""
import math


def erlang_c(agents: int, load: float) -> float:
    """Probability that a contact has to wait (Erlang C).

    Uses the Erlang B recursion, which stays stable for large numbers of agents.
    """
    if agents <= 0:
        return 1.0
    if agents <= load:
        return 1.0
    b = 1.0
    for k in range(1, agents + 1):
        b = load * b / (k + load * b)
    return b / (1 - (load / agents) * (1 - b))


def service_level(agents: int, load: float, aht: float, threshold: float = 20) -> float:
    """Expected share of contacts answered within `threshold` seconds."""
    if agents <= load:
        return 0.0
    pw = erlang_c(agents, load)
    return 1 - pw * math.exp(-(agents - load) * threshold / aht)


def asa(agents: int, load: float, aht: float) -> float:
    """Expected average speed of answer in seconds (inf when overloaded)."""
    if agents <= load:
        return float("inf")
    return erlang_c(agents, load) * aht / (agents - load)


def occupancy(agents: int, load: float) -> float:
    """Share of logged-in time spent handling contacts (capped at 100%)."""
    if agents <= 0:
        return 1.0
    return min(load / agents, 1.0)


def required_agents(volume: float, aht: float, interval_sec: int = 1800,
                    target_sl: float = 0.80, threshold: float = 20,
                    max_occupancy: float = 0.90) -> int:
    """Smallest number of agents that meets the service-level target
    without pushing occupancy above `max_occupancy`."""
    if volume <= 0:
        return 0
    load = volume * aht / interval_sec
    n = max(1, math.ceil(load))
    while True:
        if (service_level(n, load, aht, threshold) >= target_sl
                and load / n <= max_occupancy):
            return n
        n += 1
