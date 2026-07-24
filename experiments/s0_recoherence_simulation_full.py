#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 — Primitive Recoherence Simulation with Batch Robustness Test
================================================================

This program is a small experimental simulation prototype inspired by FUT / CoMath.

For readers with no FUT/CoMath background:
------------------------------------------
This code does NOT directly simulate real physics.
It does not simulate actual particles, spacetime, or quantum mechanics.

The research question is much simpler:

    Can stable structure emerge from:
    - continuation,
    - connection capability,
    - interference,
    - and selection?

In plain language:
------------------
We begin with random states.
Each state changes slightly over time.
States that connect well with other states survive more easily.
States with poor connectivity disappear.
Very stable states may generate similar successor states.

This models the FUT/CoMath statement:

    "Coherence is not assumed. Coherence is what survives."

Important:
----------
If stable clusters appear, this is NOT proof of FUT/CoMath.
It is only a first test of whether the core idea of
"persistence through recoherence selection"
is algorithmically viable.

This version includes:
----------------------
1. A single-run simulation.
2. Strong-join detection.
3. A batch robustness test over multiple random seeds.
"""

import math
import random
from statistics import mean


class State:
    """
    A State here is NOT a particle or a point in space.

    It is a simplified "continuation profile".

    Parameters:
    -----------
    w:
        Continuation weight.
        Higher w means the state has greater internal stability/capacity.

    sigma:
        Connection orientation.
        Describes how a state can align or connect with others.

    chi:
        Interference phase.
        States with similar phases may reinforce each other.
        States with opposing phases may weaken each other.
    """

    def __init__(self, w, sigma, chi):
        self.w = w
        self.sigma = sigma
        self.chi = chi


def angle_diff(a, b):
    """Computes the smallest angular distance between two angles on a circle."""
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def J(x, y):
    """
    Connection capability between two states.

    J near 1 means strong compatibility.
    J near 0 means weak compatibility.
    """
    return math.exp(
        -angle_diff(x.sigma, y.sigma)
        -angle_diff(x.chi, y.chi)
    )


def I(x, y):
    """
    Interference between two states.

    I > 0 means constructive recoherence.
    I < 0 means destructive recoherence.
    """
    return J(x, y) * math.cos(angle_diff(x.chi, y.chi))


def persistence(x, states):
    """
    Computes the persistence of a state relative to all other states.

    Persistence = internal weight * average positive recoherence with others.
    """
    others = [y for y in states if y is not x]

    if not others:
        return 0.0

    return x.w * sum(
        J(x, y) * max(0.0, I(x, y))
        for y in others
    ) / len(others)


def recurse(x):
    """
    Generates the next version of a state.

    A state does not remain exactly identical to itself.
    It continues through small variation.
    """
    eta = random.uniform(-0.02, 0.04)
    kappa = random.uniform(0.00, 0.02)
    delta = random.uniform(-0.08, 0.08)
    phi = random.uniform(-0.08, 0.08)

    return State(
        max(0.0, x.w + eta - kappa),
        (x.sigma + delta) % (2 * math.pi),
        (x.chi + phi) % (2 * math.pi)
    )


def split(x):
    """
    Generates a similar successor state.

    The successor is not identical, but also not completely disconnected.
    """
    return State(
        x.w * random.uniform(0.8, 1.05),
        (x.sigma + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        (x.chi + random.uniform(-0.05, 0.05)) % (2 * math.pi)
    )


def run_simulation(
    initial_count=40,
    steps=300,
    delta_min=0.001,
    delta_split=0.08,
    max_states=120,
    verbose=True
):
    """
    Runs the S0 simulation.

    Simulation loop:
    1. All states recurse forward.
    2. Persistence is computed.
    3. Weak states die.
    4. Strong states reproduce.
    5. Statistics are stored.
    """
    states = [
        State(
            random.uniform(0.2, 1.0),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi)
        )
        for _ in range(initial_count)
    ]

    history = []

    for step in range(steps):
        states = [recurse(x) for x in states]

        scored = [(x, persistence(x, states)) for x in states]
        survivors = [x for x, p in scored if p >= delta_min]

        offspring = []
        for x, p in scored:
            if p >= delta_split and len(survivors) + len(offspring) < max_states:
                offspring.append(split(x))

        states = survivors + offspring

        avg_p = sum(p for _, p in scored) / len(scored) if scored else 0.0
        history.append((step, len(states), avg_p))

        if verbose and step % 25 == 0:
            print("step", step, "states", len(states), "avg_p", round(avg_p, 5))

        if not states:
            break

    return states, history


def count_strong_joins(states, threshold=0.15):
    """Counts pairs of states with strong connection capability."""
    count = 0

    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue

            if J(a, b) > threshold:
                count += 1

    return count


def print_strong_joins(states, threshold=0.15):
    """Prints strongly connected state pairs."""
    print("\nPairwise strong joins:")
    count = 0

    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue

            join = J(a, b)

            if join > threshold:
                print(
                    "pair", i, j,
                    "J=", round(join, 4),
                    "I=", round(I(a, b), 4)
                )
                count += 1

    print("Strong joins:", count)


def run_single_demo(seed=0):
    """Runs one visible demonstration simulation."""
    random.seed(seed)

    print("=== Single S0 run ===")
    print("seed:", seed)

    states, history = run_simulation(verbose=True)

    print("\nFinal states:", len(states))

    print("Last 10 history entries:")
    for row in history[-10:]:
        print(row)

    if states:
        print("\nSample surviving states:")
        for s in states[:10]:
            print(
                "w=", round(s.w, 4),
                "sigma=", round(s.sigma, 4),
                "chi=", round(s.chi, 4)
            )

    print_strong_joins(states)
    return states, history


def run_batch(runs=20, threshold=0.15):
    """
    Runs many simulations with different random seeds.

    A single run can be a random accident.
    A batch test checks whether the behavior is robust.
    """
    print("\n=== Batch robustness test ===")
    print("runs:", runs)

    results = []

    for seed in range(runs):
        random.seed(seed)
        states, history = run_simulation(verbose=False)

        final_states = len(states)
        final_avg_p = history[-1][2] if history else 0.0
        strong_joins = count_strong_joins(states, threshold=threshold)

        results.append({
            "seed": seed,
            "final_states": final_states,
            "final_avg_p": final_avg_p,
            "strong_joins": strong_joins,
        })

        print(
            "seed", seed,
            "final_states", final_states,
            "final_avg_p", round(final_avg_p, 5),
            "strong_joins", strong_joins
        )

    final_state_counts = [r["final_states"] for r in results]
    final_p_values = [r["final_avg_p"] for r in results]
    join_counts = [r["strong_joins"] for r in results]

    print("\n=== Batch summary ===")
    print("mean final states:", round(mean(final_state_counts), 3))
    print("mean final avg_p:", round(mean(final_p_values), 6))
    print("mean strong joins:", round(mean(join_counts), 3))
    print("collapsed runs:", sum(1 for x in final_state_counts if x == 0), "of", runs)

    return results


if __name__ == "__main__":
    # Run one detailed demonstration.
    run_single_demo(seed=0)

    # Run a robustness test over many seeds.
    run_batch(runs=20)


"""
How to interpret the results
============================

Final states > 0:
    The system did not completely collapse.

Final states remains moderate:
    The system did not explode chaotically.

avg_p stabilizes:
    A persistence plateau emerged.

Strong joins > 0:
    Surviving states form connection clusters.

Batch test:
    If many random seeds produce similar survival and strong joins,
    then the result is more robust and less likely to be a one-run accident.

Scientific caution:
    This is NOT proof of FUT/CoMath.
    It is only an initial experiment testing whether primitive recoherence
    selection can generate stable structure.

Next useful steps:
    - cluster analysis
    - visualization in the (sigma, chi) plane
    - study the proto-metric g_f = -log(J)
    - parameter sweeps over delta_min, delta_split, drift size, and threshold
"""
