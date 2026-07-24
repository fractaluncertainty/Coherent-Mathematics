import math
import random


class State:
    def __init__(self, w, sigma, chi):
        self.w = w
        self.sigma = sigma
        self.chi = chi


def angle_diff(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def J(x, y):
    return math.exp(
        -angle_diff(x.sigma, y.sigma)
        -angle_diff(x.chi, y.chi)
    )


def I(x, y):
    return J(x, y) * math.cos(angle_diff(x.chi, y.chi))


def persistence(x, states):
    others = [y for y in states if y is not x]

    if not others:
        return 0.0

    return x.w * sum(
        J(x, y) * max(0.0, I(x, y))
        for y in others
    ) / len(others)


def recurse(x):
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
    max_states=120
):
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

        if step % 25 == 0:
            print("step", step, "states", len(states), "avg_p", round(avg_p, 5))

        if not states:
            break

    return states, history


if __name__ == "__main__":
    states, history = run_simulation()

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

    print("\nPairwise strong joins:")
    count = 0

    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue

            join = J(a, b)

            if join > 0.15:
                print(
                    "pair", i, j,
                    "J=", round(join, 4),
                    "I=", round(I(a, b), 4)
                )
                count += 1

    print("Strong joins:", count)
