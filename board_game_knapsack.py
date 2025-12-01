import itertools
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class Item:
    """Represents one board game piece."""
    name: str
    cost: int          # printing cost in dollars
    value: int         # “interestingness” score


def generate_items(n: int, seed: int = 0) -> List[Item]:
    """
    Generate n synthetic board game pieces.
    Costs are between 1 and 50 dollars.
    Interestingness values are between 1 and 10.
    A fixed seed makes the experiment reproducible.
    """
    random.seed(seed)
    items = []
    for i in range(n):
        cost = random.randint(1, 50)
        value = random.randint(1, 10)
        items.append(Item(name=f"piece_{i+1}", cost=cost, value=value))
    return items


def brute_force_knapsack(items: List[Item], budget: int) -> Tuple[int, List[int]]:
    best_value = 0
    best_subset: List[int] = []
    n = len(items)

    # Each bitmask encodes a subset
    for mask in range(1 << n):
        total_cost = 0
        total_value = 0
        subset: List[int] = []

        for i in range(n):
            if mask & (1 << i):
                total_cost += items[i].cost
                if total_cost > budget:
                    # Early break if we already exceed the budget
                    break
                total_value += items[i].value
                subset.append(i)
        else:
            # Only update if we did not break out due to over-budget
            if total_value > best_value:
                best_value = total_value
                best_subset = subset

    return best_value, best_subset


def dp_topdown_knapsack(items: List[Item], budget: int) -> Tuple[int, List[int]]:
    from functools import lru_cache
    n = len(items)

    @lru_cache(maxsize=None)
    def helper(i: int, remaining: int) -> int:
        # i = current index, remaining = budget left
        if i == n or remaining <= 0:
            return 0

        # Option 1: skip current item
        skip = helper(i + 1, remaining)

        # Option 2: take current item (if it fits)
        take = 0
        cost = items[i].cost
        value = items[i].value
        if cost <= remaining:
            take = value + helper(i + 1, remaining - cost)

        return max(skip, take)

    def reconstruct(i: int, remaining: int) -> List[int]:
        # Rebuild chosen item indices from the DP table
        if i == n or remaining <= 0:
            return []

        cost = items[i].cost
        value = items[i].value

        best_without = helper(i + 1, remaining)
        best_with = -1
        if cost <= remaining:
            best_with = value + helper(i + 1, remaining - cost)

        if best_with > best_without:
            return [i] + reconstruct(i + 1, remaining - cost)
        else:
            return reconstruct(i + 1, remaining)

    best_value = helper(0, budget)
    chosen_indices = reconstruct(0, budget)
    return best_value, chosen_indices


def greedy_ratio_knapsack(items: List[Item], budget: int) -> Tuple[int, List[int]]:
    sorted_indices = sorted(
        range(len(items)),
        key=lambda i: items[i].value / items[i].cost,
        reverse=True,
    )

    total_cost = 0
    total_value = 0
    chosen: List[int] = []

    for idx in sorted_indices:
        cost = items[idx].cost
        value = items[idx].value

        if total_cost + cost <= budget:
            chosen.append(idx)
            total_cost += cost
            total_value += value

    return total_value, chosen


def measure_algorithm(
    alg_name: str,
    func,
    items: List[Item],
    budget: int,
    repeats: int = 5,
) -> Dict:
    """
    Run one algorithm several times and return average
    time, peak memory, and value.
    """
    times: List[float] = []
    mems: List[int] = []
    best_values: List[int] = []

    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()

        best_value, _ = func(items, budget)

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        mems.append(peak)
        best_values.append(best_value)

    avg_time = sum(times) / len(times)
    avg_mem = sum(mems) / len(mems)
    avg_value = sum(best_values) / len(best_values)

    return {
        "algorithm": alg_name,
        "budget": budget,
        "avg_time_sec": avg_time,
        "avg_memory_bytes": avg_mem,
        "avg_value": avg_value,
    }


def run_experiments():
    sizes = [10, 20, 50, 100]
    budgets = [50, 100, 200, 500]
    repeats = 5

    algorithms = [
        ("BruteForce", brute_force_knapsack),
        ("DP_TopDown", dp_topdown_knapsack),
        ("Greedy_Ratio", greedy_ratio_knapsack),
    ]

    results = []

    for n in sizes:
        # Seed by size for reproducibility
        items = generate_items(n, seed=n)

        for budget in budgets:
            for alg_name, func in algorithms:
                # Keep brute force only for smaller n to avoid extreme runtimes
                if alg_name == "BruteForce" and n > 20:
                    continue

                res = measure_algorithm(
                    alg_name,
                    func,
                    items,
                    budget,
                    repeats=repeats,
                )

                res["n_items"] = n
                results.append(res)

                print(
                    f"n={n}, budget={budget}, alg={alg_name}, "
                    f"time={res['avg_time_sec']:.6f}s, "
                    f"mem={res['avg_memory_bytes']:.0f}B, "
                    f"value={res['avg_value']:.2f}"
                )

    # Save all results to CSV file
    import csv

    fieldnames = [
        "n_items",
        "budget",
        "algorithm",
        "avg_time_sec",
        "avg_memory_bytes",
        "avg_value",
    ]

    with open("knapsack_experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})


if __name__ == "__main__":
    """
    Prototype Board Game Knapsack Experiments

    How to run from command line:

        python board_game_knapsack.py

    This script runs three algorithms (Brute Force,
    Top-Down Dynamic Programming, and Greedy by value-to-cost ratio)
    multiple times, and writes an output file:

        knapsack_experiment_results.csv
    """
    run_experiments()


