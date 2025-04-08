import math
from collections import defaultdict

exchangeRates = [
    [1.0, 1.45, 0.52, 0.72],
    [0.7, 1.0, 0.31, 0.48],
    [1.95, 3.1, 1.0, 1.49],
    [1.34, 1.98, 0.64, 1.0]
]

currencies = ["Snowballs", "Pizzas", "Silicon Nuggets", "SeaShells"]
k = 5  
start_currency = 3
n = len(currencies)

exchange_graph = defaultdict(list)
for i in range(n):
    for j in range(n):
        if i != j:
            rate = exchangeRates[i][j]
            exchange_graph[i].append((j, -math.log(rate)))

dp = {}

def rec(start, curr, k_left):
    if k_left == 1:
        for next_curr, log_weight in exchange_graph[curr]:
            if next_curr == start:
                return log_weight
        return float('inf')

    if (curr, k_left) in dp:
        return dp[(curr, k_left)]

    best = float('inf')
    for next_curr, log_weight in exchange_graph[curr]:
        val = log_weight + rec(start, next_curr, k_left - 1)
        if val < best:
            best = val

    dp[(curr, k_left)] = best
    return best

best_profit = rec(start_currency, start_currency, k)

if best_profit == float('inf'):
    print("NO path found starting and ending at SeaShells in 5 trades")
else:
    path = []

    def generate_path(start, curr, k_left):
        if k_left == 1:
            path.append(curr)
            path.append(start)
            return

        best = float('inf')
        next_best = None

        for next_curr, log_weight in exchange_graph[curr]:
            val = log_weight + rec(start, next_curr, k_left - 1)
            if val < best:
                best = val
                next_best = next_curr

        path.append(curr)
        generate_path(start, next_best, k_left - 1)

    generate_path(start_currency, start_currency, k)

    print(" → ".join(currencies[i] for i in path))
    product = 1.0
    for i in range(len(path) - 1):
        from_idx = path[i]
        to_idx = path[i + 1]
        product *= exchangeRates[from_idx][to_idx]

    print(f"Profit multiplier: {product:.6f}x")
