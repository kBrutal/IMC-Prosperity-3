import pandas as pd
BASE_TREASURE = 10000
OPENING_PERCENTAGE = 0.10  
SECOND_CONTAINER_COST = 50 
def compute_profit(multiplier, inhabitants, is_second=False):
    divisor = inhabitants + OPENING_PERCENTAGE * 100  
    profit = (BASE_TREASURE * multiplier) / divisor
    if is_second:
        profit -= SECOND_CONTAINER_COST
    return profit

if __name__ == "__main__":
    containers = [
        (10, 1),
        (80, 6),
        (37, 3),
        (17, 1),
        (31, 2),
        (50, 4),
        (90, 10),
        (20, 2),
        (73, 4),
        (89, 8)
    ]
    profits = []
    df = pd.DataFrame()
    first_multiplier = []
    second_multiplier = []
    for multiplier, inhabitants in containers: #picking the first one 
        profit = compute_profit(multiplier, inhabitants)
        for multiplier2, inhabitants2 in containers: #picking the second one
            if multiplier2 == multiplier:
                continue
            first_multiplier.append(multiplier)
            second_multiplier.append(multiplier2)
            profit2 = compute_profit(multiplier2, inhabitants2, is_second=True)
            profits.append(profit + profit2)
    df['first_multiplier'] = first_multiplier
    df['second_multiplier'] = second_multiplier
    df['profit'] = profits
    df.sort_values(by='profit', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('The best combination of multipliers is:\n')
    print(df.head(1))