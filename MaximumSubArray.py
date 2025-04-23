# Inputs (example placeholders)
BHP_close_0 = 65.94  # starting close price
BHP_delta = [
    2.017, 2.155, 2.2, 1.471, 0.271, -2.157, 0.666, -0.885, 2.821,
    -3.289, 0.835, 1.346, -0.454, -0.541, -0.582, 0.592, -2.675,
    0.166, -2.535
]  # list of 19 predicted deltas
# or use returns:
# returns = [...]  # list of 19 predicted returns in percent
BP_Close = 34.23
BP_Delta = [
    0.424644828, 0.530625641, 0.530625105, 0.498833001, 0.50176543,
    0.445262969, 0.475804389, 0.520931542, 0.481832981, 0.459449768,
    0.482140481, 0.459218323, 0.484443724, 0.473891497, 0.455157936,
    0.523367047, 0.4835338, 0.530406475, 0.524789333
]

GSK_Close = 42.985
GSK_Delta = [
    0.389790922, 0.465204179, 0.494214833, 0.445074081, 0.475181699,
    0.440859318, 0.429904014, 0.465981245, 0.423900276, 0.417046517,
    0.436304867, 0.405381888, 0.409231633, 0.402049839, 0.472243309,
    0.4047876, 0.434147, 0.445156902, 0.445061028
]

VOD_CLOSE = 9.835
VOD_DELTA = [
    0.464651287, 0.509419262, 0.493182003, 0.496929049, 0.474987119,
    0.471900463, 0.475801468, 0.471812278, 0.472466052, 0.510243177,
    0.495815486, 0.470006675, 0.487554818, 0.471439332, 0.4985829,
    0.486416578, 0.509591758, 0.49278459, 0.488710433
]

def calculate_predictions(close_0, delta):
    # 1. Reconstruct close prices from delta
    close_prices = [close_0]
    for d in delta:
        next_close = close_prices[-1] + d
        close_prices.append(next_close)

    # Now we have 20 close prices, remove the first one
    close_pred = close_prices[1:]

    # 2. Predict open prices (simple assumption: Open[i] = Close[i-1])
    open_pred = close_prices[:-1]  # Day i open = previous day close

    return close_prices, close_pred, open_pred

def max_profit_single_transaction(prices):
    if not prices or len(prices) < 2:
        return 0, -1, -1
    
    max_profit = 0
    min_price_index = 0
    buy_day = sell_day = -1
    
    for i in range(1, len(prices)):
        current_profit = prices[i] - prices[min_price_index]
        if current_profit > max_profit:
            max_profit = current_profit
            buy_day = min_price_index
            sell_day = i
        if prices[i] < prices[min_price_index]:
            min_price_index = i
    
    return max_profit, buy_day, sell_day

def max_profit_multiple_transactions(prices):
    total_profit = 0
    transactions = []
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit = prices[i] - prices[i-1]
            total_profit += profit
            transactions.append((i-1, i, profit))
    
    return total_profit, transactions

def maximum_subarray(arr):
    max_so_far = float('-inf')
    max_ending_here = 0
    start = end = s = 0
    
    for i in range(len(arr)):
        max_ending_here += arr[i]
        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
            start = s
            end = i
        if max_ending_here < 0:
            max_ending_here = 0
            s = i + 1
    
    return max_so_far, start, end

def analyze_stock(name, close_0, delta):
    print(f"\n=== Analysis for {name} ===")
    close_prices, close_pred, open_pred = calculate_predictions(close_0, delta)

    # Output predictions
    print("\nDaily Predictions:")
    for i in range(19):
        print(f"Day {i+1} - Open: {open_pred[i]:.2f}, Close: {close_pred[i]:.2f}")

    # Single transaction analysis
    profit, buy_day, sell_day = max_profit_single_transaction(close_prices)
    if buy_day != -1 and sell_day != -1:
        print(f"\nBest Single Transaction:")
        print(f"Buy on day {buy_day} at {close_prices[buy_day]:.2f}")
        print(f"Sell on day {sell_day} at {close_prices[sell_day]:.2f}")
        print(f"Profit: {profit:.2f} ({(profit/close_prices[buy_day]*100):.2f}%)")

    # Multiple transactions analysis
    multi_profit, transactions = max_profit_multiple_transactions(close_prices)
    print(f"\nMultiple Transactions Strategy:")
    print(f"Total profit: {multi_profit:.2f}")
    if transactions:
        print("Transactions:")
        for buy, sell, p in transactions:
            print(f"Buy on day {buy} at {close_prices[buy]:.2f}, sell on day {sell} at {close_prices[sell]:.2f}, profit: {p:.2f}")

    # Maximum subarray analysis
    price_changes = [close_prices[i+1] - close_prices[i] for i in range(len(close_prices)-1)]
    max_gain, start_idx, end_idx = maximum_subarray(price_changes)
    print(f"\nMaximum Subarray (on price changes):")
    print(f"Buy at day {start_idx} (price: {close_prices[start_idx]:.2f})")
    print(f"Sell at day {end_idx+1} (price: {close_prices[end_idx+1]:.2f})")
    print(f"Maximum profit: {max_gain:.2f}")

# Analyze each stock
analyze_stock("BHP", BHP_close_0, BHP_delta)
analyze_stock("BP", BP_Close, BP_Delta)
analyze_stock("GSK", GSK_Close, GSK_Delta)
analyze_stock("VOD", VOD_CLOSE, VOD_DELTA)
