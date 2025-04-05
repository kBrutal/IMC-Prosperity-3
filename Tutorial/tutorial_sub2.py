from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np

class Trader:
    def __init__(self):
        # Initialize price history for each product
        self.price_history = {}  # Format: {product: [list of prices]}
        self.max_history_length = 20  # Keep last 20 prices for filtering
        # Define Gaussian kernel parameters
        self.gaussian_std = 3  # Standard deviation for Gaussian filter
        self.gaussian_window = 7  # Window size for Gaussian filter (odd number)

    def _gaussian_filter(self, prices):
        """Apply Gaussian filter to the last few prices"""
        if len(prices) < 3:
            return prices[-1] if prices else 0
        
        # Create Gaussian weights
        x = np.linspace(-self.gaussian_std, self.gaussian_std, self.gaussian_window)
        weights = np.exp(-x**2/2)
        weights /= weights.sum()
        
        # Apply weights to recent prices
        recent_prices = prices[-self.gaussian_window:]
        return np.dot(recent_prices, weights[-len(recent_prices):])

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Calculate mid price
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            
            if best_bid and best_ask:
                current_mid = (best_bid + best_ask) / 2
            elif best_bid:
                current_mid = best_bid
            elif best_ask:
                current_mid = best_ask
            else:
                current_mid = getattr(self, f'last_mid_{product}', 10)
            
            # Update price history
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append(current_mid)
            self.price_history[product] = self.price_history[product][-self.max_history_length:]
            
            # Calculate acceptable price using Gaussian filter
            if len(self.price_history[product]) > 0:
                if product == "RAINFOREST_RESIN":
                    # Conservative approach for RESIN
                    acceptable_price = current_mid
                elif product == "KELP":
                    # Use Gaussian filtered price for KELP
                    acceptable_price = self._gaussian_filter(self.price_history[product])
                else:
                    acceptable_price = current_mid
            else:
                acceptable_price = current_mid
            
            # Store current mid price for future reference
            setattr(self, f'last_mid_{product}', current_mid)
            
            print(f"Acceptable price for {product}: {acceptable_price} (Current mid: {current_mid})")
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            # Trading logic
            if product == "KELP":
                # Buy if best ask is below our acceptable price
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < acceptable_price:
                        print(f"BUY {product} {str(-best_ask_amount)}x {best_ask}")
                        orders.append(Order(product, best_ask, -best_ask_amount))
                
                # Sell if best bid is above our acceptable price
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        print(f"SELL {product} {str(best_bid_amount)}x {best_bid}")
                        orders.append(Order(product, best_bid, -best_bid_amount))
            
            elif product == "RAINFOREST_RESIN":
                # More aggressive strategy for RESIN
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < 9999:
                        print(f"BUY {product} {str(-3*best_ask_amount)}x {best_ask}")
                        orders.append(Order(product, best_ask, -3*best_ask_amount))
                
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if best_bid > 10001:
                        print(f"SELL {product} {str(-3*best_bid_amount)}x {best_bid}")
                        orders.append(Order(product, best_bid, -3*best_bid_amount))
            
            result[product] = orders

        # Sample conversion request
        conversions = 1
        traderData = "SAMPLE"
        return result, conversions, traderData