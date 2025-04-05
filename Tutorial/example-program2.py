from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Calculate acceptable price based on market data
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                # Calculate mid price (average of best bid and best ask)
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Calculate acceptable price with a small discount/premium
                # For example, willing to buy at 0.5% below mid price and sell at 0.5% above mid price
                if product == "RAINFOREST_RESIN":  # Different strategy for different products
                    acceptable_price = mid_price  # Conservative approach for RAINFOREST_RESIN
                elif product == "KELP":
                    # More aggressive approach for kelp - using weighted average of recent prices
                    if hasattr(self, 'price_history') and product in self.price_history:
                        # Use exponential moving average if we have history
                        alpha = 0.7  # Weight for current price vs historical
                        acceptable_price = alpha * mid_price + (1-alpha) * self.price_history[product]
                    else:
                        acceptable_price = mid_price
                else:
                    acceptable_price = mid_price
            elif len(order_depth.buy_orders) > 0:
                # Only buy orders exist, set acceptable price slightly below best bid
                best_bid = max(order_depth.buy_orders.keys())
                acceptable_price = best_bid * 0.99
            elif len(order_depth.sell_orders) > 0:
                # Only sell orders exist, set acceptable price slightly above best ask
                best_ask = min(order_depth.sell_orders.keys())
                acceptable_price = best_ask * 1.01
            else:
                # No orders in the book, use previous acceptable price or default
                acceptable_price = getattr(self, f'last_acceptable_price_{product}', 10)
            
            # Store this acceptable price for future reference
            setattr(self, f'last_acceptable_price_{product}', acceptable_price)
            
            print(f"Acceptable price for {product}: {acceptable_price}")
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            # Rest of the trading logic using the calculated acceptable_price
            if (len(order_depth.sell_orders)!= 0 and product == "KELP"):
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if (len(order_depth.sell_orders)!= 0 and product == "RAINFOREST_RESIN"):
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < 9999:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -3*best_ask_amount))
            
            if (len(order_depth.buy_orders) != 0 and product == "RAINFOREST_RESIN"):
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > 10001:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -3*best_bid_amount))

            if (len(order_depth.buy_orders) != 0 and product == "KELP"):
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders

        # Update price history for future reference
        if not hasattr(self, 'price_history'):
            self.price_history = {}
        
        for product in state.order_depths:
            if len(state.order_depths[product].buy_orders) > 0 and len(state.order_depths[product].sell_orders) > 0:
                best_bid = max(state.order_depths[product].buy_orders.keys())
                best_ask = min(state.order_depths[product].sell_orders.keys())
                self.price_history[product] = (best_bid + best_ask) / 2

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
        # Sample conversion request
        conversions = 1
        return result, conversions, traderData
