from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import jsonpickle

class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"

PARAMS = {
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "reversion_beta": -0.22,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "soft_position_limit": 15,
        "ema_alpha": 0.2,  # EMA smoothing factor (higher = more weight to recent prices)
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        # Position limits for each product
        self.LIMIT = {Product.KELP: 20, Product.RAINFOREST_RESIN: 20}
        
        # Initialize price history for each product
        self.price_history = {}  # Format: {product: [list of prices]}
        self.max_history_length = 20  # Keep last 20 prices for filtering
        
        # Initialize EMA values for each product
        self.ema_values = {}  # Format: {product: current_ema}

    def _calculate_ema(self, product, current_price):
        """Calculate Exponential Moving Average"""
        alpha = self.params[product]["ema_alpha"]
        
        # If first calculation, use current price as starting EMA
        if product not in self.ema_values:
            self.ema_values[product] = current_price
            return current_price
        
        # EMA formula: EMA_today = alpha * price_today + (1 - alpha) * EMA_yesterday
        self.ema_values[product] = alpha * current_price + (1 - alpha) * self.ema_values[product]
        return self.ema_values[product]

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[int, int]:
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def dynamic_fair_value(self, product: str, order_depth: OrderDepth, traderObject) -> float:
        """Calculate fair value based on order book data with possible mean reversion"""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[product]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[product]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get(f"{product}_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject[f"{product}_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get(f"{product}_last_price", None) is not None:
                last_price = traderObject[f"{product}_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[product]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject[f"{product}_last_price"] = mmmid_price
            return fair
        
        # Fall back to configured fair value if available, or return None
        return self.params[product].get("fair_value", None)

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        print("traderData: " + (state.traderData if state.traderData else "None"))
        print("Observations: " + str(state.observations))

        # Initialize or load trader object from state
        traderObject = {}
        if state.traderData and state.traderData != "" and state.traderData != "SAMPLE":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except:
                print("Failed to decode trader data")

        # Orders to be placed on exchange matching engine
        result = {}

        # Process each product
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            
            # Skip if product not in our parameters
            if product not in self.params:
                continue
                
            position = state.position[product] if product in state.position else 0
            
            # Calculate mid price for history
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            
            if best_bid and best_ask:
                current_mid = (best_bid + best_ask) / 2
            elif best_bid:
                current_mid = best_bid
            elif best_ask:
                current_mid = best_ask
            else:
                current_mid = traderObject.get(f"{product}_last_price", 
                                               self.params[product].get("fair_value", 10000))
            
            # Update price history
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append(current_mid)
            self.price_history[product] = self.price_history[product][-self.max_history_length:]
            
            # Calculate fair value based on product
            if product == Product.KELP:
                # Use EMA for KELP price filtering instead of triangular filter
                ema_price = self._calculate_ema(product, current_mid)
                fair_value = self.dynamic_fair_value(product, order_depth, traderObject)
                if fair_value is None:
                    fair_value = ema_price
            else:  # RAINFOREST_RESIN
                # Use fixed fair value with slight dynamic adjustment
                fair_value = self.params[product]["fair_value"]
                if len(self.price_history[product]) > 5:
                    # Slight adjustment based on recent price movements
                    deviation = current_mid - fair_value
                    adjustment = deviation * 0.1  # Small adjustment factor
                    fair_value += adjustment
            
            print(f"Fair value for {product}: {fair_value}")
            
            # Trading logic with three-step approach
            buy_order_volume = 0
            sell_order_volume = 0
            
            # 1. Take opportunistic trades when prices are favorable
            take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                product,
                order_depth,
                fair_value,
                self.params[product]["take_width"],
                position,
                self.params[product].get("prevent_adverse", False),
                self.params[product].get("adverse_volume", 0)
            )
            
            # 2. Clear positions when needed
            clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                product,
                order_depth,
                fair_value,
                self.params[product]["clear_width"],
                position,
                buy_order_volume,
                sell_order_volume
            )
            
            # 3. Make markets to earn the spread
            make_orders, _, _ = self.make_orders(
                product,
                order_depth,
                fair_value,
                position,
                buy_order_volume,
                sell_order_volume,
                self.params[product]["disregard_edge"],
                self.params[product]["join_edge"], 
                self.params[product]["default_edge"],
                True,  # Always manage position
                self.params[product]["soft_position_limit"]
            )
            
            # Combine all orders for this product
            result[product] = take_orders + clear_orders + make_orders
            
            # Debug information
            print(f"Position for {product}: {position}")
            print(f"Orders for {product}: {len(result[product])}")

        # Store data for next iteration
        traderData = jsonpickle.encode(traderObject)
        
        # For now, no conversions
        conversions = 0
        
        return result, conversions, traderData