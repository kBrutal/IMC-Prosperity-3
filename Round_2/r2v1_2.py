import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias, Dict, List, Optional
import jsonpickle
import numpy as np
import math

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if value is None:
            return ""
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

    def get_mid_price(self, state: TradingState, symbol: str) -> Optional[float]:
        order_depth = state.order_depths.get(symbol, None)
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return None

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        return (best_bid + best_ask) / 2

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10
        self.ema_short = None
        self.ema_long = None
        self.alpha_short = 0.6  
        self.alpha_long = 0.1   

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        if true_value is None:
            return

        mid_price = self.get_mid_price(state, self.symbol)
        if mid_price is not None:
            if self.ema_short is None:
                self.ema_short = mid_price
                self.ema_long = mid_price
            else:
                self.ema_short = self.alpha_short * mid_price + (1 - self.alpha_short) * self.ema_short
                self.ema_long = self.alpha_long * mid_price + (1 - self.alpha_long) * self.ema_long

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        trend_adjustment = 0
        if self.ema_short is not None and self.ema_long is not None:
            if self.ema_short < self.ema_long:
                trend_adjustment = -2  
        max_buy_price = true_value - 1 + trend_adjustment if position > self.limit * 0.5 else true_value + trend_adjustment
        min_sell_price = true_value + 1 + trend_adjustment if position < self.limit * -0.5 else true_value + trend_adjustment

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return {
            "window": list(self.window),
            "ema_short": self.ema_short,
            "ema_long": self.ema_long
        }

    def load(self, data: JSON) -> None:
        if not data:
            return
        self.window = deque(data.get("window", []))
        self.ema_short = data.get("ema_short")
        self.ema_long = data.get("ema_long")

class CroissantsStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        mid_price = self.get_mid_price(state, self.symbol)
        if mid_price is None:
            return None
        if self.ema_short is not None and self.ema_long is not None:
            if self.ema_short < self.ema_long:
                return int(mid_price * 0.98) 
        return int(mid_price)

class JamsStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        mid_price = self.get_mid_price(state, self.symbol)
        if mid_price is None:
            return None
            
        if self.ema_short is not None and self.ema_long is not None:
            if self.ema_short < self.ema_long:
                return int(mid_price * 0.98)
        
        return int(mid_price)

class DjembesStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        mid_price = self.get_mid_price(state, self.symbol)
        if mid_price is None:
            return None
            
        if self.ema_short is not None and self.ema_long is not None:
            if self.ema_short < self.ema_long:
                return int(mid_price * 0.98) 
        
        return int(mid_price)

class BasketArbitrageStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, components: dict[str, int]) -> None:
        super().__init__(symbol, limit)
        self.components = components  
        self.historical_premiums = deque(maxlen=20) 
        self.avg_premium = None

    def act(self, state: TradingState) -> None:
        required_symbols = list(self.components.keys()) + [self.symbol]
        if any(symbol not in state.order_depths for symbol in required_symbols):
            return

        basket_theoretical_value = 0
        component_prices = {}
        
        for component, quantity in self.components.items():
            mid_price = self.get_mid_price(state, component)
            if mid_price is None:
                return  
            component_prices[component] = mid_price
            basket_theoretical_value += mid_price * quantity

        basket_price = self.get_mid_price(state, self.symbol)
        if basket_price is None:
            return

        premium = basket_price - basket_theoretical_value
        
        self.historical_premiums.append(premium)
        if not self.avg_premium and len(self.historical_premiums) >= 5:
            self.avg_premium = sum(self.historical_premiums) / len(self.historical_premiums)
        elif self.avg_premium:
            self.avg_premium = 0.8 * self.avg_premium + 0.2 * premium  
            
        if not self.avg_premium:
            return  
            
        position = state.position.get(self.symbol, 0)
        component_positions = {comp: state.position.get(comp, 0) for comp in self.components}
        
        premium_threshold_buy = self.avg_premium * 1 
        premium_threshold_sell = self.avg_premium * 1
        
        if premium < premium_threshold_buy:
            logger.print(f"Basket {self.symbol} is cheap: premium {premium} vs avg {self.avg_premium}")
            self.execute_basket_arbitrage(state, True, position, component_positions)
        elif premium > premium_threshold_sell:
            logger.print(f"Basket {self.symbol} is expensive: premium {premium} vs avg {self.avg_premium}")
            self.execute_basket_arbitrage(state, False, position, component_positions)
            
    def execute_basket_arbitrage(self, state: TradingState, buy_basket: bool, 
                               basket_position: int, component_positions: dict[str, int]) -> None:
        max_baskets = self.limit - abs(basket_position)
        if max_baskets <= 0:
            return
            
        if buy_basket:
            for component, qty in self.components.items():
                comp_position = component_positions.get(component, 0)
                max_sell = self.limit + comp_position 
                max_baskets = min(max_baskets, max_sell // qty)
                
            if max_baskets <= 0:
                return
                
            order_depth = state.order_depths[self.symbol]
            sell_orders = sorted(order_depth.sell_orders.items())
            
            baskets_to_buy = max_baskets
            for price, volume in sell_orders:
                if baskets_to_buy <= 0:
                    break
                    
                quantity = min(baskets_to_buy, -volume)
                if quantity > 0:
                    self.buy(price, quantity)
                    baskets_to_buy -= quantity
                    
                    for component, qty in self.components.items():
                        comp_order_depth = state.order_depths[component]
                        buy_orders = sorted(comp_order_depth.buy_orders.items(), reverse=True)
                        
                        if buy_orders:
                            best_buy_price = buy_orders[0][0]
                            self.orders.append(Order(component, best_buy_price, -quantity * qty))
        else:
            for component, qty in self.components.items():
                comp_position = component_positions.get(component, 0)
                max_buy = self.limit - comp_position  # How many more we can buy
                max_baskets = min(max_baskets, max_buy // qty)
                
            if max_baskets <= 0:
                return
                
            order_depth = state.order_depths[self.symbol]
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            
            baskets_to_sell = max_baskets
            for price, volume in buy_orders:
                if baskets_to_sell <= 0:
                    break
                    
                quantity = min(baskets_to_sell, volume)
                if quantity > 0:
                    self.sell(price, quantity)
                    baskets_to_sell -= quantity
                    
                    for component, qty in self.components.items():
                        comp_order_depth = state.order_depths[component]
                        sell_orders = sorted(comp_order_depth.sell_orders.items())
                        
                        if sell_orders:
                            best_sell_price = sell_orders[0][0]
                            self.orders.append(Order(component, best_sell_price, quantity * qty))

    def save(self) -> JSON:
        return {
            "historical_premiums": list(self.historical_premiums),
            "avg_premium": self.avg_premium
        }

    def load(self, data: JSON) -> None:
        if not data:
            return
        self.historical_premiums = deque(data.get("historical_premiums", []), maxlen=20)
        self.avg_premium = data.get("avg_premium")

class PicnicBasket1Strategy(BasketArbitrageStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        components = {
            "CROISSANTS": 6,
            "JAMS": 3,
            "DJEMBES": 1
        }
        super().__init__(symbol, limit, components)


class OceanProductStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, params: dict) -> None:
        super().__init__(symbol, limit)
        self.params = params
        self.last_price = None
        self.moving_average = None
        
    def take_best_orders(self, state: TradingState, fair_value: int, take_width: float) -> None:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        position_limit = self.limit
        prevent_adverse = self.params.get("prevent_adverse", False)
        adverse_volume = self.params.get("adverse_volume", 0)
        
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)  # max amt to buy
                    if quantity > 0:
                        self.buy(best_ask, quantity)

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)  # max amt to sell
                    if quantity > 0:
                        self.sell(best_bid, quantity)
    
    def clear_position(self, state: TradingState, fair_value: float, width: int) -> None:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        
        if position > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position)
            if clear_quantity > 0:
                self.sell(fair_for_ask, clear_quantity)

        if position < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position))
            if clear_quantity > 0:
                self.buy(fair_for_bid, clear_quantity)
    
    def make_market(self, state: TradingState, fair_value: float) -> None:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        disregard_edge = self.params.get("disregard_edge", 1)
        join_edge = self.params.get("join_edge", 2)
        default_edge = self.params.get("default_edge", 4)
        soft_position_limit = self.params.get("soft_position_limit", 60)

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
                ask = best_ask_above_fair 
            else:
                ask = best_ask_above_fair - 1 

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if position > soft_position_limit:
            ask -= 1 
        elif position < -soft_position_limit:
            bid += 1 

        buy_quantity = self.limit - position
        if buy_quantity > 0:
            self.buy(bid, buy_quantity)

        sell_quantity = self.limit + position
        if sell_quantity > 0:
            self.sell(ask, sell_quantity)
    
    def act(self, state: TradingState) -> None:
        fair_value = self.calculate_fair_value(state)
        if fair_value is None:
            return
        self.take_best_orders(state, fair_value, self.params.get("take_width", 1))
        self.clear_position(state, fair_value, self.params.get("clear_width", 0))
        self.make_market(state, fair_value)
    
    @abstractmethod
    def calculate_fair_value(self, state: TradingState) -> Optional[float]:
        raise NotImplementedError()
    
    def save(self) -> JSON:
        return {
            "last_price": self.last_price,
            "moving_average": self.moving_average
        }

    def load(self, data: JSON) -> None:
        if not data:
            return
        self.last_price = data.get("last_price")
        self.moving_average = data.get("moving_average")

class RainforestResinStrategy(OceanProductStrategy):
    def calculate_fair_value(self, state: TradingState) -> float:
        return self.params["fair_value"]

class PicnicBasket2Strategy(OceanProductStrategy):
    def calculate_fair_value(self, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths[self.symbol]
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params["adverse_volume"]
        ]
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params["adverse_volume"]
        ]
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
        
        if mm_ask is None or mm_bid is None:
            if self.last_price is None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = self.last_price
        else:
            mmmid_price = (mm_ask + mm_bid) / 2

        if self.last_price is not None:
            last_returns = (mmmid_price - self.last_price) / self.last_price
            pred_returns = last_returns * self.params["reversion_beta"]
            fair = mmmid_price + (mmmid_price * pred_returns)
        else:
            fair = mmmid_price
            
        self.last_price = mmmid_price
        return fair

class KelpStrategy(OceanProductStrategy):
    def calculate_fair_value(self, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths[self.symbol]
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params["adverse_volume"]
        ]
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params["adverse_volume"]
        ]
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
        
        if mm_ask is None or mm_bid is None:
            if self.last_price is None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = self.last_price
        else:
            mmmid_price = (mm_ask + mm_bid) / 2

        if self.last_price is not None:
            last_returns = (mmmid_price - self.last_price) / self.last_price
            pred_returns = last_returns * self.params["reversion_beta"]
            fair = mmmid_price + (mmmid_price * pred_returns)
        else:
            fair = mmmid_price
            
        self.last_price = mmmid_price
        return fair

class SquidInkStrategy(OceanProductStrategy):
    def calculate_fair_value(self, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths[self.symbol]
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params["adverse_volume"]
        ]
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params["adverse_volume"]
        ]
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
        
        if mm_ask is None or mm_bid is None:
            if self.last_price is None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = self.last_price
        else:
            mmmid_price = (mm_ask + mm_bid) / 2
        window = self.params.get("deviation_window", 50)
        if self.moving_average is None:
            self.moving_average = mmmid_price
        else:
            alpha = 2.0 / (window + 1)
            self.moving_average = alpha * mmmid_price + (1 - alpha) * self.moving_average
        deviation = (mmmid_price - self.moving_average) / self.moving_average
        threshold = self.params.get("deviation_threshold", 0.07)
        reversion_factor = self.params.get("reversion_factor", 0.9)
        
        if abs(deviation) > threshold:
            pred_returns = -reversion_factor * deviation
            fair_value = mmmid_price + (mmmid_price * pred_returns)
            logger.print(f"SQUID_INK deviation: {deviation:.4f}, applying reversion to {fair_value:.2f}")
        else:
            fair_value = mmmid_price
            
        self.last_price = mmmid_price
        return fair_value

class Trader:
    def __init__(self) -> None:
        limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 0,
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 10
        }
        
        # Parameters for ocean products
        ocean_params = {
            "RAINFOREST_RESIN": {
                "fair_value": 10000,
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 40,
            },
            "KELP": {
                "take_width": 1,
                "clear_width": 0,
                "prevent_adverse": True,
                "adverse_volume": 20,
                "reversion_beta": -0.22,
                "disregard_edge": 1,
                "join_edge": 0,
                "default_edge": 1,
            },
            "PICNIC_BASKET2": {
                "take_width": 1,
                "clear_width": 0,
                "prevent_adverse": True,
                "adverse_volume": 20,
                "reversion_beta": -0.7,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 1,
            },
            "SQUID_INK": {
                "take_width": 20,
                "clear_width": 0,
                "disregard_edge": 0,
                "join_edge": 4,
                "default_edge": 2,
                "prevent_adverse": True,
                "adverse_volume": 20,
                "deviation_window": 500,
                "deviation_threshold": 0.01,
                "reversion_factor": -0.75,
            }

        }
        
        self.strategies: dict[Symbol, Strategy] = {
            "CROISSANTS": CroissantsStrategy("CROISSANTS", limits["CROISSANTS"]),
            "JAMS": JamsStrategy("JAMS", limits["JAMS"]),
            "DJEMBES": DjembesStrategy("DJEMBES", limits["DJEMBES"]),
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1", limits["PICNIC_BASKET1"]),
            "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2", limits["PICNIC_BASKET2"], ocean_params["PICNIC_BASKET2"]),
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"],                                                ocean_params["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"], ocean_params["KELP"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"], ocean_params["SQUID_INK"])
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])
            if symbol in state.order_depths:
                position = state.position.get(symbol, 0)
                logger.print(f"{symbol} position: {position}")
                strategy_orders, strategy_conversions = strategy.run(state)
                
                if strategy_orders:
                    orders[symbol] = strategy_orders
                    
                conversions += strategy_conversions
                new_trader_data[symbol] = strategy.save()
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data