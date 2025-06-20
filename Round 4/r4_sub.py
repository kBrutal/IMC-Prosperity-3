from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import List, Dict, Any, Tuple
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
import json
from statistics import NormalDist
from typing import List
import string


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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()


class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"  # Add this line
    KELP = "KELP"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANT = "CROISSANTS"
    JAM = "JAMS"
    DJEMBE = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"


PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "initial_mean_volatility": 3.054914250441493,
        "strike": 9500,
        "starting_time_to_expiry": 4.50/250,
        "std_window": 150,
        "zscore_threshold": 5.0,
        "max_iv_multiplier": 2.5,
        "min_iv": 0.05,
        "ema_span": 500  
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "initial_mean_volatility": 2.1993493676716134,
        "strike": 9750,
        "starting_time_to_expiry": 4.50/250,
        "std_window": 150,
        "zscore_threshold": 0.3,
        "max_iv_multiplier": 2.5,
        "min_iv": 0.05,
        "ema_span": 500
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "initial_mean_volatility": 1.3780988354692982,
        "strike": 10000,
        "starting_time_to_expiry": 4.50/250,
        "std_window": 80,
        "zscore_threshold": 0.6,
        "max_iv_multiplier": 2.0,
        "min_iv": 0.05,
        "ema_span": 100
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "initial_mean_volatility": 0.636751294108902,
        "strike": 10250,
        "starting_time_to_expiry": 4.50/250,
        "std_window": 150,
        "zscore_threshold": 2.0,
        "max_iv_multiplier": 1.8,
        "min_iv": 0.05,
        "ema_span": 500
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "initial_mean_volatility": 0.19992167398753208,
        "strike": 10500,
        "starting_time_to_expiry": 4.50/250,
        "std_window": 150,
        "zscore_threshold": 5.0,
        "max_iv_multiplier": 1.5,
        "min_iv": 0.05,
        "ema_span": 500
    },
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.8,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size": 0.5
    },
    "CSI": {
        "threshold": 50,
    },
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "reversion_beta": 0.229,
        "kelp_min_edge": 2,
    },
    Product.SPREAD1: {
        "default_spread_mean": 48.77785696398906,
        "default_spread_std": 85.12114180548527,
        "spread_std_window": 40,
        "zscore_threshold": 10,
        "target_position": 60,
        },
    Product.SPREAD2: {
        "default_spread_mean": 30.233664373870205,
        "default_spread_std": 59.85465092438831,
        "spread_std_window": 30,
        "zscore_threshold": 6,
        "target_position": 100,
    },
}

BASKET1_WEIGHTS = {
    Product.CROISSANT: 6,
    Product.JAM: 3,
    Product.DJEMBE: 1,
}

BASKET2_WEIGHTS = {
    Product.CROISSANT: 4,
    Product.JAM: 2,
}


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
            self.params = params
        self.params = params

        self.LIMIT = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.MAGNIFICENT_MACARONS: 75,
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANT: 250,
            Product.JAM: 350,
            Product.DJEMBE: 60,
        }
        
        self.VOUCHERS = [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500,
        ]
        
        # Volatility tracking
        self.current_iv = {voucher: params[voucher]['initial_mean_volatility'] for voucher in self.VOUCHERS}
        self.ema_iv = {voucher: params[voucher]['initial_mean_volatility'] for voucher in self.VOUCHERS}
        
        # Hedge tracking
        self.last_hedge_time = 0
        self.last_spot_price = 10000  # Default starting price
        self.hedge_count = 0

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
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
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

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
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
            if abs(best_bid_amount) <= adverse_volume:
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
    ) -> (int, int):
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
    ) -> (int, int):
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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def make_rainforest_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        try:
            baaf = min(
                [
                    price
                    for price in order_depth.sell_orders.keys()
                    if price > fair_value + 1
                ]
            )
        except:
            baaf = fair_value + 3

        try:
            bbbf = max(
                [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            )
        except:
            bbbf = fair_value - 3

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern
        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
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
    ) -> (List[Order], int, int):
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

    def make_kelp_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)

        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket1_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANT_PER_BASKET = BASKET1_WEIGHTS[Product.CROISSANT]
        JAM_PER_BASKET = BASKET1_WEIGHTS[Product.JAM]
        DJEMBE_PER_BASKET = BASKET1_WEIGHTS[Product.DJEMBE]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissant_best_bid = (
            max(order_depths[Product.CROISSANT].buy_orders.keys())
            if order_depths[Product.CROISSANT].buy_orders
            else 0
        )
        croissant_best_ask = (
            min(order_depths[Product.CROISSANT].sell_orders.keys())
            if order_depths[Product.CROISSANT].sell_orders
            else float("inf")
        )
        jam_best_bid = (
            max(order_depths[Product.JAM].buy_orders.keys())
            if order_depths[Product.JAM].buy_orders
            else 0
        )
        jam_best_ask = (
            min(order_depths[Product.JAM].sell_orders.keys())
            if order_depths[Product.JAM].sell_orders
            else float("inf")
        )
        djembe_best_bid = (
            max(order_depths[Product.DJEMBE].buy_orders.keys())
            if order_depths[Product.DJEMBE].buy_orders
            else 0
        )
        djembe_best_ask = (
            min(order_depths[Product.DJEMBE].sell_orders.keys())
            if order_depths[Product.DJEMBE].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissant_best_bid * CROISSANT_PER_BASKET
            + jam_best_bid * JAM_PER_BASKET
            + djembe_best_bid * DJEMBE_PER_BASKET
        )
        implied_ask = (
            croissant_best_ask * CROISSANT_PER_BASKET
            + jam_best_ask * JAM_PER_BASKET
            + djembe_best_ask * DJEMBE_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissant_bid_volume = (
                order_depths[Product.CROISSANT].buy_orders[croissant_best_bid]
                // CROISSANT_PER_BASKET
            )
            jam_bid_volume = (
                order_depths[Product.JAM].buy_orders[jam_best_bid]
                // JAM_PER_BASKET
            )
            djembe_bid_volume = (
                order_depths[Product.DJEMBE].buy_orders[djembe_best_bid]
                // DJEMBE_PER_BASKET
            )
            implied_bid_volume = min(
                croissant_bid_volume, jam_bid_volume, djembe_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissant_ask_volume = (
                -order_depths[Product.CROISSANT].sell_orders[croissant_best_ask]
                // CROISSANT_PER_BASKET
            )
            jam_ask_volume = (
                -order_depths[Product.JAM].sell_orders[jam_best_ask]
                // JAM_PER_BASKET
            )
            djembe_ask_volume = (
                -order_depths[Product.DJEMBE].sell_orders[djembe_best_ask]
                // DJEMBE_PER_BASKET
            )
            implied_ask_volume = min(
                croissant_ask_volume, jam_ask_volume, djembe_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def get_synthetic_basket2_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANT_PER_BASKET = BASKET2_WEIGHTS[Product.CROISSANT]
        JAM_PER_BASKET = BASKET2_WEIGHTS[Product.JAM]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissant_best_bid = (
            max(order_depths[Product.CROISSANT].buy_orders.keys())
            if order_depths[Product.CROISSANT].buy_orders
            else 0
        )
        croissant_best_ask = (
            min(order_depths[Product.CROISSANT].sell_orders.keys())
            if order_depths[Product.CROISSANT].sell_orders
            else float("inf")
        )
        jam_best_bid = (
            max(order_depths[Product.JAM].buy_orders.keys())
            if order_depths[Product.JAM].buy_orders
            else 0
        )
        jam_best_ask = (
            min(order_depths[Product.JAM].sell_orders.keys())
            if order_depths[Product.JAM].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissant_best_bid * CROISSANT_PER_BASKET
            + jam_best_bid * JAM_PER_BASKET
        )
        implied_ask = (
            croissant_best_ask * CROISSANT_PER_BASKET
            + jam_best_ask * JAM_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissant_bid_volume = (
                order_depths[Product.CROISSANT].buy_orders[croissant_best_bid]
                // CROISSANT_PER_BASKET
            )
            jam_bid_volume = (
                order_depths[Product.JAM].buy_orders[jam_best_bid]
                // JAM_PER_BASKET
            )
            implied_bid_volume = min(croissant_bid_volume, jam_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissant_ask_volume = (
                -order_depths[Product.CROISSANT].sell_orders[croissant_best_ask]
                // CROISSANT_PER_BASKET
            )
            jam_ask_volume = (
                -order_depths[Product.JAM].sell_orders[jam_best_ask]
                // JAM_PER_BASKET
            )
            implied_ask_volume = min(croissant_ask_volume, jam_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket1_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANT: [],
            Product.JAM: [],
            Product.DJEMBE: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket1_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissant_price = min(
                    order_depths[Product.CROISSANT].sell_orders.keys()
                )
                jam_price = min(
                    order_depths[Product.JAM].sell_orders.keys()
                )
                djembe_price = min(order_depths[Product.DJEMBE].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissant_price = max(order_depths[Product.CROISSANT].buy_orders.keys())
                jam_price = max(
                    order_depths[Product.JAM].buy_orders.keys()
                )
                djembe_price = max(order_depths[Product.DJEMBE].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissant_order = Order(
                Product.CROISSANT,
                croissant_price,
                quantity * BASKET1_WEIGHTS[Product.CROISSANT],
            )
            jam_order = Order(
                Product.JAM,
                jam_price,
                quantity * BASKET1_WEIGHTS[Product.JAM],
            )
            djembe_order = Order(
                Product.DJEMBE, djembe_price, quantity * BASKET1_WEIGHTS[Product.DJEMBE]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANT].append(croissant_order)
            component_orders[Product.JAM].append(jam_order)
            component_orders[Product.DJEMBE].append(djembe_order)

        return component_orders

    def convert_synthetic_basket2_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANT: [],
            Product.JAM: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissant_price = min(
                    order_depths[Product.CROISSANT].sell_orders.keys()
                )
                jam_price = min(
                    order_depths[Product.JAM].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissant_price = max(order_depths[Product.CROISSANT].buy_orders.keys())
                jam_price = max(
                    order_depths[Product.JAM].buy_orders.keys()
                )
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissant_order = Order(
                Product.CROISSANT,
                croissant_price,
                quantity * BASKET2_WEIGHTS[Product.CROISSANT],
            )
            jam_order = Order(
                Product.JAM,
                jam_price,
                quantity * BASKET2_WEIGHTS[Product.JAM],
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANT].append(croissant_order)
            component_orders[Product.JAM].append(jam_order)

        return component_orders

    def execute_spread1_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC1, synthetic_bid_price, -execute_volume)
            ]
            aggregate_orders = self.convert_synthetic_basket1_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC1, synthetic_ask_price, execute_volume)
            ]
            aggregate_orders = self.convert_synthetic_basket1_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def execute_spread2_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)
            ]
            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)
            ]
            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

    def spread1_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD1]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD1]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        zscore = (
                     spread - self.params[Product.SPREAD1]["default_spread_mean"]
                 ) / spread_std

        if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    -self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def spread2_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        zscore = (
                     spread - self.params[Product.SPREAD2]["default_spread_mean"]
                 ) / spread_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def resolve_component_collisions(self, result, state, traderObject, product, basket1_orders, basket2_orders):
        # If either basket doesn't have orders for this component, just use the other
        if not basket1_orders:
            if basket2_orders:
                result[product] = basket2_orders
            return

        if not basket2_orders:
            result[product] = basket1_orders
            return

        # Calculate total quantities from both baskets
        basket1_quantity = sum(order.quantity for order in basket1_orders)
        basket2_quantity = sum(order.quantity for order in basket2_orders)

        # Get current position and limits
        current_position = state.position.get(product, 0)
        position_limit = self.LIMIT[product]

        # Calculate z-scores for prioritization
        basket1_zscore = abs(traderObject[Product.SPREAD_BASKET1].get("prev_zscore", 0))
        basket2_zscore = abs(traderObject[Product.SPREAD_BASKET2].get("prev_zscore", 0))

        # Case 1: Opposite directions (one long, one short)
        if (basket1_quantity > 0 and basket2_quantity < 0) or (basket1_quantity < 0 and basket2_quantity > 0):
            net_quantity = basket1_quantity + basket2_quantity

            # If net position within limits, can do both
            if current_position + net_quantity <= position_limit and current_position + net_quantity >= -position_limit:
                result[product] = basket1_orders + basket2_orders
                logger.print(f"Taking both opposite {product} orders, net: {net_quantity}")
            else:
                # Prioritize based on z-score
                if basket1_zscore >= basket2_zscore:
                    result[product] = basket1_orders
                    logger.print(
                        f"Prioritizing Basket1 {product} orders due to higher z-score: {basket1_zscore} vs {basket2_zscore}")
                else:
                    result[product] = basket2_orders
                    logger.print(
                        f"Prioritizing Basket2 {product} orders due to higher z-score: {basket2_zscore} vs {basket1_zscore}")

        # Case 2: Same direction (both long or both short)
        else:
            total_quantity = basket1_quantity + basket2_quantity

            # Check if adding both would exceed limit
            if (total_quantity > 0 and current_position + total_quantity <= position_limit) or \
                (total_quantity < 0 and current_position + total_quantity >= -position_limit):
                # Can add both without exceeding limit
                result[product] = basket1_orders + basket2_orders
                logger.print(f"Taking both same-direction {product} orders, total: {total_quantity}")
            else:
                # Need to scale down - prioritize by z-score
                # Calculate available capacity
                available_capacity = position_limit - current_position if total_quantity > 0 else position_limit + current_position
                available_capacity = abs(available_capacity)

                # Weighted allocation based on z-scores
                total_zscore = basket1_zscore + basket2_zscore

                if total_zscore > 0:
                    basket1_allocation = int(available_capacity * (basket1_zscore / total_zscore))
                    basket2_allocation = available_capacity - basket1_allocation

                    # Ensure neither allocation exceeds the requested quantity
                    basket1_allocation = min(basket1_allocation, abs(basket1_quantity))
                    basket2_allocation = min(basket2_allocation, abs(basket2_quantity))

                    # Scale orders
                    scaled_basket1 = self.scale_component_orders(product, basket1_orders,
                                                                 basket1_allocation / abs(basket1_quantity))
                    scaled_basket2 = self.scale_component_orders(product, basket2_orders,
                                                                 basket2_allocation / abs(basket2_quantity))

                    result[product] = scaled_basket1 + scaled_basket2
                    logger.print(
                        f"Scaled {product} orders - Basket1: {basket1_allocation}/{abs(basket1_quantity)}, Basket2: {basket2_allocation}/{abs(basket2_quantity)}")
                else:
                    # Equal allocation if both z-scores are zero
                    allocation_ratio = available_capacity / abs(total_quantity)
                    scaled_basket1 = self.scale_component_orders(product, basket1_orders, allocation_ratio)
                    scaled_basket2 = self.scale_component_orders(product, basket2_orders, allocation_ratio)

                    result[product] = scaled_basket1 + scaled_basket2
                    logger.print(f"Equal scaling for {product} orders: {allocation_ratio}")

    def scale_component_orders(self, product, orders, scale_factor):
        scaled_orders = []
        for order in orders:
            # Scale quantity and ensure it's an integer
            scaled_quantity = int(order.quantity * scale_factor)
            if abs(scaled_quantity) > 0:  # Skip orders that would scale to zero
                scaled_orders.append(Order(product, order.price, scaled_quantity))
        return scaled_orders
    
    def magnificent_macarons_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1, observation.askPrice + observation.importTariff + observation.transportFees

    def magnificent_macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float: 
        if timestamp == 0:
            traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]

        traderObject["MAGNIFICENT_MACARONS"]["volume_history"].append(abs(position))
        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) > self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            traderObject["MAGNIFICENT_MACARONS"]["volume_history"].pop(0)

        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) < self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["MAGNIFICENT_MACARONS"]["optimized"]:
            volume_avg = np.mean(traderObject["MAGNIFICENT_MACARONS"]["volume_history"])

            if volume_avg >= self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]:
                traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = []
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                return curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]

            elif self.params[Product.MAGNIFICENT_MACARONS]["dec_edge_discount"] * self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"] * (curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"] > self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]:
                    traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = []
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                    traderObject["MAGNIFICENT_MACARONS"]["optimized"] = True
                    return curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                else:
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]
                    return self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]

        traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge
        return curr_edge

    def magnificent_macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.magnificent_macarons_implied_bid_ask(observation)                                                                                                                                                                    

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.MAGNIFICENT_MACARONS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume
    
    def predict_fair_value(self, observation):
        # Extract features
        sugar_price = observation.sugarPrice
        sunlight_index = observation.sunlightIndex
        transport_fees = observation.transportFees
        export_tariff = observation.exportTariff
        import_tariff = observation.importTariff
        
        # Coefficients derived from historical data analysis
        # These would be calibrated from regression analysis
        sugar_coef = 3.5  # Strong positive correlation observed
        sunlight_coef = -2.0  # Negative correlation observed
        transport_coef = -5.0  # Negative impact on price
        export_coef = -0.8  # Slight negative impact
        import_coef = 1.2  # Positive impact when import tariff decreases
        
        # Base value (intercept) from regression
        base_value = 100.0
        
        # Calculate predicted fair value
        fair_value = (base_value + 
                    sugar_coef * sugar_price +
                    sunlight_coef * sunlight_index +
                    transport_coef * transport_fees +
                    export_coef * export_tariff +
                    import_coef * import_tariff)
        
        return fair_value


    def calculate_dynamic_spread(self, timestamp, curr_edge, position, traderObject, observation):
        # Get base spread from existing adaptive edge function
        base_spread = self.magnificent_macarons_adap_edge(timestamp, curr_edge, position, traderObject)
        
        # Calculate predicted fair value
        fair_value = self.predict_fair_value(observation)
        
        # Calculate market volatility using price history
        if "price_history" not in traderObject["MAGNIFICENT_MACARONS"]:
            traderObject["MAGNIFICENT_MACARONS"]["price_history"] = []
        
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        traderObject["MAGNIFICENT_MACARONS"]["price_history"].append(foreign_mid)
        
        # Keep only last 20 prices
        if len(traderObject["MAGNIFICENT_MACARONS"]["price_history"]) > 20:
            traderObject["MAGNIFICENT_MACARONS"]["price_history"].pop(0)
        
        # Calculate volatility as standard deviation
        if len(traderObject["MAGNIFICENT_MACARONS"]["price_history"]) > 5:
            volatility = np.std(traderObject["MAGNIFICENT_MACARONS"]["price_history"])
            # Adjust spread based on volatility
            volatility_adjustment = min(volatility * 0.5, 2.0)  # Cap at 2.0
        else:
            volatility_adjustment = 0
        
        # Calculate fair value deviation
        market_mid = (traderObject["MAGNIFICENT_MACARONS"].get("best_bid", 0) + 
                    traderObject["MAGNIFICENT_MACARONS"].get("best_ask", 0)) / 2
        
        if market_mid > 0:
            fair_value_deviation = abs(market_mid - fair_value) / market_mid
            fair_value_adjustment = min(fair_value_deviation * 10, 1.5)  # Cap at 1.5
        else:
            fair_value_adjustment = 0
        
        # Final spread calculation
        final_spread = base_spread + volatility_adjustment + fair_value_adjustment
        
        # Ensure minimum spread
        final_spread = max(final_spread, self.params[Product.MAGNIFICENT_MACARONS]["min_edge"])
        
        return final_spread


    def calculate_position_bias(self, observation, traderObject):
        # Calculate expected price movement direction and magnitude
        fair_value = self.predict_fair_value(observation)
        
        if "last_fair_value" not in traderObject["MAGNIFICENT_MACARONS"]:
            traderObject["MAGNIFICENT_MACARONS"]["last_fair_value"] = fair_value
            expected_movement = 0
        else:
            expected_movement = fair_value - traderObject["MAGNIFICENT_MACARONS"]["last_fair_value"]
            traderObject["MAGNIFICENT_MACARONS"]["last_fair_value"] = fair_value
        
        # Calculate expected holding time (in timestamps)
        expected_holding_time = 10  # Default assumption
        
        # Calculate storage cost for expected holding period
        storage_cost = 0.1 * expected_holding_time
        
        # If expected upward movement is less than storage cost, bias toward short
        if expected_movement < storage_cost:
            return -1  # Short bias
        elif expected_movement > storage_cost * 2:
            return 1   # Long bias (only if movement significantly exceeds cost)
        else:
            return 0   # Neutral

    def buy_and_hold_strategy(self, state, product, position, position_limit):
        """
        Implements a simple buy and hold strategy when sunlightIndex is below threshold
        """
        orders = []
        
        # Only for MAGNIFICENT_MACARONS since that's what has the sunlightIndex
        if product != Product.MAGNIFICENT_MACARONS:
            return orders
        
        # Check available space in position
        available_buy = position_limit - position
        
        if available_buy <= 0:
            return orders  # Already at position limit
        
        # Get order depth
        order_depth = state.order_depths[product]
        
        # Buy at market price if there are sell orders
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            quantity = min(available_buy, abs(order_depth.sell_orders[best_ask]))
            if quantity > 0:
                orders.append(Order(product, best_ask, quantity))
        
        return orders

    def optimize_conversions(self, position, observation):
        # Calculate implied prices
        implied_bid, implied_ask = self.magnificent_macarons_implied_bid_ask(observation)
        
        # Calculate foreign midpoint
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        
        # Determine if conversion is favorable
        if position > 0:  # We're long, consider selling to Pristine Cuisine
            conversion_price = implied_bid
            market_price = foreign_mid
            
            # Only convert if we get better price from Pristine than market
            if conversion_price > market_price:
                return min(position, 10)  # Convert up to 10 units
            else:
                return 0  # Hold position
                
        elif position < 0:  # We're short, consider buying from Pristine Cuisine
            conversion_price = implied_ask
            market_price = foreign_mid
            
            # Only convert if we get better price from Pristine than market
            if conversion_price < market_price:
                return max(position, -10)  # Convert up to 10 units
            else:
                return 0  # Hold position
        
        return 0  # No position, no conversion


    
    def magnificent_macarons_arb_clear(
        self,
        position: int
    ) -> int:
        conversions = min(abs(position), 10)  # Conversion limit of 10
        return -conversions if position > 0 else conversions

    def magnificent_macarons_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]

        implied_bid, implied_ask = self.magnificent_macarons_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask >= implied_ask + self.params[Product.MAGNIFICENT_MACARONS]['min_edge']:
            ask = aggressive_ask

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 25]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]

        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    def calculate_ema(self, new_value, current_ema, span):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (span + 1)
        return alpha * new_value + (1 - alpha) * current_ema

    def safe_implied_vol(self, price, spot, strike, tte, voucher):
        """Bounds-checked IV calculation"""
        try:
            iv = BlackScholes.implied_volatility(price, spot, strike, tte)
            max_iv = self.params[voucher]['initial_mean_volatility'] * self.params[voucher]['max_iv_multiplier']
            min_iv = self.params[voucher]['min_iv']
            return min(max(iv, min_iv), max_iv)
        except:
            return None

    def adjust_vol_for_moneyness(self, spot_price):
        """Adjust IV based on moneyness (volatility smile)"""
        for voucher in self.VOUCHERS:
            strike = self.params[voucher]['strike']
            moneyness = spot_price / strike
            
            if moneyness < 0.95:  # Deep OTM
                self.current_iv[voucher] = self.ema_iv[voucher] * 1.3
            elif moneyness > 1.05:  # Deep ITM
                self.current_iv[voucher] = self.ema_iv[voucher] * 0.7
            else:
                self.current_iv[voucher] = self.ema_iv[voucher]

    def calculate_gamma_adjusted_delta(self, spot, positions, tte):
        """Calculate hedge needs considering gamma"""
        total_delta = 0
        for voucher in self.VOUCHERS:
            if voucher not in positions:
                continue
                
            strike = self.params[voucher]['strike']
            iv = self.current_iv.get(voucher, self.ema_iv[voucher])
            
            delta = BlackScholes.delta(spot, strike, tte, iv)
            gamma = BlackScholes.gamma(spot, strike, tte, iv)
            
            # Project delta change for next 1% move
            projected_delta = delta + (gamma * spot * 0.01)
            total_delta += positions[voucher] * projected_delta
        
        return round(-total_delta)  # Negative because we hedge

    def should_rehedge(self, current_time, spot_price):
        """Determine if we need to hedge"""
        time_diff = current_time - self.last_hedge_time
        price_diff_pct = abs(spot_price - self.last_spot_price) / self.last_spot_price
        
        # Rehedge if:
        # - More than 1 hour passed OR
        # - Price moved more than 0.5% OR
        # - Never hedged before
        return (time_diff > 3600 or 
                price_diff_pct > 0.005 or 
                self.last_hedge_time == 0)

    def execute_hedge(self, state, target_delta):
        """Execute hedge in chunks"""
        current_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        delta_diff = target_delta - current_position
        
        if abs(delta_diff) < 5:  # No hedge for tiny differences
            return []
        
        orders = []
        order_depth = state.order_depths.get(Product.VOLCANIC_ROCK, OrderDepth())
        
        if delta_diff > 0:  # Need to buy
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if best_ask:
                chunk_size = max(1, delta_diff // 5)
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, chunk_size))
        else:  # Need to sell
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            if best_bid:
                chunk_size = max(1, abs(delta_diff) // 5)
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -chunk_size))
        
        if orders:
            self.last_hedge_time = state.timestamp
            self.last_spot_price = self.get_spot_price(state)
            self.hedge_count += 1
        
        return orders

    def get_spot_price(self, state):
        """Get current spot price with fallback"""
        order_depth = state.order_depths.get(Product.VOLCANIC_ROCK, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders.keys()) + 
                    min(order_depth.sell_orders.keys())) / 2
        return self.last_spot_price

    def get_time_to_expiry(self, timestamp):
        """Convert timestamp to years to expiry"""
        days_remaining = 7 - (timestamp // 1_000_000)  # Assuming 1M per day
        return max(days_remaining, 1) / 365.25

    def get_voucher_mid_price(self, voucher_order_depth, traderData, voucher):
        if (
            len(voucher_order_depth.buy_orders) > 0
            and len(voucher_order_depth.sell_orders) > 0
        ):
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            traderData[f"prev_{voucher}_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData.get(f"prev_{voucher}_price", 0)

    def volcanic_rock_voucher_orders(
        self,
        voucher: str,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> Tuple[List[Order], List[Order]]:
        vol_key = f"past_{voucher}_vol"
        ema_key = f"ema_{voucher}_vol"
        
        if vol_key not in traderData:
            traderData[vol_key] = []
            
        # Update EMA
        if ema_key not in traderData:
            traderData[ema_key] = self.params[voucher]['initial_mean_volatility']
        else:
            traderData[ema_key] = self.calculate_ema(
                volatility, 
                traderData[ema_key],
                self.params[voucher]['ema_span']
            )
            
        traderData[vol_key].append(volatility)
        
        if len(traderData[vol_key]) < self.params[voucher]['std_window']:
            return None, None

        if len(traderData[vol_key]) > self.params[voucher]['std_window']:
            traderData[vol_key].pop(0)
        
        # Use EMA as the mean for z-score calculation
        vol_z_score = (volatility - traderData[ema_key]) / np.std(traderData[vol_key])
        
        if vol_z_score >= self.params[voucher]['zscore_threshold']:
            if voucher_position != -self.LIMIT[voucher]:
                target_voucher_position = -self.LIMIT[voucher]
                if len(voucher_order_depth.buy_orders) > 0:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(target_voucher_position - voucher_position)
                    quantity = min(
                        target_quantity,
                        abs(voucher_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(voucher, best_bid, -quantity)], []
                    else:
                        return [Order(voucher, best_bid, -quantity)], [Order(voucher, best_bid - 1, -quote_quantity)]
        
        elif vol_z_score <= -self.params[voucher]['zscore_threshold']:
            if voucher_position != self.LIMIT[voucher]:
                target_voucher_position = self.LIMIT[voucher]
                if len(voucher_order_depth.sell_orders) > 0:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(target_voucher_position - voucher_position)
                    quantity = min(
                        target_quantity,
                        abs(voucher_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(voucher, best_ask, quantity)], []
                    else:
                        return [Order(voucher, best_ask, quantity)], [Order(voucher, best_ask + 1, quote_quantity)]

        return None, None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        conversions = 0

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_rainforest_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_kelp_orders(
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["kelp_min_edge"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket1_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        spread1_orders = self.spread1_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket1_position,
            traderObject[Product.SPREAD1],
        )
        if spread1_orders != None:
            result[Product.CROISSANT] = spread1_orders[Product.CROISSANT]
            result[Product.JAM] = spread1_orders[Product.JAM]
            result[Product.DJEMBE] = spread1_orders[Product.DJEMBE]
            result[Product.PICNIC_BASKET1] = spread1_orders[Product.PICNIC_BASKET1]

        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket2_position = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )
        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.SPREAD2],
        )
        if spread2_orders != None:
            if Product.CROISSANT in result:
                result[Product.CROISSANT].extend(spread2_orders[Product.CROISSANT])
            else:
                result[Product.CROISSANT] = spread2_orders[Product.CROISSANT]

            if Product.JAM in result:
                result[Product.JAM].extend(spread2_orders[Product.JAM])
            else:
                result[Product.JAM] = spread2_orders[Product.JAM]

            result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]

        traderData = jsonpickle.encode(traderObject)
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)

        if Product.MAGNIFICENT_MACARONS in self.params and Product.MAGNIFICENT_MACARONS in state.order_depths:
            # Initialize trader object for MAGNIFICENT_MACARONS if not exists
            if "MAGNIFICENT_MACARONS" not in traderObject:
                traderObject["MAGNIFICENT_MACARONS"] = {
                    "curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"], 
                    "volume_history": [], 
                    "optimized": False,
                    "price_history": [],
                    "sugar_history": [],
                    "sunlight_history": [],
                    "regression_coefficients": {
                        "sugar": 3.5,
                        "sunlight": -2.0,
                        "transport": -5.0,
                        "export": -0.8,
                        "import": 1.2,
                        "intercept": 100.0
                    }
                }
            
            position = (
                state.position[Product.MAGNIFICENT_MACARONS]
                if Product.MAGNIFICENT_MACARONS in state.position
                else 0
            )
            
            # Get observation and market data
            observation = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
            order_depth = state.order_depths[Product.MAGNIFICENT_MACARONS]

            sunlight_index = observation.sunlightIndex

            if sunlight_index < self.params["CSI"]["threshold"]:
                # Buy and hold strategy
                buy_hold_orders = self.buy_and_hold_strategy(
                    state, 
                    Product.MAGNIFICENT_MACARONS, 
                    position, 
                    self.LIMIT[Product.MAGNIFICENT_MACARONS]
                )
                result[Product.MAGNIFICENT_MACARONS] = buy_hold_orders

            else:
            
                # Store market data for reference
                if len(order_depth.buy_orders) > 0:
                    traderObject["MAGNIFICENT_MACARONS"]["best_bid"] = max(order_depth.buy_orders.keys())
                if len(order_depth.sell_orders) > 0:
                    traderObject["MAGNIFICENT_MACARONS"]["best_ask"] = min(order_depth.sell_orders.keys())

                
                
                # Calculate implied prices
                implied_bid, implied_ask = self.magnificent_macarons_implied_bid_ask(observation)
                
                # Store external factors for prediction model
                traderObject["MAGNIFICENT_MACARONS"]["sugar_history"].append(observation.sugarPrice)
                traderObject["MAGNIFICENT_MACARONS"]["sunlight_history"].append(observation.sunlightIndex)
                
                # Keep only last 50 data points
                if len(traderObject["MAGNIFICENT_MACARONS"]["sugar_history"]) > 50:
                    traderObject["MAGNIFICENT_MACARONS"]["sugar_history"].pop(0)
                if len(traderObject["MAGNIFICENT_MACARONS"]["sunlight_history"]) > 50:
                    traderObject["MAGNIFICENT_MACARONS"]["sunlight_history"].pop(0)
                
                # Predict fair value based on external factors
                # predicted_fair_value = (
                #     traderObject["MAGNIFICENT_MACARONS"]["regression_coefficients"]["intercept"] +
                #     traderObject["MAGNIFICENT_MACARONS"]["regression_coefficients"]["sugar"] * observation.sugarPrice +
                #     traderObject["MAGNIFICENT_MACARONS"]["regression_coefficients"]["sunlight"] * observation.sunlightIndex +
                #     traderObject["MAGNIFICENT_MACARONS"]["regression_coefficients"]["transport"] * observation.transportFees +
                #     traderObject["MAGNIFICENT_MACARONS"]["regression_coefficients"]["export"] * observation.exportTariff +
                #     traderObject["MAGNIFICENT_MACARONS"]["regression_coefficients"]["import"] * observation.importTariff
                # )

                predicted_fair_value = self.predict_fair_value(observation)
                
                # Calculate foreign midpoint
                foreign_mid = (observation.askPrice + observation.bidPrice) / 2
                
                # Calculate market midpoint
                market_mid = (traderObject["MAGNIFICENT_MACARONS"].get("best_bid", 0) + 
                            traderObject["MAGNIFICENT_MACARONS"].get("best_ask", 0)) / 2
                if market_mid == 0:
                    market_mid = (implied_bid + implied_ask) / 2
                
                # Store price history for volatility calculation
                traderObject["MAGNIFICENT_MACARONS"]["price_history"].append(market_mid)
                if len(traderObject["MAGNIFICENT_MACARONS"]["price_history"]) > 20:
                    traderObject["MAGNIFICENT_MACARONS"]["price_history"].pop(0)
                
                # Calculate volatility
                volatility = 0
                if len(traderObject["MAGNIFICENT_MACARONS"]["price_history"]) > 5:
                    volatility = np.std(traderObject["MAGNIFICENT_MACARONS"]["price_history"])
                
                # Determine position bias based on storage costs and predicted price movement
                position_bias = 0
                if len(traderObject["MAGNIFICENT_MACARONS"]["price_history"]) > 1:
                    recent_movement = traderObject["MAGNIFICENT_MACARONS"]["price_history"][-1] - traderObject["MAGNIFICENT_MACARONS"]["price_history"][0]
                    expected_holding_time = 10  # Default assumption
                    storage_cost = 0.1 * expected_holding_time
                    
                    if recent_movement < storage_cost:
                        position_bias = -1  # Bias toward short positions
                    elif recent_movement > storage_cost * 2:
                        position_bias = 1   # Bias toward long positions
                
                # Optimize conversions based on position and price differentials
                if position > 0:  # We're long, consider selling to Pristine Cuisine
                    if implied_bid > market_mid:
                        conversions = min(position, 10)  # Convert up to 10 units
                elif position < 0:  # We're short, consider buying from Pristine Cuisine
                    if implied_ask < market_mid:
                        conversions = max(position, -10)  # Convert up to 10 units
                
                # Adjust position after conversion
                position -= conversions
                
                # Calculate dynamic edge based on market conditions
                base_edge = self.magnificent_macarons_adap_edge(
                    state.timestamp,
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"],
                    position,
                    traderObject
                )
                
                # Adjust edge based on volatility and fair value deviation
                volatility_adjustment = min(volatility * 0.5, 2.0)  # Cap at 2.0
                
                fair_value_deviation = 0
                if market_mid > 0:
                    fair_value_deviation = abs(market_mid - predicted_fair_value) / market_mid
                
                # Apply position bias to edge
                position_adjustment = abs(position) * 0.005  # 0.5% penalty per position unit

                fair_value_adjustment = min(fair_value_deviation * 10, 1.5)  # Cap at 1.5
                
                # Calculate final edge
                # final_edge = base_edge + volatility_adjustment + fair_value_adjustment + position_adjustment
                min_edge = self.params[Product.MAGNIFICENT_MACARONS]["make_min_edge"]
                final_edge = max(
                    base_edge * 0.75 + (volatility * 0.3) + (fair_value_deviation * 5),
                    min_edge  # Ensure profitability
                )
                final_edge = max(final_edge, self.params[Product.MAGNIFICENT_MACARONS]["min_edge"])
                
                # Execute arbitrage (taking liquidity)
                take_orders, buy_order_volume, sell_order_volume = self.magnificent_macarons_arb_take(
                    order_depth,
                    observation,
                    final_edge,
                    position
                )
                
                # Adjust order sizes based on position bias
                if position_bias != 0:
                    adjusted_orders = []
                    for order in take_orders:
                        if (position_bias > 0 and order.quantity > 0) or (position_bias < 0 and order.quantity < 0):
                            # Increase size for orders aligned with bias (but respect position limits)
                            if order.quantity > 0:
                                max_buy = self.LIMIT[Product.MAGNIFICENT_MACARONS] - position
                                order.quantity = min(int(order.quantity * 1.5), max_buy)
                            else:
                                max_sell = self.LIMIT[Product.MAGNIFICENT_MACARONS] + position
                                order.quantity = max(int(order.quantity * 1.5), -max_sell)
                        adjusted_orders.append(order)
                    take_orders = adjusted_orders
                
                # Execute market making (providing liquidity)
                make_orders, _, _ = self.magnificent_macarons_arb_make(
                    order_depth,
                    observation,
                    position,
                    final_edge,
                    buy_order_volume,
                    sell_order_volume
                )
                
                # Combine all orders
                result[Product.MAGNIFICENT_MACARONS] = take_orders + make_orders

                trader_data = jsonpickle.encode(traderObject)
                conversions = 0
                logger.flush(state, result, conversions, trader_data)
        
        if "volcanic_rock_data" not in traderObject:
            traderObject["volcanic_rock_data"] = {
                "prev_volcanic_rock_price": 0
            }
            for voucher in self.VOUCHERS:
                traderObject["volcanic_rock_data"][f"prev_{voucher}_price"] = 0
                traderObject["volcanic_rock_data"][f"past_{voucher}_vol"] = []
                traderObject["volcanic_rock_data"][f"ema_{voucher}_vol"] = self.params[voucher]['initial_mean_volatility']
        
        # Get positions
        volcanic_rock_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        voucher_positions = {voucher: state.position.get(voucher, 0) for voucher in self.VOUCHERS}
        
        # Get spot price and update volatility
        spot_price = self.get_spot_price(state)
        tte = self.get_time_to_expiry(state.timestamp)
        
        # Process vouchers
        voucher_orders = {}
        deltas = {}
        
        for voucher in self.VOUCHERS:
            if voucher not in state.order_depths:
                continue
                
            order_depth = state.order_depths[voucher]
            position = voucher_positions[voucher]
            
            # Get mid price and calculate IV
            mid_price = self.get_voucher_mid_price(order_depth, traderObject["volcanic_rock_data"], voucher)
            iv = self.safe_implied_vol(mid_price, spot_price, 
                                     self.params[voucher]['strike'], tte, voucher)
            
            if iv is None:
                continue
                
            self.current_iv[voucher] = iv
            delta = BlackScholes.delta(spot_price, self.params[voucher]['strike'], tte, iv)
            deltas[voucher] = delta
            
            # Generate voucher orders
            take_orders, make_orders = self.volcanic_rock_voucher_orders(
                voucher, order_depth, position, traderObject["volcanic_rock_data"], iv)
            
            if take_orders or make_orders:
                voucher_orders[voucher] = take_orders + make_orders if make_orders else take_orders
                result[voucher] = voucher_orders[voucher]
        
        # Update EMA-based volatility for moneyness adjustment
        for voucher in self.VOUCHERS:
            if f"ema_{voucher}_vol" in traderObject["volcanic_rock_data"]:
                self.ema_iv[voucher] = traderObject["volcanic_rock_data"][f"ema_{voucher}_vol"]
        
        # Adjust current IV based on moneyness using EMA as baseline
        self.adjust_vol_for_moneyness(spot_price)
        
        # Gamma-adjusted hedging
        if self.should_rehedge(state.timestamp, spot_price):
            target_delta = self.calculate_gamma_adjusted_delta(
                spot_price, 
                {**voucher_positions, **{k: sum(o.quantity for o in v) for k,v in voucher_orders.items()}},
                tte
            )
            hedge_orders = self.execute_hedge(state, target_delta)
            if hedge_orders:
                result[Product.VOLCANIC_ROCK] = hedge_orders
        
        trader_data = ""
        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, trader_data)
        return result, 0, traderData