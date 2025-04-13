from typing import Dict, List
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import jsonpickle
import numpy as np
import math

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

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = {
                "RAINFOREST_RESIN": {
                    "fair_value": 10000,
                    "take_width": 1,
                    "clear_width": 0,
                    # for making
                    "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
                    "join_edge": 2,  # joins orders within this edge
                    "default_edge": 4,
                    "soft_position_limit": 40,
                },
                "KELP": {
                    "take_width": 1,
                    "clear_width": 0,
                    "prevent_adverse": True,
                    "adverse_volume": 20,
                    "reversion_beta": 0.229,
                    "disregard_edge": 1,
                    "join_edge": 0,
                    "default_edge": 1,
                },
                "SQUID_INK": {
                    "adverse_volume": 20,
                    "clear_width": 0,
                    "disregard_edge": 1,
                    "join_edge": 4,
                    "default_edge": 2,
                    "prevent_adverse": True,
                    # New parameters for deviation metric:
                    "deviation_window": 300,  # number of ticks to use for moving average (or EMA window)
                    "deviation_threshold": 0.009,  # 1% deviation threshold (adjust as needed)
                    "reversion_factor": 0.5,  # how strong is the reversion (e.g. 50% of the deviation)
                    "take_width": 2  # as before
                },
            }
        self.params = params

        self.LIMIT = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: list,
        order_depth,
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

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)  # max amt to buy
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
                    quantity = min(best_bid_amount, position_limit + position)  # max amt to sell
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
        orders: list,
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
        orders: list,
        order_depth,
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

    def kelp_fair_value(self, order_depth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params["KELP"]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params["KELP"]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params["KELP"]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def squid_fair_value(self, order_depth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params["SQUID_INK"]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params["SQUID_INK"]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("squid_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squid_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # Update a running moving average. We use a simple exponential moving average (EMA)
            window = self.params["SQUID_INK"].get("deviation_window", 50)
            if "squid_moving_average" not in traderObject:
                traderObject["squid_moving_average"] = mmmid_price
            else:
                # alpha = 2 / (window+1) gives weight to the new price
                alpha = 2.0 / (window + 1)
                traderObject["squid_moving_average"] = (
                    alpha * mmmid_price + (1 - alpha) * traderObject["squid_moving_average"]
                )
            moving_average = traderObject["squid_moving_average"]

            # Calculate percentage deviation from the moving average
            deviation = (mmmid_price - moving_average) / moving_average

            # Apply reversion: if the deviation exceeds the threshold, adjust the fair price to favor reversion trades.
            threshold = self.params["SQUID_INK"].get("deviation_threshold", 0.01)
            reversion_factor = self.params["SQUID_INK"].get("reversion_factor", 0.5)
            if abs(deviation) > threshold:
                # For example, if price is above average, expect a downward reversion and vice versa.
                pred_returns = - reversion_factor * deviation
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            traderObject["squid_last_price"] = mmmid_price
            return fair
        return None


    def take_orders(
        self,
        product: str,
        order_depth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (list, int, int):
        orders: list = []
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
        order_depth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (list, int, int):
        orders: list = []
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
        order_depth,
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
        orders: list = []
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
            elif position < -soft_position_limit:
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

    def run(self, state):
        trader_data = "Class"
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if "RAINFOREST_RESIN" in self.params and "RAINFOREST_RESIN" in state.order_depths:
            resin_position = (
                state.position["RAINFOREST_RESIN"]
                if "RAINFOREST_RESIN" in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                "RAINFOREST_RESIN",
                state.order_depths["RAINFOREST_RESIN"],
                self.params["RAINFOREST_RESIN"]["fair_value"],
                self.params["RAINFOREST_RESIN"]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                "RAINFOREST_RESIN",
                state.order_depths["RAINFOREST_RESIN"],
                self.params["RAINFOREST_RESIN"]["fair_value"],
                self.params["RAINFOREST_RESIN"]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                "RAINFOREST_RESIN",
                state.order_depths["RAINFOREST_RESIN"],
                self.params["RAINFOREST_RESIN"]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params["RAINFOREST_RESIN"]["disregard_edge"],
                self.params["RAINFOREST_RESIN"]["join_edge"],
                self.params["RAINFOREST_RESIN"]["default_edge"],
                True,
                self.params["RAINFOREST_RESIN"]["soft_position_limit"],
            )
            result["RAINFOREST_RESIN"] = resin_take_orders + resin_clear_orders + resin_make_orders

        if "KELP" in self.params and "KELP" in state.order_depths:
            kelp_position = (
                state.position["KELP"]
                if "KELP" in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(state.order_depths["KELP"], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                "KELP",
                state.order_depths["KELP"],
                kelp_fair_value,
                self.params["KELP"]["take_width"],
                kelp_position,
                self.params["KELP"]["prevent_adverse"],
                self.params["KELP"]["adverse_volume"],
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                "KELP",
                state.order_depths["KELP"],
                kelp_fair_value,
                self.params["KELP"]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_orders(
                "KELP",
                state.order_depths["KELP"],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params["KELP"]["disregard_edge"],
                self.params["KELP"]["join_edge"],
                self.params["KELP"]["default_edge"],
            )
            result["KELP"] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        if "SQUID_INK" in self.params and "SQUID_INK" in state.order_depths:
            squid_position = (
                state.position["SQUID_INK"]
                if "SQUID_INK" in state.position
                else 0
            )
            squid_fair_value = self.squid_fair_value(state.order_depths["SQUID_INK"], traderObject)
            squid_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                "SQUID_INK",
                state.order_depths["SQUID_INK"],
                squid_fair_value,
                self.params["SQUID_INK"]["take_width"],
                squid_position,
                self.params["SQUID_INK"]["prevent_adverse"],
                self.params["SQUID_INK"]["adverse_volume"],
            )
            squid_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                "SQUID_INK",
                state.order_depths["SQUID_INK"],
                squid_fair_value,
                self.params["SQUID_INK"]["clear_width"],
                squid_position,
                buy_order_volume,
                sell_order_volume,
            )
            squid_make_orders, _, _ = self.make_orders(
                "SQUID_INK",
                state.order_depths["SQUID_INK"],
                squid_fair_value,
                squid_position,
                buy_order_volume,
                sell_order_volume,
                self.params["SQUID_INK"]["disregard_edge"],
                self.params["SQUID_INK"]["join_edge"],
                self.params["SQUID_INK"]["default_edge"],
            )
            result["SQUID_INK"] = squid_take_orders + squid_clear_orders + squid_make_orders

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, traderData