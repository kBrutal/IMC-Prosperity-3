from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import jsonpickle


class Product:
    KELP = 'KELP'
    RAINFOREST_RESIN = 'RAINFOREST_RESIN'


PARAMS = {'KELP': {'take_width': 2, 'clear_width': 0, 'prevent_adverse': 
    True, 'adverse_volume': 10, 'reversion_beta': -0.13208468537510837,
    'disregard_edge': 1, 'join_edge': 0, 'default_edge': 1,
    'soft_position_limit': 20}, 'RAINFOREST_RESIN': {'fair_value': 10000,
    'take_width': 4, 'clear_width': 0, 'disregard_edge': 1, 'join_edge': 2,
    'default_edge': 4, 'soft_position_limit': 7}}


class Trader:

    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.KELP: 20, Product.RAINFOREST_RESIN: 20}
        self.price_history = {}
        self.max_history_length = 20
        self.triangular_window = 7

    def _triangular_filter(self, prices):
        """Apply triangular filter to the last few prices"""
        if len(prices) < 3:
            return prices[-1] if prices else 0
        weights = np.linspace(1, 0, self.triangular_window)
        weights /= weights.sum()
        recent_prices = prices[-self.triangular_window:]
        return np.dot(recent_prices, weights[-len(recent_prices):])

    def take_best_orders(self, product: str, fair_value: int, take_width:
        float, orders: List[Order], order_depth: OrderDepth, position: int,
        buy_order_volume: int, sell_order_volume: int, prevent_adverse:
        bool=False, adverse_volume: int=0) ->tuple[int, int]:
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
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
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask:
        int, position: int, buy_order_volume: int, sell_order_volume: int
        ) ->tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width:
        int, orders: List[Order], order_depth: OrderDepth, position: int,
        buy_order_volume: int, sell_order_volume: int) ->tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.
                buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity))
                    )
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in
                order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def dynamic_fair_value(self, product: str, order_depth: OrderDepth,
        traderObject) ->float:
        """Calculate fair value based on order book data with possible mean reversion"""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders
            ) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys
                () if abs(order_depth.sell_orders[price]) >= self.params[
                product]['adverse_volume']]
            filtered_bid = [price for price in order_depth.buy_orders.keys(
                ) if abs(order_depth.buy_orders[price]) >= self.params[
                product]['adverse_volume']]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get(f'{product}_last_price', None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject[f'{product}_last_price']
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get(f'{product}_last_price', None) is not None:
                last_price = traderObject[f'{product}_last_price']
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[product][
                    'reversion_beta']
                fair = mmmid_price + mmmid_price * pred_returns
            else:
                fair = mmmid_price
            traderObject[f'{product}_last_price'] = mmmid_price
            return fair
        return self.params[product].get('fair_value', None)

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value:
        float, take_width: float, position: int, prevent_adverse: bool=
        False, adverse_volume: int=0) ->tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(product,
            fair_value, take_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume, prevent_adverse,
            adverse_volume)
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth,
        fair_value: float, clear_width: int, position: int,
        buy_order_volume: int, sell_order_volume: int) ->tuple[List[Order],
        int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(product
            , fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value:
        float, position: int, buy_order_volume: int, sell_order_volume: int,
        disregard_edge: float, join_edge: float, default_edge: float,
        manage_position: bool=False, soft_position_limit: int=0):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys(
            ) if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if
            price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair
            ) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair
            ) > 0 else None
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
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(product,
            orders, bid, ask, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        print('traderData: ' + (state.traderData if state.traderData else
            'None'))
        print('Observations: ' + str(state.observations))
        traderObject = {}
        if (state.traderData and state.traderData != '' and state.
            traderData != 'SAMPLE'):
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except:
                print('Failed to decode trader data')
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            if product not in self.params:
                continue
            position = state.position[product
                ] if product in state.position else 0
            best_bid = max(order_depth.buy_orders.keys()
                ) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()
                ) if order_depth.sell_orders else 0
            if best_bid and best_ask:
                current_mid = (best_bid + best_ask) / 2
            elif best_bid:
                current_mid = best_bid
            elif best_ask:
                current_mid = best_ask
            else:
                current_mid = traderObject.get(f'{product}_last_price',
                    self.params[product].get('fair_value', 10000))
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append(current_mid)
            self.price_history[product] = self.price_history[product][-self
                .max_history_length:]
            if product == Product.KELP:
                triangular_estimate = self._triangular_filter(self.
                    price_history[product])
                fair_value = self.dynamic_fair_value(product, order_depth,
                    traderObject)
                if fair_value is None:
                    fair_value = triangular_estimate
            else:
                fair_value = self.params[product]['fair_value']
                if len(self.price_history[product]) > 10:
                    deviation = current_mid - fair_value
                    adjustment = deviation * 0.4
                    fair_value += adjustment
            print(f'Fair value for {product}: {fair_value}')
            buy_order_volume = 0
            sell_order_volume = 0
            take_orders, buy_order_volume, sell_order_volume = (self.
                take_orders(product, order_depth, fair_value, self.params[
                product]['take_width'], position, self.params[product].get(
                'prevent_adverse', False), self.params[product].get(
                'adverse_volume', 0)))
            clear_orders, buy_order_volume, sell_order_volume = (self.
                clear_orders(product, order_depth, fair_value, self.params[
                product]['clear_width'], position, buy_order_volume,
                sell_order_volume))
            make_orders, _, _ = self.make_orders(product, order_depth,
                fair_value, position, buy_order_volume, sell_order_volume,
                self.params[product]['disregard_edge'], self.params[product
                ]['join_edge'], self.params[product]['default_edge'], True,
                self.params[product]['soft_position_limit'])
            result[product] = take_orders + clear_orders + make_orders
            print(f'Position for {product}: {position}')
            print(f'Orders for {product}: {len(result[product])}')
        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        return result, conversions, traderData
