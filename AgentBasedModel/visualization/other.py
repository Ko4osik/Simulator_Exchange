from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math
# from AgentBasedModel.orders import Order, OrderIter, OrderList
# from AgentBasedModel.exchange import ExchangeAgent
import matplotlib.pyplot as plt
import pandas as pd


def plot_book_stat(
        info:    SimulatorInfo,
        idx:     int,
        stat:    str = 'quantity',
        rolling: int   = 1,
        figsize: tuple = (6, 6)
    ):
    """Lineplot ExchangeAgent`s Order Book chosen statistic

    :param info: SimulatorInfo instance
    :param idx: ExchangeAgent id
    :param stat: Order Book statistic to plot, defaults 'quantity'
    :param rolling: MA applied to list, defaults to 1
    :param figsize: figure size, defaults to (6, 6)
    """
    plt.figure(figsize=figsize)
    plt.title(
             f'Exchange{idx}: Order Book {stat} by order type ' if rolling == 1
        else f'Exchange{idx}: Order Book {stat} by order type  (MA {rolling})'
    )
    plt.xlabel('Iterations')
    plt.ylabel(f'Order {stat}')

    iterations = range(rolling - 1, len(info.dividends[idx]))
    v_bid = math.rolling([v[stat]['bid'] for v in info.orders[idx]], rolling)
    v_ask = math.rolling([v[stat]['ask'] for v in info.orders[idx]], rolling)
    
    plt.plot(iterations, v_bid, label='bid', color='green')
    plt.plot(iterations, v_ask, label='ask', color='red')

    plt.legend()
    plt.show()


def print_book(info: SimulatorInfo, idx: int, n=5):
    val = pd.concat([
        pd.DataFrame({
            'Sell': [v.price for v in info.exchanges[idx].order_book['ask']],
            'Quantity': [v.qty for v in info.exchanges[idx].order_book['ask']]
            }).groupby('Sell').sum().reset_index().head(n),
        pd.DataFrame({
            'Buy': [v.price for v in info.exchanges[idx].order_book['bid']],
            'Quantity': [v.qty for v in info.exchanges[idx].order_book['bid']]
        }).groupby('Buy').sum().reset_index().sort_values('Buy', ascending=False).head(n)
    ])
    print(val[['Buy', 'Sell', 'Quantity']].fillna('').to_string(index=False))


def plot_book(info: SimulatorInfo, idx: int, bins=50, figsize=(6, 6)):
    bid = list()
    for order in info.exchanges[idx].order_book['bid']:
        for p in range(order.qty):
            bid.append(order.price)

    ask = list()
    for order in info.exchanges[idx].order_book['ask']:
        for p in range(order.qty):
            ask.append(order.price)

    plt.figure(figsize=figsize)
    plt.title('Order book')
    plt.hist(bid, label='bid', color='green', bins=bins)
    plt.hist(ask, label='ask', color='red', bins=bins)
    plt.show()

def order_book_snapshot(info : SimulatorInfo, idx: int, bins=100, figsize = (6, 6)):
        exchange = info.exchanges[idx]
        plt.figure(figsize = figsize)
        bids = exchange.order_book['bid'].to_list()
        prices_bid = [bid['price'] for bid in bids]
        amounts_bid = [bid['qty'] for bid in bids]

        asks = exchange.order_book['ask'].to_list()
        prices_ask = [ask['price'] for ask in asks]
        amounts_ask = [ask['qty'] for ask in asks]

        plt.hist(prices_bid, bins=bins, color = 'green')
        plt.hist(prices_ask, bins=bins, color = 'red')

        plt.xlabel('Price')
        plt.ylabel('Amount')
        plt.title('OrderBookSnapshot')

        plt.show()

