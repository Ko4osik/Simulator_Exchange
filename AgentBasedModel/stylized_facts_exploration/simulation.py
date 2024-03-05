from AgentBasedModel import *
from AgentBasedModel.extra import *
from AgentBasedModel.visualization import (
    plot_price,
    plot_price_fundamental
)
from random import randint

from AgentBasedModel.model.features import BidAskVolumeImbalance



def make_simulation(n_random, n_fundamentalist, n_chartist, n_marketmaker, event = None, n_iters = 2000):
    risk_free_rate = 5e-4
    price = 100
    dividend = price * risk_free_rate
    ExchangeAgent.id = 0
    # Initialize objects
    assets = [
        Stock(dividend) for _ in range(1)
    ]
    exchanges = [
        ExchangeAgent(assets[0], risk_free_rate) for i in range(1)  # single asset
    ]
    traders = [
        *[Random(exchanges[randint(0, 0)])         for _ in range(n_random)],
        *[Fundamentalist(exchanges[randint(0, 0)]) for _ in range(n_fundamentalist)],
        *[Chartist2D(exchanges)                    for _ in range(n_chartist)],
        *[MarketMaker2D(exchanges)                 for _ in range(n_marketmaker)]
    ]

    # Run simulation
    simulator = Simulator(**{
        'assets': assets,
        'exchanges': exchanges,
        'traders': traders,
        'events': [MarketPriceShock(0, int(n_iters * 0.3), -20)] * (event == 'MarketPriceShockDown') + 
            [MarketPriceShock(0, int(n_iters * 0.3), 20)] * (event == 'MarketPriceShockUp') + 
            [LiquidityShock(0, int(n_iters * 0.3), 100)] * (event == 'LiquidityShockSell') +
            [LiquidityShock(0, int(n_iters * 0.3), -100)] * (event == 'LiquidityShockBuy') + 
            [MarketMakerIn(0, int(n_iters * 0.3))] * (event == 'MarketMakerIn' or event == 'MarketMakerInOut') + 
            [MarketMakerOut(0, int(n_iters * 0.7))] * (event == 'MarketMakerInOut') + 
            [TransactionCost(0, int(n_iters * 0.3), 0.01)] * (event == 'TransactionCost') +
            [InformationShock(0, int(n_iters * 0.3), 1)] * (event == 'InformationShock')
    })

    info= simulator.info
    simulator.simulate(n_iters, silent=False)
    return info
