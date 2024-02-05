from AgentBasedModel.simulator import SimulatorInfo
from AgentBasedModel.model.plots import plot_feature
import pandas as pd

class BidAskSpread():
    def __init__(
            self,
            info : SimulatorInfo
    ):
        self.spreads = info.spreads
        self.exchanges = info.exchanges

    def get_feature_name(self):
        return f"bid ask spread"
    
    def compile_feature(self):
        self.bid_ask_spreads = {idx: list() for idx in range(len(self.exchanges))}
        for exchange in self.exchanges:
            for i in range(len(self.spreads[exchange])):
                self.bid_ask_spreads[exchange].append(round((self.spreads[exchange][i]['ask'] - self.spreads[exchange][i]['bid']) / (self.spreads[exchange][i]['ask'] + self.spreads[exchange][i]['bid']), 3))
        return self.bid_ask_spreads
    

class BidAskVolumeImbalance():
    def __init__(
            self,
            info : SimulatorInfo,
            depth : int = 1 # depth of orderbook
    ):
        self.orderbook_history = info.orderbook_history
        self.exchanges = info.exchanges
        self.depth = depth 

    def get_feature_name(self):
        return f"bid ask volume imbalance at depth {self.depth}"
    
    def compile_feature(self):
        self.bid_ask_volume_imbalance = {idx: list() for idx in range(len(self.exchanges))}
        for exchange in range(len(self.exchanges)):
            for tick in range(len(self.orderbook_history[exchange])):
                bid_qty = self.orderbook_history[exchange][tick]['bid'][0]['qty']
                ask_qty = self.orderbook_history[exchange][tick]['ask'][0]['qty']
                self.bid_ask_volume_imbalance[exchange].append(round((bid_qty - ask_qty) / (bid_qty + ask_qty), 2))
        return self.bid_ask_volume_imbalance


class TradeVolumeImbalance():
    def __init__(
            self,
            info : SimulatorInfo,
            n_iter : int = 10 # number of iterations before the current according to which trades are counted  
    ):
        self.trades_history = info.trades_history
        self.exchanges = info.exchanges
        self.n_iter = n_iter
    
    def get_feature_name(self):
        return f"trade imbalance of {self.n_iter} previous iterations"
    
    def compile_feature(self):
        trade_imbalance = {idx: list() for idx in range(len(self.exchanges))}
        self.trade_volume_imbalance = {idx: list() for idx in range(len(self.exchanges))}
        for exchange in range(len(self.exchanges)):
                for tick in range(len(self.trades_history[exchange])):
                    if len(self.trades_history[exchange][tick]) != 0:
                        df_tick = pd.DataFrame(self.trades_history[exchange][tick])
                        
                        trade_imbalance[exchange].append(df_tick[df_tick['side'] == 'buy']['qty'].sum() - df_tick[df_tick['side'] == 'sell']['qty'].sum())
                    else:
                        trade_imbalance[exchange].append(0)
                    
                    self.trade_volume_imbalance[exchange].append(round(sum(trade_imbalance[exchange][max(0, tick - self.n_iter) : tick]), 2))
        return self.trade_volume_imbalance
    