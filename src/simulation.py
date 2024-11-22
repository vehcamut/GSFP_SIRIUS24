import numpy as np
import pandas as pd
from src.options import BasketOption

class SimulateBasketPrices:
    ''' 
    Simulate the prices of many European basket call options. 
    The model used to approximate the option price assumes equal 
    volatility of the assets.
    '''
    
    def __init__(self, n_assets, n_prices=100):
        '''
        Parameters
        ----------
        n_assets : int
            Number of assets in the basket
        n_prices : int, optional
            Number of prices to simulate
        '''
        self.n_assets = n_assets
        self.n_prices = n_prices

    def get_price_list(self, base_price):
        if len(base_price) == self.n_assets:
            rand_jitter = (np.random.randn(self.n_prices * self.n_assets)*10)\
                            .reshape((self.n_prices,self.n_assets))
            prices = rand_jitter + base_price
        else:
            prices = np.random.randn(self.n_prices)*10 + base_price
        
        return prices        

    def get_weight_list(self, weights):
        if weights is None:
            weights = (np.random.rand(self.n_prices * self.n_assets))\
                        .reshape((self.n_prices,self.n_assets))
            weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
        else:
            weights = np.tile(weights, (self.n_prices,1))
        
        return weights

    def get_df(self, values, prefix=None):
        df = pd.DataFrame(values)
        df.columns = list(map(lambda x: f'{prefix}_'+str(x), 
                                range(1, self.n_assets+1)))
        return df

    def simulate_prices(self, base_price, vol, corr, base_strike=None, weights=None, rate=0.):
        '''
        Parameters
        ----------
        base_price : ndarray, optional
            If len=1, average price of the simulated assets
            Otherwise, len should be equal to len(self.n_prices)
        vol : float
            Average volatility of the simulated assets
        corr : ndarray
            Correlation matrix of the assets
        base_strike : float, optional
            Average strike of the simulated assets
            Default is strike equal to price of first asset
        weights : ndarray, optional
            Weights of each asset in the basket. Default is random weighting
        rate : float, optional
            Risk-free interest rate
        '''
        
        if (len(base_price)>1) & (len(base_price)!=self.n_assets):
            raise ValueError('Number of prices should be 1 or equal to n_assets')

        prices = self.get_price_list(base_price)
        times = np.random.randint(1,16,self.n_prices)
        vols = np.abs((np.random.rand(self.n_prices)-0.5)*0.05 + vol)

        if base_strike is None:
            base_strike = base_price[0]
        strikes = base_strike * (0.5 + np.random.rand(self.n_prices))

        weights = self.get_weight_list(weights)
        basket_prices = []
        for i in range(self.n_prices):
            basket = BasketOption(weights[i], prices[i],
                                  vols[i], corr, strikes[i], 
                                  times[i].astype('float'), rate)
            basket_prices.append(basket.get_levy_price())
        
        prices_df = self.get_df(prices, prefix='Price')
        weights_df = self.get_df(weights, prefix='Weight')

        self.simulated_baskets = pd.DataFrame({
                                    'Strike' : strikes,
                                    'Maturity': times,
                                    'Volatility': vols,
                                    'Rate': rate*len(prices),
                                    'Basket_Price': basket_prices})

        self.simulated_baskets = prices_df.join(weights_df).join(self.simulated_baskets)

        return self.simulated_baskets