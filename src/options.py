import numpy as np
import itertools
from scipy import stats

class BasketOption:
    '''Functions for a European basket call option.'''
    
    def __init__(self, weights, prices, vol, corr, strike, time, rate):
        '''           
        Parameters
        ----------
        weights : ndarray
            Floats representing weights of the underlying assets in the basket. 
            Should sum to 1, be 1-D, and be of length equal to the length of prices.
        prices : ndarray
            Floats representing the asset prices at time zero. Should be 1-D, and same
            length as prices.
        vol : float
            The volatility of the assets. N.B. the Levy formula assumes homogeneous asset
            volatility.
        corr : ndarray
            Correlation matrix of the assets. Should be of shape (n,n), where n is the
            number of assets.
        strike : float
            Strike price.
        time : float
            Time to maturity.
        rate : float
            Riskless interest rate.
        '''
        self.weights = weights
        self.prices = prices
        self.vol = vol
        self.corr = corr
        self.strike = strike
        self.time = time
        self.rate = rate
    
        if not len(weights) == len(prices) == len(corr):
            raise ValueError('Number of weights, prices, corr rows should be equal')
            
        if abs(1-sum(weights))>0.01:
            raise ValueError('The weights must cumulatively sum to 1.0')
            
    def get_levy_price(self):
        """
        Use the Levy formula to approximate price of option.
        """
    
        discount = np.exp(-self.rate*self.time)
    
        # First moment of T-forward prices (also the basket T-forward price)
        m1 = np.sum(self.weights * self.prices * discount)

        # Second moment of T-forward prices
        w_ij, f_ij = [list(map(lambda x: np.product(x), list(itertools.product(q, q)))) 
                          for q in [self.weights, self.prices * discount]]
        m2 = np.sum(np.array(w_ij) * np.array(f_ij)
                    * np.exp(self.corr.flatten() * self.vol**2 * self.time))
    
        vol_basket = ( self.time**(-1) * np.log(m2 / m1**2) )**(0.5)
    
        # Parameters of the price formula
        d1 = np.log(m1 / self.strike)/(vol_basket * self.time**(0.5))\
                + (vol_basket * self.time**(0.5))/2
        d2 = d1 - vol_basket * self.time**(0.5)

        # Levy formula for basket call option price
        self.levy_price = discount * (m1 * stats.norm.cdf(d1) - self.strike * stats.norm.cdf(d2))

        return self.levy_price

    def get_mc_price(self, n_paths=10000):
        """
        Use Monte Carlo to estimate price of option.

        Parameters
        ----------
        n_paths : int, optional
            Number of asset price paths to simulate
            Default is 10,000
        """

        rn = stats.multivariate_normal(np.zeros(len(self.weights)), cov=self.corr).rvs(size=n_paths)

        wt = self.time**0.5 * rn
        asset_prices = self.prices * np.exp((self.rate-0.5*self.vol**2)*self.time + self.vol * wt)
        
        # Check if basket option or one-asset option
        if len(self.weights)>1:
            payoffs = (np.sum(self.weights*asset_prices, axis=1)-self.strike).clip(0)
        else:
            payoffs = (asset_prices - self.strike).clip(0)

        self.mc_price = np.mean(payoffs)

        return self.mc_price

    def get_bs_price(self):
        
        if len(self.weights) > 1.:
            raise ValueError('Number of assets seems to be >1. Use Levy price instead')
            
        d1 = 1/(self.vol*(self.time)**0.5)*(np.log(self.prices/self.strike)\
                                            + (self.rate+(0.5*self.vol**2))*self.time)
        d2 = d1 - self.vol*self.time**0.5
        
        self.bs_price = stats.norm.cdf(d1)*self.prices\
                        - stats.norm.cdf(d2)*self.strike*np.exp(-self.rate*self.time)
        
        return self.bs_price

    def get_ann_price(self, model):
        '''
        Use network model to estimate option price. This function allows estimation of one-asset 
        options with a network trained to estimate 4-asset basket option prices.        
        '''

        config = model.get_config() # Returns pretty much every information about your model
        n_assets = int((config["layers"][0]["config"]["batch_input_shape"][1] - 4)/2)
        
        # Pad with the single price if trying to estimate single-asset option price
        p = np.pad(self.prices, (0, int(max(n_assets-len(self.prices),0))), constant_values=self.prices[0])
        # If single-asset option, assign very small weights to the padded assets
        if len(self.weights) < n_assets: 
            # Assume we're using a multi-asset model on a one-asset option
            w = np.pad(self.weights, (0, int(max(n_assets-len(self.prices),0))), constant_values=0.001)
            w[0] = 1-(n_assets-1)*0.001
        else:
            w = self.weights.copy()
        
        price_weights = np.append(p, w)
        X = np.append(price_weights, [self.strike, self.time, self.vol, self.rate])

        self.ann_price = model.predict(X[np.newaxis,:])[0][0]

        return self.ann_price
