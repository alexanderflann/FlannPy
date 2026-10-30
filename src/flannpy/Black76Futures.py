
# A Python Library for Option Pricing and Sensitivities

# Options on Futures

import numpy as np
from scipy.stats import norm

# Black-76 Model for Options on Futures

class Black76Futures:

    # F             :   futures_price
    # E             :   strike_price
    # T             :   time_to_maturity
    # r             :   risk_free_rate
    # sigma         :   underlying_volatility
    # call / put    :   option_type

    def __init__(self, futures_price, strike_price, time_to_maturity, risk_free_rate, underlying_volatility, option_type):
        self.futures_price = futures_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.underlying_volatility = underlying_volatility
        self.option_type = option_type
    # Constructor which intialises the object's instance variables

    def __str__(self):
        return (
            f"Price : {self.price():>8.4f}\n"
            f"Delta : {self.delta():>8.4f}\n"
            f"Gamma : {self.gamma():>8.4f}\n"
            f"Rho   : {self.rho():>8.4f}\n"
            f"Theta : {self.theta():>8.4f}\n"
            f"Vega  : {self.vega():>8.4f}"
        )
    # String to return a summary

    def _d1(self):

        # d1 = d1 = ( ln(F/E) + 0.5 * sigma^2 * T ) / ( sigma * sqrt(T) )
        d1 = ( np.log(self.futures_price / self.strike_price) + 0.5 * self.underlying_volatility**2 * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d1

    def _d2(self):

        # d2 = d1 - sigma * sqrt(T)
        d2 = self._d1() - self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return d2

    def price(self):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # C = exp(-r*T) * ( F * Phi(d1) - E * Phi(d2) )
            price = np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.futures_price * norm.cdf(self._d1(), 0, 1) - self.strike_price * norm.cdf(self._d2(), 0, 1) )
        else:

            # P = exp(-r*T) * ( E * Phi(-d2) - F * Phi(-d1) )
            price = np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.strike_price * norm.cdf(-self._d2(), 0, 1) - self.futures_price * norm.cdf(-self._d1(), 0, 1) )
        return price

    def delta(self):
    # Delta measures how the option price will change with a change in the underlying asset price
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):
            # Delta = exp(-r*T) * Phi(d1)
            delta = np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._d1(), 0, 1)
        else:
            # Delta = - exp(-r*T) * Phi(-d1)
            delta = - np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(-self._d1(), 0, 1)
        return delta
    
    def gamma(self):
    # Gamma measures the rate of change of Delta

        # Gamma = ( exp(-r* T) * phi(d1) ) / ( F * sigma * sqrt(T) ) 
        gamma = np.exp(-self.risk_free_rate * self.time_to_maturity) * ( (norm.pdf(self._d1(), 0, 1)) / (self.futures_price * self.underlying_volatility * np.sqrt(self.time_to_maturity)) )
        return gamma

    def rho(self):
    # Rho measures how the option price will change with a change in the risk-free interest rate
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):
            # Rho = - T * C
            rho = - self.time_to_maturity * ( np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.futures_price * norm.cdf(self._d1(), 0, 1) - self.strike_price * norm.cdf(self._d2(), 0, 1) ) )
        else:
    
            # Rho = - T * P
            rho = - self.time_to_maturity * ( np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.strike_price * norm.cdf(-self._d2(), 0, 1) - self.futures_price * norm.cdf(-self._d1(), 0, 1) ) )
        return rho

    def theta(self):
    # Theta measures how the option price will change as time passes
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # Theta = - F * exp(-r*T) * phi(d1) * sigma / ( 2 * sqrt(T) ) + r * C
            theta = - self.futures_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.pdf(self._d1(), 0, 1) * ( self.underlying_volatility / (2 * np.sqrt(self.time_to_maturity)) ) + self.risk_free_rate * ( np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.futures_price * norm.cdf(self._d1(), 0, 1) - self.strike_price * norm.cdf(self._d2(), 0, 1) ) )
        else:

            # Theta = - F * exp(-r*T) * phi(d1) * sigma / ( 2 * sqrt(T) ) + r * P
            theta = - self.futures_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.pdf(self._d1(), 0, 1) * ( self.underlying_volatility / (2 * np.sqrt(self.time_to_maturity)) ) + self.risk_free_rate * ( np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.strike_price * norm.cdf(-self._d2(), 0, 1) - self.futures_price * norm.cdf(-self._d1(), 0, 1) ) )
        return theta

    def vega(self):
    # Vega measures how the option price will change with a change in the underlying asset volatility

        # Vega = F * exp(-r*T) * sqrt(T) * phi(d1)
        vega = self.futures_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * np.sqrt(self.time_to_maturity) * norm.pdf(self._d1(), 0, 1)
        return vega
