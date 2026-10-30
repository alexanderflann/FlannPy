
# A Python Library for Option Pricing and Sensitivities

# Exchange Options

import numpy as np
from scipy.stats import norm

# Margrabe Model for Exchange Options

class MargrabeExchange:

    # S_1       :   underlying_price_1
    # S_2       :   underlying_price_2
    # T         :   time_to_maturity
    # q_1       :   dividend_yield_1
    # q_2       :   dividend_yield_2
    # sigma_1   :   underlying_volatility_1
    # sigma_2   :   underlying_volatility_2
    # rho       :   correlation

    def __init__(self, underlying_price_1, underlying_price_2, time_to_maturity, dividend_yield_1, dividend_yield_2, underlying_volatility_1, underlying_volatility_2, correlation):
        self.underlying_price_1 = underlying_price_1
        self.underlying_price_2 = underlying_price_2
        self.time_to_maturity = time_to_maturity
        self.dividend_yield_1 = dividend_yield_1
        self.dividend_yield_2 = dividend_yield_2
        self.underlying_volatility_1 = underlying_volatility_1
        self.underlying_volatility_2 = underlying_volatility_2
        self.correlation = correlation
    # Constructor which intialises the object's instance variables
    
    def __str__(self):
        return (
            f"Price             : {self.price():>8.4f}\n"
            f"Correlation Vega  : {self.chi():>8.4f}\n"
            f"Delta1            : {self.delta1():>8.4f}\n"
            f"Delta2            : {self.delta2():>8.4f}\n"
            f"Gamma1            : {self.gamma1():>8.4f}\n"
            f"Gamma2            : {self.gamma2():>8.4f}\n"
            f"Cross-Gamma       : {self.gamma12():>8.4f}\n"
            f"Theta             : {self.theta():>8.4f}\n"
            f"Vega              : {self.vega():>8.4f}"
        )
    # String to return a summary

    def _d1(self):
        
        # d1 = ( ln( x2 / x1 ) + 0.5 * Sigma^2 * T ) / ( Sigma * sqrt(T) )
        d1 = ( np.log( self._x2() / self._x1() ) + 0.5 * ( self._combined_volatility() )**2 * self.time_to_maturity ) / ( self._combined_volatility() * np.sqrt(self.time_to_maturity) )
        return d1

    def _d2(self):

        # d2 = d1 - Sigma * sqrt(T)
        d2 = self._d1() - self._combined_volatility() * np.sqrt(self.time_to_maturity)
        return d2

    def _x1(self):

        # x1 = S1 * exp(-q1*T)
        x1 = self.underlying_price_1 * np.exp( -self.dividend_yield_1 * self.time_to_maturity )
        return x1

    def _x2(self):
    
        # x2 = S2 * exp(-q2*T)
        x2 = self.underlying_price_2 * np.exp( -self.dividend_yield_2 * self.time_to_maturity )
        return x2

    def _combined_volatility(self):

        # Sigma = sqrt( sigma1^2 + sigma2^2 - 2 * rho * sigma1 * sigma2 )
        Sigma = np.sqrt( self.underlying_volatility_1**2 + self.underlying_volatility_2**2 - 2 * self.correlation * self.underlying_volatility_1 * self.underlying_volatility_2 )
        return Sigma

    def price(self):

        # V = x2 * Phi(d1) - x1 * Phi(d2)
        price = self._x2() * norm.cdf(self._d1(), 0, 1) - self._x1() * norm.cdf(self._d2(), 0, 1)
        return price

    def chi(self):

        # Chi = -x2 * ( ( sigma1 * sigma2 ) / Sigma ) * sqrt(T) * phi(d1)
        chi = -self._x2() * ( ( self.underlying_volatility_1 * self.underlying_volatility_2 ) / self._combined_volatility() ) * np.sqrt(self.time_to_maturity) * norm.pdf(self._d1(), 0, 1)
        return chi

    def delta1(self):

        # Delta1 = exp(-q1*T) * Phi(d2)
        delta1 = np.exp(-self.dividend_yield_1 * self.time_to_maturity) * norm.cdf(self._d2(), 0, 1)
        return delta1

    def delta2(self):
    
        # Delta2 = exp(-q2*T) * Phi(d1)
        delta2 = np.exp(-self.dividend_yield_2 * self.time_to_maturity) * norm.cdf(self._d1(), 0, 1)
        return delta2

    def gamma1(self):

        # Gamma1 = ( exp(-q1* T) * phi(d2) ) / ( S2 * Sigma * sqrt(T) )
        gamma1 = np.exp(-self.dividend_yield_1 * self.time_to_maturity) * ( (norm.pdf(self._d2(), 0, 1)) / (self.underlying_price_2 * self._combined_volatility() * np.sqrt(self.time_to_maturity)) )
        return gamma1

    def gamma2(self):
    
        # Gamma2 = ( exp(-q2* T) * phi(d1) ) / ( S2 * Sigma * sqrt(T) )
        gamma2 = np.exp(-self.dividend_yield_2 * self.time_to_maturity) * ( (norm.pdf(self._d1(), 0, 1)) / (self.underlying_price_2 * self._combined_volatility() * np.sqrt(self.time_to_maturity)) )
        return gamma2

    def gamma12(self):
        
        # Gamma12 = ( exp(-(q1 + q2)* T) * phi(d1) ) / ( S1 * Sigma * sqrt(T) )
        gamma12 = np.exp(-( self.dividend_yield_1 + self.dividend_yield_2 ) * self.time_to_maturity) * ( (norm.pdf(self._d1(), 0, 1)) / (self.underlying_price_1 * self._combined_volatility() * np.sqrt(self.time_to_maturity)) )
        return gamma12

    def theta(self):

        # Theta = -x2 * ( Sigma / ( 2 * sqrt(T) ) ) * phi(d1) + q2 * x2 * Phi(d1) - q1 * x1 * Phi(d2)
        theta = -self._x2() * ( self._combined_volatility() / ( 2 * np.sqrt(self.time_to_maturity) ) ) * norm.pdf(self._d1(), 0, 1) + self.dividend_yield_2 * self._x2() * norm.cdf(self._d1(), 0, 1) - self.dividend_yield_1 * self._x1() * norm.cdf(self._d2(), 0, 1)
        return theta

    def vega(self):

        # Vega = x2 * sqrt(T) * phi(d1)
        vega = self._x2() * np.sqrt(self.time_to_maturity) * norm.pdf(self._d1(), 0, 1)
        return vega
