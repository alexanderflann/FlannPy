
# A Python Library for Option Pricing and Sensitivities

# Forward-Start Options

import numpy as np
from scipy.stats import norm

from .BlackScholesEuropean import BlackScholesEuropean

# Rubinstein Formula for Forward-Start Options

class RubinsteinForwardStart:

    # S             :   underlying_price
    # t             :   grant_date
    # T             :   expiration_date
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # sigma         :   underlying_volatility
    # call / put    :   option_type

    def __init__(self, underlying_price, grant_date, expiration_date, risk_free_rate, dividend_yield, underlying_volatility, option_type):
        self.underlying_price = underlying_price
        self.grant_date = grant_date
        self.expiration_date = expiration_date
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
        self.option_type = option_type
    # Constructor which intialises the object's instance variables
    
    def __str__(self):
        return (
            f"Price : {self.price():>8.4f}\n"
            f"Delta : {self.delta():>8.4f}\n"
            f"Rho   : {self.rho():>8.4f}\n"
            f"Theta : {self.theta():>8.4f}\n"
            f"Vega  : {self.vega():>8.4f}"
        )
    # String to return a summary

    def _d1(self):
        
        # d1 = ( ( r - q + 0.5 * sigma^2 ) * ( T - t ) ) / ( sigma * sqrt( T - t ) )
        d1 = ( ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * ( self.expiration_date - self.grant_date ) ) / ( self.underlying_volatility * np.sqrt( self.expiration_date - self.grant_date ) )
        return d1

    def _d2(self):

        # d2 = d1 - sigma * sqrt( T - t )
        d2 = self._d1() - self.underlying_volatility * np.sqrt( self.expiration_date - self.grant_date )
        return d2

    def price(self):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # C = S * exp(-q*t) * ( exp( -q * ( T - t ) ) * Phi(d1) - exp( -r * ( T - t ) ) * Phi(d2) )
            price = self.underlying_price * np.exp(-self.dividend_yield * self.grant_date) * ( np.exp( -self.dividend_yield * ( self.expiration_date - self.grant_date ) ) * norm.cdf(self._d1(), 0, 1) - np.exp( -self.risk_free_rate * ( self.expiration_date - self.grant_date ) ) * norm.cdf(self._d2(), 0, 1) )
        else:

            # P = S * exp(-q*t) * ( exp( -r * ( T - t ) ) * Phi(-d2) - exp( -q * ( T - t ) ) * Phi(-d1) )
            price = self.underlying_price * np.exp(-self.dividend_yield * self.grant_date) * ( np.exp( -self.risk_free_rate * ( self.expiration_date - self.grant_date ) ) * norm.cdf(-self._d2(), 0, 1) - np.exp( -self.dividend_yield * ( self.expiration_date - self.grant_date ) ) * norm.cdf(-self._d1(), 0, 1) )
        return price

    def delta(self):
    # Delta measures how the option price will change with a change in the underlying asset price
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):
        
            # C = exp(-q*t) * ( exp( -q * ( T - t ) ) * Phi(d1) - exp( -r * ( T - t ) ) * Phi(d2) )
            delta = np.exp(-self.dividend_yield * self.grant_date) * ( np.exp( -self.dividend_yield * ( self.expiration_date - self.grant_date ) ) * norm.cdf(self._d1(), 0, 1) - np.exp( -self.risk_free_rate * ( self.expiration_date - self.grant_date ) ) * norm.cdf(self._d2(), 0, 1) )
        else:

            # P = exp(-q*t) * ( exp( -r * ( T - t ) ) * Phi(-d2) - exp( -q * ( T - t ) ) * Phi(-d1) )
            delta = np.exp(-self.dividend_yield * self.grant_date) * ( np.exp( -self.risk_free_rate * ( self.expiration_date - self.grant_date ) ) * norm.cdf(-self._d2(), 0, 1) - np.exp( -self.dividend_yield * ( self.expiration_date - self.grant_date ) ) * norm.cdf(-self._d1(), 0, 1) )
        return delta

    def rho(self):
    # Rho measures how the option price will change with a change in the risk-free interest rate
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # Rho = S * exp(-q*t) * ( T - t ) * exp( -r * ( T - t ) ) * Phi(d2)
            rho = self.underlying_price * np.exp(-self.dividend_yield * self.grant_date) * ( self.expiration_date - self.grant_date ) * np.exp( -self.risk_free_rate * ( self.expiration_date - self.grant_date ) ) * norm.cdf(self._d2(), 0, 1)
        else:

            # Rho = -S * exp(-q*t) * ( T - t ) * exp( -r * ( T - t ) ) * Phi(-d2)
            rho = -self.underlying_price * np.exp(-self.dividend_yield * self.grant_date) * ( self.expiration_date - self.grant_date ) * np.exp( -self.risk_free_rate * ( self.expiration_date - self.grant_date ) ) * norm.cdf(-self._d2(), 0, 1)
        return rho

    def theta(self):
    # Theta measures how the option price will change as time passes

        tau = self.expiration_date - self.grant_date # tau = T - t
        black_scholes = BlackScholesEuropean(1, 1, tau, self.risk_free_rate, self.dividend_yield, self.underlying_volatility, self.option_type)
        black_scholes_theta = black_scholes.theta()

        # Theta = -q * V + S * exp(-q*t) * black_scholes_theta, as defined above
        theta = -self.dividend_yield * self.price() * np.exp(-self.dividend_yield * self.grant_date) * black_scholes_theta
        return theta

    def vega(self):
    # Vega measures how the option price will change with a change in the underlying asset volatility

        # Vega = S * exp(-q*T) * sqrt( T - t ) * phi(d1)
        vega = self.underlying_price * np.exp(-self.dividend_yield * self.expiration_date) * np.sqrt( self.expiration_date - self.grant_date ) * norm.pdf(self._d1(), 0, 1)
        return vega
