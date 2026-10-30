
# A Python Library for Volatility Modelling

# Implied Volatility

import numpy as np
from scipy.stats import norm

from .NumericalMethods import NewtonRaphson

# Implied Volatility via Newton-Raphson Method

class ImpliedVolatility:

    # S             :   underlying_price
    # E             :   strike_price
    # T             :   time_to_maturity
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # call / put    :   option_type
    # market price  :   market_price

    def __init__(self, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, option_type, market_price):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.option_type = option_type
        self.market_price = market_price
    # Constructor which intialises the object's instance variables

    def __str__(self):
        return (
            f"Implied Volatility : {self.implied_volatility():>8.4f}"
        )

    def _d1(self, sigma_guess):
    # d1 determines how many standard deviations the strike price is from the mean
        
        # d1 = ( ln(S/E) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d1 = ( np.log(self.underlying_price / self.strike_price) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma_guess**2) * self.time_to_maturity ) / ( sigma_guess * np.sqrt(self.time_to_maturity) )
        return d1

    def _d2(self, sigma_guess):
    # d2 adjusts d1 to use the median from the mean in a lognormal distribution

        # d2 = d1 - sigma * sqrt(T)
        d2 = self._d1(sigma_guess) - sigma_guess * np.sqrt(self.time_to_maturity)
        return d2

    def _black_scholes(self, sigma_guess):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # C = S * exp(-q*T) * Phi(d1) - E * exp(-r*T) * Phi(d2)
            price = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._d1(sigma_guess), 0, 1) - self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._d2(sigma_guess), 0, 1)
        else:

            # P = E * exp(-r*T) * Phi(-d2) - S * exp(-q*T) * Phi(-d1)
            price = self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(-self._d2(sigma_guess), 0, 1) - self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(-self._d1(sigma_guess), 0, 1)
        return price
    
    def _loss_function(self, sigma_guess):
    # Define the loss function

        theoretical_price = self._black_scholes(sigma_guess) # Price with the guess for volatility

        return theoretical_price - self.market_price # Loss is the difference between the theoretical price and the actual price, we want to minimise this loss

    def _loss_gradient(self, sigma_guess):
    # Vega measures the sensitivity of the Black-Scholes price to volatility

        # Vega = S * exp(-q*T) * sqrt(T) * phi(d1)
        vega = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * np.sqrt(self.time_to_maturity) * norm.pdf(self._d1(sigma_guess), 0, 1)
        return vega

    def implied_volatility(self, sigma_guess = 0.8, tolerance = 1e-3, max_iteration = 20):
    # Solve for implied volatility via the Newton-Raphson method

        sigma = NewtonRaphson(
            self._loss_function,
            self._loss_gradient,
            sigma_guess,
            tolerance,
            max_iteration
        ).newton_raphson()

        return sigma
