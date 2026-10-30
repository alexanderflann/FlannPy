
# A Python Library for Option Pricing and Sensitivities

# FX Options

import numpy as np
from scipy.stats import norm

# Garman-Kohlhagen Model for FX Options

class GarmanKohlhagenFX:
    
    # S             :   underlying_price
    # E             :   strike_price
    # T             :   time_to_maturity
    # r_d           :   domestic_rate
    # r_f           :   foreign_rate
    # q             :   dividend_yield
    # sigma         :   underlying_volatility
    # call / put    :   option_type

    def __init__(self, underlying_price, strike_price, time_to_maturity, domestic_rate, foreign_rate, dividend_yield, underlying_volatility, option_type):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.domestic_rate = domestic_rate
        self.foreign_rate = foreign_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
        self.option_type = option_type
    # Constructor which intialises the object's instance variables
    
    def __str__(self):
        return (
            f"Price        : {self.price():>8.4f}\n"
            f"Delta        : {self.delta():>8.4f}\n"
            f"Gamma        : {self.gamma():>8.4f}\n"
            f"Domestic Rho : {self.rho_domestic():>8.4f}\n"
            f"Foreign Rho  : {self.rho_foreign():>8.4f}\n"
            f"Theta        : {self.theta():>8.4f}\n"
            f"Vega         : {self.vega():>8.4f}"
        )
    # String to return a summary

    def _d1(self):
    # d1 determines how many standard deviations the strike price is from the mean
        
        # d1 = ( ln(S/E) + ( r_d - r_f - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d1 = ( np.log(self.underlying_price / self.strike_price) + (self.domestic_rate - self.foreign_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d1

    def _d2(self):
    # d2 adjusts d1 to use the median from the mean in a lognormal distribution

        # d2 = d1 - sigma * sqrt(T)
        d2 = self._d1() - self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return d2

    def price(self):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # C = S * exp(-(r_f + q)*T) * Phi(d1) - E * exp(-r_d*T) * Phi(d2)
            price = self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(self._d1(), 0, 1) - self.strike_price * np.exp(-self.domestic_rate * self.time_to_maturity) * norm.cdf(self._d2(), 0, 1)
        else:

            # P = E * exp(-r_d*T) * Phi(-d2) - S * exp(-(r_f + q)*T) * Phi(-d1)
            price = self.strike_price * np.exp(-self.domestic_rate * self.time_to_maturity) * norm.cdf(-self._d2(), 0, 1) - self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(-self._d1(), 0, 1)
        return price

    def delta(self):
    # Delta measures how the option price will change with a change in the underlying asset price
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # Delta = exp(-(r_f + q)*T) * Phi(d1)
            delta = np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(self._d1(), 0, 1)
        else:

            # Delta = exp(-(r_f + q)*T) * Phi(-d1)
            delta = np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(-self._d1(), 0, 1)
        return delta

    def gamma(self):
    # Gamma measures the rate of change of Delta

        # Gamma = ( exp(-(r_f + q)* T) * phi(d1) ) / ( S * sigma * sqrt(T) )
        gamma = np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * ( (norm.pdf(self._d1(), 0, 1)) / (self.underlying_price * self.underlying_volatility * np.sqrt(self.time_to_maturity)) )
        return gamma

    def rho_domestic(self):
    # Domestic Rho measures how the option price will change with a change in the domestic interest rate
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # Rho = E * exp(-r_d*T) * T * Phi(d2)
            rho = self.strike_price * np.exp(-self.domestic_rate * self.time_to_maturity) * norm.cdf(self._d2(), 0, 1)
        else:

            # Rho = - E * exp(-r_d*T) * T * Phi(-d2)
            rho = - self.strike_price * np.exp(-self.domestic_rate * self.time_to_maturity) * norm.cdf(-self._d2(), 0, 1)
        return rho

    def rho_foreign(self):
    # Foreign Rho measures how the option price will change with a change in the domestic interest rate
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):
        
            # Rho = - S * exp(-(r_f + q)*T) * T * Phi(d1)
            rho = self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(self._d1(), 0, 1)
        else:

            # Rho = - S * exp(-(r_f + q)*T) * T * Phi(-d1)
            rho = - self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(-self._d1(), 0, 1)
        return rho

    def theta(self):
    # Theta measures how the option price will change as time passes
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # Theta = - S * exp(-(r_f + q)*T) * phi(d1) * sigma / ( 2 * sqrt(T) ) + S * (r_f + q) * exp(-(r_f + q)*T) * Phi(d1) - E * r_d * exp(-r_d*T) * Phi(d2)
            theta = - self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.pdf(self._d1(), 0, 1) * ( self.underlying_volatility / (2 * np.sqrt(self.time_to_maturity)) ) + self.underlying_price * (self.foreign_rate + self.dividend_yield) * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(self._d1(), 0, 1) - self.strike_price * self.domestic_rate * np.exp(-self.domestic_rate * self.time_to_maturity) * norm.cdf(self._d2(), 0, 1)
        else:

            # Theta = - S * exp(-(r_f + q)*T) * phi(d1) * sigma / ( 2 * sqrt(T) ) - S * (r_f + q) * exp(-(r_f + q)*T) * Phi(-d1) + E * r_d * exp(-r_d*T) * Phi(-d2) 
            theta = - self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.pdf(self._d1(), 0, 1) * ( self.underlying_volatility / (2 * np.sqrt(self.time_to_maturity)) ) - self.underlying_price * (self.foreign_rate + self.dividend_yield) * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * norm.cdf(-self._d1(), 0, 1) + self.strike_price * self.domestic_rate * np.exp(-self.domestic_rate * self.time_to_maturity) * norm.cdf(-self._d2(), 0, 1)
        return theta

    def vega(self):
    # Vega measures how the option price will change with a change in the underlying asset volatility

        # Vega = S * exp(-(r_f + q)*T) * sqrt(T) * phi(d1)
        vega = self.underlying_price * np.exp(-(self.foreign_rate + self.dividend_yield) * self.time_to_maturity) * np.sqrt(self.time_to_maturity) * norm.pdf(self._d1(), 0, 1)
        return vega
