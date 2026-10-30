
# A Python Library for Option Pricing and Sensitivities

# Double Barrier Options

import numpy as np
from scipy.stats import norm

from .InfiniteSeries import InfiniteSeries
from .BlackScholesEuropean import BlackScholesEuropean
from .NumericalMethods import FiniteDifferenceMethods

# Kunitomo-Ikeda Model for Double Barrier Options

class KunitomoIkedaBarrier:

    # S                 :   underlying_price
    # E                 :   strike_price
    # X_u               :   upper_barrier_price
    # X_d               :   lower_barrier_price
    # T                 :   time_to_maturity
    # r                 :   risk_free_rate
    # q                 :   dividend_yield
    # sigma             :   underlying_volatility
    # knock out / in    :   barrier_type
    # call / put        :   option_type
    # delta_1           :   upper_barrier_curvature
    # delta_2           :   lower_barrier_curvature

    def __init__(
            self,
            underlying_price,
            strike_price,
            upper_barrier_price,
            lower_barrier_price,
            time_to_maturity,
            risk_free_rate,
            dividend_yield,
            underlying_volatility,
            barrier_type,
            option_type,
            upper_barrier_curvature=0,
            lower_barrier_curvature=0
        ):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.upper_barrier_price = upper_barrier_price
        self.lower_barrier_price = lower_barrier_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
        self.barrier_type = barrier_type
        self.option_type = option_type
        self.upper_barrier_curvature = upper_barrier_curvature
        self.lower_barrier_curvature = lower_barrier_curvature
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

    def _mu1(self, n):

        # mu1 = ( 2 ( r - q - delta_2 - n ( delta_1 - delta_2 ) ) ) / sigma^2 + 1
        mu1 = ( 2 * ( self.risk_free_rate - self.dividend_yield - self.lower_barrier_curvature - n * ( self.upper_barrier_curvature - self.lower_barrier_curvature ) ) ) / self.underlying_volatility**2 + 1
        return mu1

    def _mu2(self, n):

        # mu2 = ( 2 * n * ( delta_1 - delta_2 ) ) / sigma^2
        mu2 = ( 2 * n * ( self.upper_barrier_curvature - self.lower_barrier_curvature ) ) / self.underlying_volatility**2
        return mu2

    def _mu3(self, n):

        # mu3 = ( 2 ( r - q - delta_2 + n ( delta_1 - delta_2 ) ) ) / sigma^2 + 1
        mu3 = ( 2 * ( self.risk_free_rate - self.dividend_yield - self.lower_barrier_curvature + n * ( self.upper_barrier_curvature - self.lower_barrier_curvature ) ) ) / self.underlying_volatility**2 + 1
        return mu3

    def _F1(self):

        # F1 = X_u * exp( delta_1 * T )
        F1 = self.upper_barrier_price * np.exp( self.upper_barrier_curvature * self.time_to_maturity )
        return F1

    def _F2(self):

        # F2 = X_d * exp( delta_2 * T )
        F2 = self.lower_barrier_price * np.exp( self.lower_barrier_curvature * self.time_to_maturity )
        return F2

    def _d1(self, n):

        # d1 = ( ln ( ( S * (X_u)^(2n) ) / ( E * (X_d)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d1 = ( np.log( ( self.underlying_price * self.upper_barrier_price**(2 * n) ) / ( self.strike_price * self.lower_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d1

    def _d2(self, n):
    
        # d2 = ( ln ( ( S * (X_u)^(2n) ) / ( F1 * (X_d)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d2 = ( np.log( ( self.underlying_price * self.upper_barrier_price**(2 * n) ) / ( self._F1() * self.lower_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d2

    def _d3(self, n):
    
        # d3 = ( ln ( ( (X_d)^(2n+2) ) / ( S * E * (X_u)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d3 = ( np.log( ( self.lower_barrier_price**(2 * n + 2) ) / ( self.underlying_price * self.strike_price * self.upper_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d3

    def _d4(self, n):
    
        # d4 = ( ln ( ( (X_d)^(2n+2) ) / ( F1 * S * (X_u)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d4 = ( np.log( ( self.lower_barrier_price**(2 * n + 2) ) / ( self._F1() * self.underlying_price * self.upper_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d4

    def _y1(self, n):
    
        # y1 = ( ln ( ( S * (X_u)^(2n) ) / ( F2 * (X_d)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        y1 = ( np.log( ( self.underlying_price * self.upper_barrier_price**(2 * n) ) / ( self._F2() * self.lower_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return y1

    def _y2(self, n):

        # y2 = ( ln ( ( S * (X_u)^(2n) ) / ( E * (X_d)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        y2 = ( np.log( ( self.underlying_price * self.upper_barrier_price**(2 * n) ) / ( self.strike_price * self.lower_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return y2

    def _y3(self, n):

        # y3 = ( ln ( ( (X_d)^(2n+2) ) / ( S * F2 * (X_u)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        y3 = ( np.log( ( self.lower_barrier_price**(2 * n + 2) ) / ( self.underlying_price * self._F2() * self.upper_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return y3

    def _y4(self, n):
    
        # y4 = ( ln ( ( (X_d)^(2n+2) ) / ( S * E * (X_u)^(2n) ) ) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        y4 = ( np.log( ( self.lower_barrier_price**(2 * n + 2) ) / ( self.underlying_price * self.strike_price * self.upper_barrier_price**(2 * n) ) ) + ( self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2 ) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return y4

    def _summand14(self, n):
    # summand1 represents the first summand in a Double Knock-Out Call option
    # summand4 represents the second summand in a Double Knock-Out Put option

        # Call if True, Put if False
        if ( self.option_type == True ):

            # summand1 = ( (X_u)^n / (X_d)^n )^(mu1) * ( X_d / S )^(mu2) * ( Phi(d1) - Phi(d2) ) - ( (X_d)^(n+1) / ( S * (X_u)^n ) )^(mu3) * ( Phi(d3) - Phi(d4) )
            summand1 = ( self.upper_barrier_price**n / self.lower_barrier_price**n )**(self._mu1(n)) * ( self.lower_barrier_price / self.underlying_price )**(self._mu2(n)) * ( norm.cdf(self._d1(n), 0, 1) - norm.cdf(self._d2(n), 0, 1) ) - ( self.lower_barrier_price**(n + 1) / ( self.underlying_price * self.upper_barrier_price**n ) )**(self._mu3(n)) * ( norm.cdf(self._d3(n), 0, 1) - norm.cdf(self._d4(n), 0, 1) )
            return summand1
        else:

            # summand4 = ( (X_u)^n / (X_d)^n )^(mu1) * ( X_d / S )^(mu2) * ( Phi(y1) - Phi(y2) ) - ( (X_d)^(n+1) / ( S * (X_u)^n ) )^(mu3) * ( Phi(y3) - Phi(y4) )
            summand4 = ( self.upper_barrier_price**n / self.lower_barrier_price**n )**(self._mu1(n)) * ( self.lower_barrier_price / self.underlying_price )**(self._mu2(n)) * ( norm.cdf(self._y1(n), 0, 1) - norm.cdf(self._y2(n), 0, 1) ) - ( self.lower_barrier_price**(n + 1) / ( self.underlying_price * self.upper_barrier_price**n ) )**(self._mu3(n)) * ( norm.cdf(self._y3(n), 0, 1) - norm.cdf(self._y4(n), 0, 1) )
            return summand4

    def _summand23(self, n):
    # summand2 represents the second summand in a Double Knock-Out Call option
    # summand3 represents the first summand in a Double Knock-Out Put option

        # Call if True, Put if False
        if ( self.option_type == True ):

            # summand2 = ( (X_u)^n / (X_d)^n )^(mu1 - 2) * ( X_d / S )^(mu2) * ( Phi(d1 - sigma * sqrt(T)) - Phi(d2 - sigma * sqrt(T)) ) - ( (X_d)^(n+1) / ( S * (X_u)^n ) )^(mu3 - 2) * ( Phi(d3 - sigma * sqrt(T)) - Phi(d4 - sigma * sqrt(T)) )
            summand2 = ( self.upper_barrier_price**n / self.lower_barrier_price**n )**(self._mu1(n) - 2) * ( self.lower_barrier_price / self.underlying_price )**(self._mu2(n)) * ( norm.cdf(self._d1(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) - norm.cdf(self._d2(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) ) - ( self.lower_barrier_price**(n + 1) / ( self.underlying_price * self.upper_barrier_price**n ) )**(self._mu3(n) - 2) * ( norm.cdf(self._d3(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) - norm.cdf(self._d4(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) )
            return summand2
        else:

            # summand3 = ( (X_u)^n / (X_d)^n )^(mu1 - 2) * ( X_d / S )^(mu2) * ( Phi(y1 - sigma * sqrt(T)) - Phi(y2 - sigma * sqrt(T)) ) - ( (X_d)^(n+1) / ( S * (X_u)^n ) )^(mu3 - 2) * ( Phi(y3 - sigma * sqrt(T)) - Phi(y4 - sigma * sqrt(T)) )
            summand3 = ( self.upper_barrier_price**n / self.lower_barrier_price**n )**(self._mu1(n) - 2) * ( self.lower_barrier_price / self.underlying_price )**(self._mu2(n)) * ( norm.cdf(self._y1(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) - norm.cdf(self._y2(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) ) - ( self.lower_barrier_price**(n + 1) / ( self.underlying_price * self.upper_barrier_price**n ) )**(self._mu3(n) - 2) * ( norm.cdf(self._y3(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) - norm.cdf(self._y4(n) - self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1) )
            return summand3

    def price(self, tolerance = 1e-4, max_iteration = 10):
    # Return price of a double barrier option, tolerance and max iterations are for infinite series
    # Max iteration is small due to how quickly Kunitomo-Ikeda style infinite series converge fast
    # barrier_type is Out = True, In = False
    # option_type is Call = True, Put = False

        # Instantise the series using _summand14 and _summand23 as the pointer functions
        series = [
            InfiniteSeries(summand, tolerance, max_iteration).summation()
            for summand in (self._summand14, self._summand23)
        ]

        # Take sum1 and sum2 out of the series array
        sum1, sum2 = series

        # Double Knock-Out Call
        # C = S * exp(-q*T) * sum1 - E * exp(-r*T) * sum2, where sum1 is for summand1 and sum2 is for summand4
        DKOC = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * sum1 - self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * sum2

        # Double Knock-Out Put
        # P = E * exp(-r*T) * sum2 - S * exp(-q*T) * sum1, where sum1 is for summand4 and sum2 is for summand3
        DKOP = self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * sum2 - self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * sum1

        black_scholes = BlackScholesEuropean(self.underlying_price, self.strike_price, self.time_to_maturity, self.risk_free_rate, self.dividend_yield, self.underlying_volatility, self.option_type).price()

        if ( self.option_type == True ):

            if ( self.barrier_type == True ):

                # Double Knock-Out-Call
                price = DKOC
            else:

                # Double Knock-In Call
                price = black_scholes - DKOC
        else:
            if ( self.barrier_type == True ):

                # Double Knock-Out Put
                price = DKOP
            else:

                # Double Knock-In Put
                price = black_scholes - DKOP
        return price

    def delta(self):
    # Sensitivity of option price to a change in the underlying price

        delta = FiniteDifferenceMethods(
            self,
            self.price,
            "underlying_price",
            h = 1e-2
        ).first_derivative()

        return delta

    def gamma(self):
    # Sensitivity of option delta to a change in the underlying price
    
        gamma = FiniteDifferenceMethods(
            self,
            self.price,
            "underlying_price",
            h = 1e-2
        ).second_derivative()

        return gamma

    def rho(self):
    # Sensitivity of option price to a change in the risk-free rate

        rho = FiniteDifferenceMethods(
            self,
            self.price,
            "risk_free_rate",
            h = 1e-4
        ).first_derivative()

        return rho

    def theta(self):
    # Sensitivity of option price to a change as time passes

        theta = - FiniteDifferenceMethods(
            self,
            self.price,
            "time_to_maturity",
            h = 1 / 365
        ).first_derivative()

        return theta

    def vega(self):
    # Sensitivity of option price to a change in the underlying volatility

        vega = FiniteDifferenceMethods(
            self,
            self.price,
            "underlying_volatility",
            h = 1e-4
        ).first_derivative()

        return vega
