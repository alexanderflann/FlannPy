
# A Python Library for Option Pricing and Sensitivities

# Single Barrier Options

import numpy as np
from scipy.stats import norm

from .NumericalMethods import FiniteDifferenceMethods

# Reiner-Rubinstein Model for Single Barrier Options

class ReinerRubinsteinBarrier:

    # S             :   underlying_price
    # E             :   strike_price
    # X             :   barrier_price
    # T             :   time_to_maturity
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # sigma         :   underlying_volatility
    # up / down     :   barrier_direction
    # out / in      :   barrier_type
    # call / put    :   option_type
    # J             :   rebate

    def __init__(self, underlying_price, strike_price, barrier_price, time_to_maturity, risk_free_rate, dividend_yield, underlying_volatility, barrier_direction, barrier_type, option_type, rebate=0):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.barrier_price = barrier_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
        self.barrier_direction = barrier_direction
        self.barrier_type = barrier_type
        self.option_type = option_type
        self.rebate = rebate
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

    def _mu(self):

        # mu = ( r - q - sigma^2 / 2 ) / sigma^2
        mu = ( self.risk_free_rate - self.dividend_yield - self.underlying_volatility**2 / 2 ) / ( self.underlying_volatility**2 )
        return mu

    def _lambda(self):

        # lambda = sqrt( mu^2 + 2r / sigma^2 )
        lambd = np.sqrt( self._mu()**2 + ( 2 * self.risk_free_rate / ( self.underlying_volatility**2 ) ) )
        return lambd

    def _x1(self):

        # x1 = ln(S/E) / ( sigma * sqrt(T) ) + ( mu + 1 ) * sigma * sqrt(T)
        x1 = np.log( self.underlying_price / self.strike_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + ( self._mu() + 1 ) * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x1

    def _x2(self):

        # x2 = = ln(S/X) / ( sigma * sqrt(T) ) + ( mu + 1 ) * sigma * sqrt(T)
        x2 = np.log( self.underlying_price / self.barrier_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + ( self._mu() + 1 ) * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x2

    def _x3(self):
    
        # x3 = = ln( X^2 / (S * E) ) / ( sigma * sqrt(T) ) + ( mu + 1 ) * sigma * sqrt(T)
        x3 = np.log( self. barrier_price**2 / ( self.underlying_price * self.strike_price ) ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + ( self._mu() + 1 ) * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x3

    def _x4(self):

        # x4 = = ln(X/S) / ( sigma * sqrt(T) ) + ( mu + 1 ) * sigma * sqrt(T)
        x4 = np.log( self.barrier_price / self.underlying_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + ( self._mu() + 1 ) * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x4

    def _x5(self):
    
        # x5 = = ln(X/S) / ( sigma * sqrt(T) ) + lambda * sigma * sqrt(T)
        x5 = np.log( self.barrier_price / self.underlying_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + self._lambda() * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x5

    def _phi(self):

        # phi = 1 if Call and -1 if Put
        if ( self.option_type == True ):
            phi = 1
        else:
            phi = -1
        return phi

    def _eta(self):

        # eta = -1 if Up and 1 if Down
        if ( self.barrier_direction == True ):
            eta = -1
        else:
            eta = 1
        return eta
    
    def _A1(self):

        # A1 = phi * S * exp(-q*T) * Phi(phi * x1) - phi * E * exp(-r*T) * Phi(phi * x1 - phi * sigma * sqrt(T))
        A1 = self._phi() * self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._phi() * self._x1(), 0, 1) - self._phi() * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._phi() * self._x1() - self._phi() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return A1

    def _A2(self):
    
        # A2 = phi * S * exp(-q*T) * Phi(phi * x2) - phi * E * exp(-r*T) * Phi(phi * x2 - phi * sigma * sqrt(T))
        A2 = self._phi() * self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._phi() * self._x2(), 0, 1) - self._phi() * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._phi() * self._x2() - self._phi() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return A2

    def _A3(self):
    
        # A3 = phi * (X / S)^( 2 (mu + 1) ) * S * exp(-q*T) * Phi(eta * x3) - phi * (X / S)^(2 * mu) * E * exp(-r*T) * Phi(eta * x3 - eta * sigma * sqrt(T))
        A3 = self._phi() *  ( self.barrier_price / self.underlying_price )**(2 * ( self._mu() + 1 )) * self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._eta() * self._x3(), 0, 1) - self._phi() * ( self.barrier_price / self.underlying_price )**(2 * self._mu()) * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._eta() * self._x3() - self._eta() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return A3

    def _A4(self):
    
        # A4 = phi * (X / S)^( 2 (mu + 1) ) * S * exp(-q*T) * Phi(eta * x4) - phi * (X / S)^(2 * mu) * E * exp(-r*T) * Phi(eta * x4 - eta * sigma * sqrt(T))
        A4 = self._phi() *  ( self.barrier_price / self.underlying_price )**(2 * ( self._mu() + 1 )) * self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._eta() * self._x4(), 0, 1) - self._phi() * ( self.barrier_price / self.underlying_price )**(2 * self._mu()) * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._eta() * self._x4() - self._eta() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return A4

    def _A5(self):
    
        # A5 = J * exp(-r*T) * ( Phi(eta * x2 - eta * sigma * sqrt(T)) - (X / S)^(2 * mu) * Phi(eta * x4 - eta * sigma * sqrt(T)) )
        A5 = self.rebate * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( norm.cdf( self._eta() * self._x2() - self._eta() * self.underlying_volatility * np.sqrt(self.time_to_maturity),0, 1) - ( self.barrier_price / self.underlying_price )**(2 * self._mu()) * norm.cdf( self._eta() * self._x4() - self._eta() * self.underlying_volatility * np.sqrt(self.time_to_maturity),0, 1) )
        return A5

    def _A6(self):
    
        # A6 = J * ( (X / S)^(mu + lambda) * Phi(eta * x5) + (X / S)^(mu - lambda) * Phi(eta * x5 - 2 * eta * lambda * sigma * sqrt(T)) )
        A6 = self.rebate * ( ( self.barrier_price / self.underlying_price )**(self._mu() + self._lambda()) * norm.cdf( self._eta() * self._x5(),0, 1) + ( self.barrier_price / self.underlying_price )**(self._mu() - self._lambda()) * norm.cdf( self._eta() * self._x5() - 2 * self._eta() * self._lambda() * self.underlying_volatility * np.sqrt(self.time_to_maturity),0, 1) )
        return A6

    def price(self):
    # Return price of a barrier option
    # barrier_direction is Up = True, Down = False
    # barrier_type is Out = True, In = False
    # option_type is Call = True, Put = False

        if ( self.strike_price > self.barrier_price ):                                  # E > X
            if ( self.barrier_type == True ):
                if ( self.barrier_direction == True ):
                    if ( self.option_type == True ):
                        price = self._A6()                                                      # Up-and-Out Call
                    else:
                        price = self._A2() - self._A4() + self._A6()                            # Up-and-Out Put
                else:
                    if ( self.option_type == True ):
                        price = self._A1() - self._A3() + self._A6()                            # Down-and-Out Call
                    else:
                        price = self._A1() - self._A2() + self._A3() - self._A4() + self._A6()  # Down-and-Out Put
            else:
                if ( self.barrier_direction == True ):
                    if ( self.option_type == True ):
                        price = self._A1() + self._A5()                                         # Up-and-In Call
                    else:
                        price = self._A1() - self._A2() + self._A4() + self._A5()               # Up-and-In Put
                else:
                    if ( self.option_type == True ):
                        price = self._A3() + self._A5()                                         # Down-and-In Call
                    else:
                        price = self._A2() - self._A3() + self._A4() + self._A5()               # Down-and-In Put
        else:                                                                           # E <= X
            if ( self.barrier_type == True ):
                if ( self.barrier_direction == True ):
                    if ( self.option_type == True ):
                        price = self._A1() - self._A2() + self._A3() - self._A4() + self._A6()  # Up-and-Out Call
                    else:
                        price = self._A1() - self._A3() + self._A6()                            # Up-and-Out Put
                else:
                    if ( self.option_type == True ):
                        price = self._A2() - self._A4() + self._A6()                            # Down-and-Out Call
                    else:
                        price = self._A6()                                                      # Down-and-Out Put
            else:
                if ( self.barrier_direction == True ):
                    if ( self.option_type == True ):
                        price = self._A2() - self._A3() + self._A4() + self._A5()               # Up-and-In Call
                    else:
                        price = self._A3() + self._A5()                                         # Up-and-In Put
                else:
                    if ( self.option_type == True ):
                        price = self._A1() - self._A2() + self._A4() + self._A5()               # Down-and-In Call
                    else:
                        price = self._A1() + self._A5()                                         # Down-and-In Put
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
