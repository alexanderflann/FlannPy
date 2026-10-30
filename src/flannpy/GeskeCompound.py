
# A Python Library for Option Pricing and Sensitivities

# Compound Options

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from .NumericalMethods import NewtonRaphson
from .NumericalMethods import FiniteDifferenceMethods

# Geske Model for Compound Options

class GeskeCompound:

    # S             :   underlying_price
    # E_1           :   compound_strike_price
    # E_2           :   underlying_strike_price
    # T_1           :   compound_time_to_maturity
    # T_2           :   underlying_time_to_maturity
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # sigma         :   underlying_volatility
    # call / put    :   compound_type
    # call / put    :   option_type

    def __init__(self, underlying_price, compound_strike_price, underlying_strike_price, compound_time_to_maturity, underlying_time_to_maturity, risk_free_rate, dividend_yield, underlying_volatility, compound_type, option_type):
        self.underlying_price = underlying_price
        self.compound_strike_price = compound_strike_price
        self.underlying_strike_price = underlying_strike_price
        self.compound_time_to_maturity = compound_time_to_maturity
        self.underlying_time_to_maturity = underlying_time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
        self.compound_type = compound_type
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

    def _x1(self):
        
        # x1 = ( ln( S / S* ) + ( r - q + 0.5 * sigma^2 ) * T1 ) / ( sigma * sqrt(T1) )
        x1 = ( np.log( self.underlying_price / self._critical_underlying_price(self.underlying_strike_price) ) + (self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2) * self.compound_time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.compound_time_to_maturity) )
        return x1

    def _x2(self):

        # x2 = x1 - sigma * sqrt(T1)
        x2 = self._x1() - self.underlying_volatility * np.sqrt(self.compound_time_to_maturity)
        return x2

    def _y1(self):

        # y1 = ( ln( S / E2 ) + ( r - q + 0.5 * sigma^2 ) * T2 ) / ( sigma * sqrt(T2) )
        y1 = ( np.log( self.underlying_price / self.underlying_strike_price ) + (self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2) * self.underlying_time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.underlying_time_to_maturity) )
        return y1

    def _y2(self):
    
        # y2 = y1 - sigma * sqrt(T2)
        y2 = self._y1() - self.underlying_volatility * np.sqrt(self.underlying_time_to_maturity)
        return y2

    def _rho(self):

        # rho = sqrt( T1 / T2 )
        rho = np.sqrt( self.compound_time_to_maturity / self.underlying_time_to_maturity )
        return rho

    def _phi(self):

        # phi = 1 if compound option is a Call and -1 if Put
        if ( self.compound_type == True ):
            phi = 1
        else:
            phi = -1
        return phi

    def _eta(self):

        # eta = 1 if underlying option is a Call and -1 if Put
        if ( self.option_type == True ):
            eta = 1
        else:
            eta = -1
        return eta

    def _bivariate_normal(self, a, b, rho):

        # Define the cumulative distribution function of a standard bivariate normal random variable
        M = multivariate_normal.cdf(
            [a, b],
            mean = [0, 0],
            cov = [[1, rho],
                   [rho, 1]]
        )
        
        return M

    def _A1(self):

        # A1 = phi * eta * S * exp(-q * T2) * Phi2( phi * eta * x1 , eta * y1 ; phi * rho )
        A1 = self._phi() * self._eta() * self.underlying_price * np.exp(-self.dividend_yield * self.underlying_time_to_maturity) * self._bivariate_normal(self._phi() * self._eta() * self._x1(), self._eta() * self._y1(), self._phi() * self._rho())
        return A1

    def _A2(self):

        # A2 = phi * eta * E2 * exp(-r * T2) * Phi2( phi * eta * x2 , eta * y2 ; phi * rho )
        A2 = self._phi() * self._eta() * self.underlying_strike_price * np.exp(-self.risk_free_rate * self.underlying_time_to_maturity) * self._bivariate_normal(self._phi() * self._eta() * self._x2(), self._eta() * self._y2(), self._phi() * self._rho())
        return A2

    def _A3(self):

        # A3 = phi * E1 * exp(-r * T1) * Phi( phi * eta * x2 )
        A3 = self._phi() * self.compound_strike_price * np.exp(-self.risk_free_rate * self.compound_time_to_maturity) * norm.cdf(self._phi() * self._eta() * self._x2() , 0, 1)
        return A3

    def _tau(self):

        # tau = T2 - T1
        tau = self.underlying_time_to_maturity - self.compound_time_to_maturity
        return tau

    def _z1(self, critical_underlying_price_guess):
            
        # z1 = ( ln( S* / E2 ) + ( r - q + 0.5 * sigma^2 ) * tau ) / ( sigma * sqrt(tau) )
        z1 = ( np.log( critical_underlying_price_guess / self.underlying_strike_price ) + (self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2) * self._tau() ) / ( self.underlying_volatility * np.sqrt(self._tau()) )
        return z1

    def _z2(self, critical_underlying_price_guess):

        # z2 = z1 - sigma * sqrt(tau)
        z2 = self._z1(critical_underlying_price_guess) - self.underlying_volatility * np.sqrt(self._tau())
        return z2

    def _critical_underlying_price_function(self, critical_underlying_price_guess):

        # S* is found numerically using Newton-Raphson on:
        # f(S*) = eta * S* * exp(-q * tau) * Phi(eta * z1) - eta * E2 * exp(-r * tau) * Phi(eta * z2) - E1 = 0
        loss = self._eta() * critical_underlying_price_guess * np.exp( -self.dividend_yield * self._tau() ) * norm.cdf(self._eta() * self._z1(critical_underlying_price_guess), 0, 1) - self._eta() * self.underlying_strike_price * np.exp( -self.risk_free_rate * self._tau() ) * norm.cdf(self._eta() * self._z2(critical_underlying_price_guess), 0, 1) - self.compound_strike_price
        return loss

    def _critical_underlying_price_gradient(self, critical_underlying_price_guess):

        # The gradient is the derivative of the loss function w.r.t. the underlying asset price, S
        # Note that in this case it's Delta
        # Delta = eta * exp(-q * tau) * Phi(eta * z1)
        delta = self._eta() * np.exp( -self.dividend_yield * self._tau() ) * norm.cdf(self._eta() * self._z1(critical_underlying_price_guess), 0, 1)
        return delta

    def _critical_underlying_price(self, critical_underlying_price_guess, tolerance = 1e-3, max_iteration = 20):
    # Function to return the critical underlying price via the Newton-Raphson method
    # Usual initial guess for the critical underlying price is the underlying strike price

        critical_price = NewtonRaphson(
                self._critical_underlying_price_function,
                self._critical_underlying_price_gradient,
                critical_underlying_price_guess,
                tolerance,
                max_iteration
            ).newton_raphson()
        
        return critical_price

    def price(self):

        # C = A1 - A2 - A3
        price = self._A1() - self._A2() - self._A3()
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

        parameters = (
            "compound_time_to_maturity",
            "underlying_time_to_maturity"
        )

        theta = - np.sum([
            FiniteDifferenceMethods(
                self,
                self.price,
                parameter,
                h = 1 / 365
            ).first_derivative()
            for parameter in parameters
        ])

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
