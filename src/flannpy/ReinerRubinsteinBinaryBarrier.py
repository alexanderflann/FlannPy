
# A Python Library for Option Pricing and Sensitivities

# Binary Barrier Options

import numpy as np
from scipy.stats import norm

from .NumericalMethods import FiniteDifferenceMethods

# Reiner-Rubinstein Model for Binary Barrier Options

class ReinerRubinsteinBinaryBarrier:

    # S             :   underlying_price
    # E             :   strike_price
    # X             :   barrier_price
    # T             :   time_to_maturity
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # sigma         :   underlying_volatility
    # up / down     :   barrier_direction
    # out / in      :   barrier_type
    # cash / asset  :   binary_type

    # call / put    :   option_type
    # B             :   fixed_amount

    def __init__(self, underlying_price, strike_price, barrier_price, time_to_maturity, risk_free_rate, dividend_yield, underlying_volatility, barrier_direction, barrier_type, binary_type):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.barrier_price = barrier_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
        self.barrier_direction = barrier_direction
        self.barrier_type = barrier_type
        self.binary_type = binary_type
    # Constructor which intialises the object's instance variables

    def _mu(self):

        # mu = r - q - sigma^2 / 2
        mu = self.risk_free_rate - self.dividend_yield - self.underlying_volatility**2 / 2
        return mu

    def _lambda(self):

        # lambda = 1 + mu / sigma^2
        lambd = 1 + self._mu() / ( self.underlying_volatility**2 )
        return lambd

    def _a(self):

        # a = mu / sigma^2
        a = self._mu() / ( self.underlying_volatility**2 )
        return a

    def _b(self):

        # b = sqrt( mu^2 + 2 * r * sigma^2 ) / sigma^2
        b = np.sqrt( (self._mu())**2 + 2 * self.risk_free_rate * self.underlying_volatility**2 ) / self.underlying_volatility**2
        return b

    def _x1(self):

        # x1 = ln(S/E) / ( sigma * sqrt(T) ) + lambda * sigma * sqrt(T)
        x1 = np.log( self.underlying_price / self.strike_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + self._lambda() * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x1

    def _x2(self):

        # x2 = ln(S/X) / ( sigma * sqrt(T) ) + lambda * sigma * sqrt(T)
        x2 = np.log( self.underlying_price / self.barrier_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + self._lambda() * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x2

    def _x3(self):
    
        # x3 = ln( (X^2) / (S*E) ) / ( sigma * sqrt(T) ) + lambda * sigma * sqrt(T)
        x3 = np.log( ( self.barrier_price**2 )/ ( self.underlying_price * self.strike_price ) ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + self._lambda() * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x3

    def _x4(self):

        # x4 = ln(X/S) / ( sigma * sqrt(T) ) + lambda * sigma * sqrt(T)
        x4 = np.log( self.barrier_price / self.underlying_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + self._lambda() * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x4

    def _x5(self):
    
        # x5 = ln(X/S) / ( sigma * sqrt(T) ) + b * sigma * sqrt(T)
        x5 = np.log( self.barrier_price / self.underlying_price ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + self._b() * self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return x5

    def _phi(self, option_type):

        # phi = 1 if Call and -1 if Put
        if ( option_type == True ):
            phi = 1
        else:
            phi = -1
        return phi

    def _phi_2(self):
    # phi function if using payout-at-maturity - barrier only options

        # phi = eta if Out and -eta if In
        if ( self.barrier_type == True ):
            phi = self._eta()
        else:
            phi = -self._eta()
        return phi

    def _eta(self):

        # eta = -1 if Up and 1 if Down
        if ( self.barrier_direction == True ):
            eta = -1
        else:
            eta = 1
        return eta
 
    def _A1(self, option_type):

        # A1 = S * exp(-q*T) * Phi( phi * x1 )
        A1 = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._phi(option_type) * self._x1(), 0, 1)
        return A1

    def _A2(self, option_type):

        # A2 = S * exp(-q*T) * Phi( phi * x2 )
        A2 = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._phi(option_type) * self._x2(), 0, 1)
        return A2

    def _A2_2(self):
    # A2 function if using payout-at-maturity - barrier only options

        # A2 = S * exp(-q*T) * Phi( phi * x2 )
        A2 = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._phi_2() * self._x2(), 0, 1)
        return A2

    def _A3(self):

        # A3 = S * exp(-q*T) * (X/S)^(2 * lambda) * Phi( eta * x3 )
        A3 = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * ( self.barrier_price / self.underlying_price )**( 2 * self._lambda() ) * norm.cdf(self._eta() * self._x3(), 0, 1)
        return A3

    def _A4(self):

        # A4 = S * exp(-q*T) * (X/S)^(2 * lambda) * Phi( eta * x4 )
        A4 = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * ( self.barrier_price / self.underlying_price )**( 2 * self._lambda() ) * norm.cdf(self._eta() * self._x4(), 0, 1)
        return A4

    def _C1(self, option_type, fixed_amount):

        # C1 = B * exp(-r*T) * Phi( phi * x1 - phi * sigma * sqrt(T) )
        C1 = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._phi(option_type) * self._x1() - self._phi(option_type) * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return C1

    def _C2(self, option_type, fixed_amount):

        # C2 = B * exp(-r*T) * Phi( phi * x2 - phi * sigma * sqrt(T) )
        C2 = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._phi(option_type) * self._x2() - self._phi(option_type) * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return C2

    def _C2_2(self, fixed_amount):
    # C2 function if using payout-at-maturity - barrier only options

        # C2 = B * exp(-r*T) * Phi( phi * x2 - phi * sigma * sqrt(T) )
        C2 = fixed_amount* np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._phi_2() * self._x2() - self._phi_2() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return C2

    def _C3(self, fixed_amount):

        # C3 = B * exp(-r*T) * (X/S)^(2 * lambda - 2) * Phi( eta * x3 - eta * sigma * sqrt(T) )
        C3 = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.barrier_price / self.underlying_price )**( 2 * self._lambda() - 2 ) * norm.cdf(self._eta() * self._x3() - self._eta() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return C3

    def _C4(self, fixed_amount):

        # C4 = B * exp(-r*T) * (X/S)^(2 * lambda - 2) * Phi( eta * x4 - eta * sigma * sqrt(T) )
        C4 = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( self.barrier_price / self.underlying_price )**( 2 * self._lambda() - 2 ) * norm.cdf(self._eta() * self._x4() - self._eta() * self.underlying_volatility * np.sqrt(self.time_to_maturity), 0, 1)
        return C4

    def _D(self, fixed_amount):

        # D = B * ( ( X / S )^(a + b) * Phi( eta * x5 ) + ( X / S )^(a - b) * Phi( eta * x5 - 2 * eta * b * sigma * sqrt(T) ) )
        D = fixed_amount * ( ( self.barrier_price / self.underlying_price )**( self._a() + self._b() ) * norm.cdf(self._eta() * self._x5(),0 , 1) + ( self.barrier_price / self.underlying_price )**( self._a() - self._b() ) * norm.cdf(self._eta() * self._x5() - 2 * self._eta() * self._b() * self.underlying_volatility * np.sqrt(self.time_to_maturity),0 , 1) )
        return D

    def _delta(self, price_function, *args):
    # Sensitivity of option price to a change in the underlying price

        delta = FiniteDifferenceMethods(
            self,
            price_function,
            "underlying_price",
            h = 1e-2
        ).first_derivative(*args)

        return delta

    def _gamma(self, price_function, *args):
    # Sensitivity of option delta to a change in the underlying price
    
        gamma = FiniteDifferenceMethods(
            self,
            price_function,
            "underlying_price",
            h = 1e-2
        ).second_derivative(*args)

        return gamma

    def _rho(self, price_function, *args):
    # Sensitivity of option price to a change in the risk-free rate

        rho = FiniteDifferenceMethods(
            self,
            price_function,
            "risk_free_rate",
            h = 1e-4
        ).first_derivative(*args)

        return rho

    def _theta(self, price_function, *args):
    # Sensitivity of option price to a change as time passes

        theta = - FiniteDifferenceMethods(
            self,
            price_function,
            "time_to_maturity",
            h = 1 / 365
        ).first_derivative(*args)

        return theta

    def _vega(self, price_function, *args):
    # Sensitivity of option price to a change in the underlying volatility

        vega = FiniteDifferenceMethods(
            self,
            price_function,
            "underlying_volatility",
            h = 1e-4
        ).first_derivative(*args)

        return vega

    def payout_at_hit_price(self, fixed_amount = None):
    # Return price of a payout at hit binary barrier
    # barrier_type is Out = True, In = False
    # binary_type is Cash = True, Asset = False

        if ( self.barrier_type == True ):               # Out barrier

            return print(f"For all payout at hit binary barrier options, only in barriers are defined.")

        else:                                           # In barrier

            if ( self.binary_type == True ):            # Cash-at-hit

                price = self._D(fixed_amount)
            else:                                       # Asset-at-hit

                # B = X, fixed amount is equal to the barrier price
                price = self._D(self.barrier_price)
            return price

    def payout_at_maturity_barrier_strike_price_price(self, fixed_amount = None):
    # Return price of a payout at hit binary barrier
    # barrier_type is Out = True, In = False
    # binary_type is Cash = True, Asset = False

        if ( self.barrier_type == True ):               # Out barrier

            return print(f"For all payout at hit binary barrier options, only in barriers are defined.")

        else:                                           # In barrier

            if ( self.binary_type == True ):            # Cash-at-hit

                price = self._D(fixed_amount)
            else:                                       # Asset-at-hit

                # B = X, fixed amount is equal to the barrier price
                price = self._D(self.barrier_price)
            return price

    def payout_at_maturity_barrier_price(self, fixed_amount = None):
    # Return price of a payout at maturity (barrier only) binary barrier
    # barrier_type is Out = True, In = False
    # binary_type is Cash = True, Asset = False

        if ( self.barrier_type == True ):                                   # Out barrier

            if ( self.binary_type == True ):                                # Cash-or-nothing

                price = self._C2_2(fixed_amount) - self._C4(fixed_amount)
            else:                                                           # Asset-or-nothing

                price = self._A2_2() - self._A4()
        else:                                                               # In barrier

            if ( self.binary_type == True ):                                # Cash-or-nothing

                price = self._C2_2(fixed_amount) + self._C4(fixed_amount)
            else:                                                           # Asset-or-nothing

                price = self._A2_2() + self._A4()
        return price

    def payout_at_maturity_barrier_strike_price(self, option_type, fixed_amount = None):
    # Return price of a payout at maturity (barrier and strike) binary barrier
    # barrier_direction is Up = True, Down = False
    # barrier_type is Out = True, In = False
    # option_type is Call = True, Put = False
    # binary_type is Cash = True, Asset = False

        if ( self.strike_price > self.barrier_price ):                                  # E > X
            if ( self.barrier_type == True ):                                               # Out
                if ( self.barrier_direction == True ):                                          # Up
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = 0
                        else:                                                                           # Put
                            price = self._C2(option_type, fixed_amount) - self._C4(fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = 0
                        else:                                                                           # Put
                            price = self._A2(option_type) - self._A4()
                else:                                                                           # Down
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C1(option_type, fixed_amount) - self._C3(fixed_amount)
                        else:                                                                           # Put
                            price = self._C1(option_type, fixed_amount) - self._C2(option_type, fixed_amount) + self._C3(fixed_amount) - self._C4(fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A1(option_type) - self._A3()
                        else:                                                                           # Put
                            price = self._A1(option_type) - self._A2(option_type) + self._A3() - self._A4()
            else:                                                                           # In
                if ( self.barrier_direction == True ):                                          # Up
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C1(option_type, fixed_amount)
                        else:                                                                           # Put
                            price = self._C1(option_type, fixed_amount) - self._C2(option_type, fixed_amount) + self._C4(fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A1(option_type)
                        else:                                                                           # Put
                            price = self._A1(option_type) - self._A2(option_type) + self._A3()
                else:                                                                           # Down
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C3(fixed_amount)
                        else:                                                                           # Put
                            price = self._C2(option_type, fixed_amount) - self._C3(fixed_amount) + self._C4(fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A3()
                        else:                                                                           # Put
                            price = self._A2(option_type) - self._A3() + self._A4()
        else:                                                                           # E <= X
            if ( self.barrier_type == True ):                                               # Out
                if ( self.barrier_direction == True ):                                          # Up
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C1(option_type, fixed_amount) - self._C2(option_type, fixed_amount) + self._C3(fixed_amount) - self._C4(fixed_amount)
                        else:                                                                           # Put
                            price = self._C1(option_type, fixed_amount) - self._C3(fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A1(option_type) - self._A2(option_type) + self._A3() - self._A4()
                        else:                                                                           # Put
                            price = self._A1(option_type) - self._A3()
                else:                                                                           # Down
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C2(option_type, fixed_amount) - self._C4(fixed_amount)
                        else:                                                                           # Put
                            price = 0
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A2(option_type) - self._A4()
                        else:                                                                           # Put
                            price = 0
            else:                                                                           # In
                if ( self.barrier_direction == True ):                                          # Up
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C2(option_type, fixed_amount) - self._C3(fixed_amount) + self._C4(fixed_amount)
                        else:                                                                           # Put
                            price = self._C3(fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A2(option_type) - self._A3() + self._A4()
                        else:                                                                           # Put
                            price = self._A3()
                else:                                                                           # Down
                    if ( self.binary_type == True ):                                                # Cash-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._C1(option_type, fixed_amount) - self._C2(option_type, fixed_amount) + self._C4(fixed_amount)
                        else:                                                                           # Put
                            price = self._C1(option_type, fixed_amount)
                    else:                                                                           # Asset-or-nothing
                        if ( option_type == True ):                                                     # Call
                            price = self._A1(option_type) - self._A2(option_type) + self._A4()
                        else:                                                                           # Put
                            price = self._A1(option_type)
        return price

    def payout_at_hit_delta(self, fixed_amount = None):

        delta = self._delta(self.payout_at_hit_price, fixed_amount)
        return delta

    def payout_at_hit_gamma(self, fixed_amount = None):

        gamma = self._gamma(self.payout_at_hit_price, fixed_amount)
        return gamma

    def payout_at_hit_rho(self, fixed_amount = None):

        rho = self._rho(self.payout_at_hit_price, fixed_amount)
        return rho

    def payout_at_hit_theta(self, fixed_amount = None):

        theta = self._theta(self.payout_at_hit_price, fixed_amount)
        return theta

    def payout_at_hit_vega(self, fixed_amount = None):

        vega = self._vega(self.payout_at_hit_price, fixed_amount)
        return vega

    def payout_at_maturity_barrier_delta(self, fixed_amount = None):

        delta = self._delta(self.payout_at_maturity_barrier_price, fixed_amount)
        return delta

    def payout_at_maturity_barrier_gamma(self, fixed_amount = None):

        gamma = self._gamma(self.payout_at_maturity_barrier_price, fixed_amount)
        return gamma

    def payout_at_maturity_barrier_rho(self, fixed_amount = None):

        rho = self._rho(self.payout_at_maturity_barrier_price, fixed_amount)
        return rho

    def payout_at_maturity_barrier_theta(self, fixed_amount = None):

        theta = self._theta(self.payout_at_maturity_barrier_price, fixed_amount)
        return theta

    def payout_at_maturity_barrier_vega(self, fixed_amount = None):

        vega = self._vega(self.payout_at_maturity_barrier_price, fixed_amount)
        return vega

    def payout_at_maturity_barrier_strike_delta(self, option_type, fixed_amount = None):

        delta = self._delta(self.payout_at_maturity_barrier_strike_price, option_type, fixed_amount)
        return delta

    def payout_at_maturity_barrier_strike_gamma(self, option_type, fixed_amount = None):

        gamma = self._gamma(self.payout_at_maturity_barrier_strike_price, option_type, fixed_amount)
        return gamma

    def payout_at_maturity_barrier_strike_rho(self, option_type, fixed_amount = None):

        rho = self._rho(self.payout_at_maturity_barrier_strike_price, option_type, fixed_amount)
        return rho

    def payout_at_maturity_barrier_strike_theta(self, option_type, fixed_amount = None):

        theta = self._theta(self.payout_at_maturity_barrier_strike_price, option_type, fixed_amount)
        return theta

    def payout_at_maturity_barrier_strike_vega(self, option_type, fixed_amount = None):

        vega = self._vega(self.payout_at_maturity_barrier_strike_price, option_type, fixed_amount)
        return vega
