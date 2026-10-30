
# A Python Library for Option Pricing and Sensitivities

# Binary Options

import numpy as np
from scipy.stats import norm

# Black-Scholes Model for Binary Options

class BlackScholesBinary:

    # S             :   underlying_price
    # T             :   time_to_maturity
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # sigma         :   underlying_volatility

    # call / put    :   option_type
    # B             :   fixed_amount
    # E             :   strike_price
    # E_1           :   exercise_strike_price
    # E_2           :   payoff_strike_price
    # E_d           :   lower_strike_price
    # E_u           :   upper_strike_price

    def __init__(self, underlying_price, time_to_maturity, risk_free_rate, dividend_yield, underlying_volatility):
        self.underlying_price = underlying_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.underlying_volatility = underlying_volatility
    # Constructor which intialises the object's instance variables

    def _d1(self, strike_price):
    # d1 determines how many standard deviations the strike price is from the mean
        
        # d1 = ( ln(S/E) + ( r - q + 0.5 * sigma^2 ) * T ) / ( sigma * sqrt(T) )
        d1 = ( np.log(self.underlying_price / strike_price) + (self.risk_free_rate - self.dividend_yield + 0.5 * self.underlying_volatility**2) * self.time_to_maturity ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return d1

    def _d2(self, strike_price):
    # d2 adjusts d1 to use the median from the mean in a lognormal distribution

        # d2 = d1 - sigma * sqrt(T)
        d2 = self._d1(strike_price) - self.underlying_volatility * np.sqrt(self.time_to_maturity)
        return d2

    def _gap_greek(self, pointer_function_1, pointer_function_2, exercise_strike_price, payoff_strike_price, option_type):
    # Function to calculate greeks using pointer functions for asset-or-nothing and unit call-or-nothing greeks respectively

        # Find the respective binary greek
        aon_greek = pointer_function_1(exercise_strike_price, option_type)
        con_greek = pointer_function_2(1, exercise_strike_price, option_type)

        # Express a gap greek as a linear combination of binary option greeks
        greek = aon_greek - payoff_strike_price * con_greek

        return greek

    def _supershare_greek(self, pointer_function, lower_strike_price, upper_strike_price):
    # Function to calculate greeks using a pointer function for asset-or-nothing greeks

        # Find the respective greeks using the lower_strike_price and upper_strike_price
        lower_greek = pointer_function(lower_strike_price, option_type = True)
        upper_greek = pointer_function(upper_strike_price, option_type = True)

        # Express a supershare greek as a linear combination of asset-or-nothing greeks
        greek = ( 1 / lower_strike_price ) * ( lower_greek - upper_greek )

        return greek

    def cash_or_nothing_price(self, fixed_amount, strike_price, option_type):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( option_type == True ):

            # C = B * exp(-r*T) * Phi(d2)
            price = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._d2(strike_price), 0, 1)
        else:

            # P = B * exp(-r*T) * Phi(-d2)
            price = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(-self._d2(strike_price), 0, 1)
        return price

    def cash_or_nothing_delta(self, fixed_amount, strike_price, option_type):
    # Delta measures how the option price will change with a change in the underlying asset price
    # option_type is Call = True, Put = False

        if ( option_type == True ):

            # Delta = ( B * exp(-r*T) * phi(d2) ) / ( S * sigma * sqrt(T) )
            delta = ( fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.pdf(self._d2(strike_price), 0, 1) ) / ( self.underlying_price * self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        else:

            # Delta = ( -B * exp(-r*T) * phi(d2) ) / ( S * sigma * sqrt(T) )
            delta = ( -fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.pdf(self._d2(strike_price), 0, 1) ) / ( self.underlying_price * self.underlying_volatility * np.sqrt(self.time_to_maturity) )
        return delta

    def cash_or_nothing_gamma(self, fixed_amount, strike_price, option_type):
    # Gamma measures the rate of change of Delta
    # option_type is Call = True, Put = False
    
        if ( option_type == True ):

            # Gamma = ( -B * exp(-r*T) * phi(d2) * d1 ) / ( S^2 * sigma^2 * T )
            gamma = ( -fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.pdf(self._d2(strike_price), 0, 1) * self._d1(strike_price) ) / ( self.underlying_price**2 * self.underlying_volatility**2 * self.time_to_maturity )
        else:

            # Gamma = ( B * exp(-r*T) * phi(d2) * d1 ) / ( S^2 * sigma^2 * T )
            gamma = ( fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.pdf(self._d2(strike_price), 0, 1) * self._d1(strike_price) ) / ( self.underlying_price**2 * self.underlying_volatility**2 * self.time_to_maturity )
        return gamma

    def cash_or_nothing_rho(self, fixed_amount, strike_price, option_type):
    # Rho measures how the option price will change with a change in the risk-free interest rate
    # option_type is Call = True, Put = False
        
        if ( option_type == True ):

            # Rho = B * exp(-r*T) * ( ( np.sqrt(T) / sigma ) * phi(d2) - T * Phi(d2) )
            rho = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( ( np.sqrt(self.time_to_maturity) / self.underlying_volatility ) * norm.pdf(self._d2(strike_price), 0, 1) - self.time_to_maturity * norm.cdf(self._d2(strike_price), 0, 1) )
        else:

            # Rho = -B * exp(-r*T) * ( ( np.sqrt(T) / sigma ) * phi(d2) + T * Phi(-d2) )
            rho = -fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( ( np.sqrt(self.time_to_maturity) / self.underlying_volatility ) * norm.pdf(self._d2(strike_price), 0, 1) + self.time_to_maturity * norm.cdf(-self._d2(strike_price), 0, 1) )
        return rho

    def cash_or_nothing_theta(self, fixed_amount, strike_price, option_type):
    # Theta measures how the option price will change as time passes
    # option_type is Call = True, Put = False

        if ( option_type == True ):
            
            # Theta = B * exp(-r*T) * ( ( phi(d2) / ( 2 * sigma * T^(3/2) ) * ( ln(S/E) - ( r - q - ( sigma^2 / 2 ) ) * T ) + r * Phi(d2) )
            theta = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( ( norm.pdf(self._d2(strike_price)) / ( 2 * self.underlying_volatility * self.time_to_maturity**(3/2) ) ) * ( np.log( self.underlying_price / strike_price ) - ( self.risk_free_rate - self.dividend_yield - ( self.underlying_volatility**2 / 2 ) ) * self.time_to_maturity ) + self.risk_free_rate * norm.cdf(self._d2(strike_price), 0, 1) )
        else:

            # Theta = B * exp(-r*T) * ( ( -phi(d2) / ( 2 * sigma * T^(3/2) ) * ( ln(S/E) - ( r - q - ( sigma^2 / 2 ) ) * T ) + r * Phi(-d2) )
            theta = fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) * ( ( -norm.pdf(self._d2(strike_price)) / ( 2 * self.underlying_volatility * self.time_to_maturity**(3/2) ) ) * ( np.log( self.underlying_price / strike_price ) - ( self.risk_free_rate - self.dividend_yield - ( self.underlying_volatility**2 / 2 ) ) * self.time_to_maturity ) + self.risk_free_rate * norm.cdf(-self._d2(strike_price), 0, 1) )
        return theta

    def cash_or_nothing_vega(self, fixed_amount, strike_price, option_type):
    # Vega measures how the option price will change with a change in the underlying asset volatility
    # option_type is Call = True, Put = False
    
        if ( option_type == True ):

            # Vega = ( ( -B * exp(-r*T) ) / sigma ) * d1 * phi(d2)
            vega = ( ( -fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) ) / self.underlying_volatility ) * self._d1(strike_price) * norm.pdf(self._d2(strike_price), 0, 1)
        else:

            # Vega = ( ( B * exp(-r*T) ) / sigma ) * d1 * phi(d2)
            vega = ( ( fixed_amount * np.exp(-self.risk_free_rate * self.time_to_maturity) ) / self.underlying_volatility ) * self._d1(strike_price) * norm.pdf(self._d2(strike_price), 0, 1)
        return vega
    
    def asset_or_nothing_price(self, strike_price, option_type):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( option_type == True ):

            # C = S * exp(-q*T) * Phi(d1)
            price = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._d1(strike_price), 0, 1)
        else:

            # C = S * exp(-q*T) * Phi(-d1)
            price = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(-self._d1(strike_price), 0, 1)
        return price

    def asset_or_nothing_delta(self, strike_price, option_type):
    # Delta measures how the option price will change with a change in the underlying asset price
    # option_type is Call = True, Put = False

        if ( option_type == True ):
        
            # Delta = ( exp(-q*T) * phi(d1) ) / ( sigma * sqrt(T) ) + exp(-q*T) * Phi(d1)
            delta = ( np.exp(-self.dividend_yield * self.time_to_maturity) * norm.pdf(self._d1(strike_price), 0, 1) ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._d1(strike_price), 0, 1)
        else:

            # Delta = ( -exp(-q*T) * phi(d1) ) / ( sigma * sqrt(T) ) + exp(-q*T) * Phi(-d1)
            delta = ( -np.exp(-self.dividend_yield * self.time_to_maturity) * norm.pdf(self._d1(strike_price), 0, 1) ) / ( self.underlying_volatility * np.sqrt(self.time_to_maturity) ) + np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(-self._d1(strike_price), 0, 1)
        return delta

    def asset_or_nothing_gamma(self, strike_price, option_type):
    # Gamma measures the rate of change of Delta
    # option_type is Call = True, Put = False

        if ( option_type == True ):
        
            # Gamma = ( -exp(-q*T) * phi(d2) * d1 ) / ( S * sigma^2 * T )
            gamma = ( -np.exp(-self.dividend_yield * self.time_to_maturity) * norm.pdf(self._d2(strike_price), 0, 1) * self._d1(strike_price) ) / ( self.underlying_price * self.underlying_volatility**2 * self.time_to_maturity )
        else:

            # Gamma = ( exp(-q*T) * phi(d2) * d1 ) / ( S * sigma^2 * T )
            gamma = ( np.exp(-self.dividend_yield * self.time_to_maturity) * norm.pdf(self._d2(strike_price), 0, 1) * self._d1(strike_price) ) / ( self.underlying_price * self.underlying_volatility**2 * self.time_to_maturity )
        return gamma

    def asset_or_nothing_rho(self, strike_price, option_type):
    # Rho measures how the option price will change with a change in the risk-free interest rate
    # option_type is Call = True, Put = False

        if ( option_type == True ):
        
            # Rho = ( S * exp(-q*T) * sqrt(T) * phi(d1) ) / sigma
            rho = ( self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * np.sqrt(self.time_To_maturity) * norm.pdf(self._d1(strike_price), 0, 1) ) / self.underlying_volatility
        else:

            # Rho = ( -S * exp(-q*T) * sqrt(T) * phi(d1) ) / sigma
            rho = ( -self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * np.sqrt(self.time_To_maturity) * norm.pdf(self._d1(strike_price), 0, 1) ) / self.underlying_volatility
        return rho

    def asset_or_nothing_theta(self, strike_price, option_type):
    # Theta measures how the option price will change as time passes
    # option_type is Call = True, Put = False

        if ( option_type == True ):
            
            # Theta = S * exp(-q*T) * ( ( phi(d1) / ( 2 * sigma * T^(3/2) ) * ( ln(S/E) - ( r - q + ( sigma^2 / 2 ) ) * T ) + q * Phi(d1) )
            theta = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * ( ( norm.pdf(self._d1(strike_price)) / ( 2 * self.underlying_volatility * self.time_to_maturity**(3/2) ) ) * ( np.log( self.underlying_price / strike_price ) - ( self.risk_free_rate - self.dividend_yield + ( self.underlying_volatility**2 / 2 ) ) * self.time_to_maturity ) + self.dividend_yield * norm.cdf(self._d1(strike_price), 0, 1) )
        else:

            # Theta = S * exp(-q*T) * ( ( -phi(d1) / ( 2 * sigma * T^(3/2) ) * ( ln(S/E) - ( r - q + ( sigma^2 / 2 ) ) * T ) + q * Phi(-d1) )
            theta = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * ( ( -norm.pdf(self._d1(strike_price)) / ( 2 * self.underlying_volatility * self.time_to_maturity**(3/2) ) ) * ( np.log( self.underlying_price / strike_price ) - ( self.risk_free_rate - self.dividend_yield + ( self.underlying_volatility**2 / 2 ) ) * self.time_to_maturity ) + self.dividend_yield * norm.cdf(-self._d1(strike_price), 0, 1) )
        return theta

    def asset_or_nothing_vega(self, strike_price, option_type):
    # Vega measures how the option price will change with a change in the underlying asset volatility
    # option_type is Call = True, Put = False

        if ( option_type == True ):
        
            # Vega = ( -S * exp(-q*T) * d2 * phi(d1) ) / sigma
            vega = ( -self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * self._d2(strike_price) * norm.pdf(self._d1(strike_price), 0, 1) ) / self.underlying_volatility
        else:

            # Vega = ( S * exp(-q*T) * d2 * phi(d1) ) / sigma
            vega = ( self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * self._d2(strike_price) * norm.pdf(self._d1(strike_price), 0, 1) ) / self.underlying_volatility
        return vega

    def gap_price(self, exercise_strike_price, payoff_strike_price, option_type):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( option_type == True ):

            # C = S * exp(-q*T) * Phi(d1) - E_2 * exp(-r*T) * Phi(d2)
            price = self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(self._d1(exercise_strike_price), 0, 1) - payoff_strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(self._d2(exercise_strike_price), 0, 1)
        else:

            # P = E_2 * exp(-r*T) * Phi(-d2) - S * exp(-q*T) * Phi(-d1)
            price = payoff_strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(-self._d2(exercise_strike_price), 0, 1) - self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(-self._d1(exercise_strike_price), 0, 1)
        return price

    def gap_delta(self, exercise_strike_price, payoff_strike_price, option_type):
    # Delta measures how the option price will change with a change in the underlying asset price
    # option_type is Call = True, Put = False

        delta = self._gap_greek(
            self.asset_or_nothing_delta,
            self.cash_or_nothing_delta,
            exercise_strike_price,
            payoff_strike_price,
            option_type
        )

        return delta

    def gap_gamma(self, exercise_strike_price, payoff_strike_price, option_type):
    # Gamma measures the rate of change of Delta
    # option_type is Call = True, Put = False

        gamma = self._gap_greek(
            self.asset_or_nothing_gamma,
            self.cash_or_nothing_gamma,
            exercise_strike_price,
            payoff_strike_price,
            option_type
        )

        return gamma

    def gap_rho(self, exercise_strike_price, payoff_strike_price, option_type):
    # Rho measures how the option price will change with a change in the risk-free interest rate
    # option_type is Call = True, Put = False

        rho = self._gap_greek(
            self.asset_or_nothing_rho,
            self.cash_or_nothing_rho,
            exercise_strike_price,
            payoff_strike_price,
            option_type
        )

        return rho

    def gap_theta(self, exercise_strike_price, payoff_strike_price, option_type):
    # Theta measures how the option price will change as time passes
    # option_type is Call = True, Put = False

        theta = self._gap_greek(
            self.asset_or_nothing_theta,
            self.cash_or_nothing_theta,
            exercise_strike_price,
            payoff_strike_price,
            option_type
        )

        return theta

    def gap_vega(self, exercise_strike_price, payoff_strike_price, option_type):
    # Vega measures how the option price will change with a change in the underlying asset volatility
    # option_type is Call = True, Put = False

        vega = self._gap_greek(
            self.asset_or_nothing_vega,
            self.cash_or_nothing_vega,
            exercise_strike_price,
            payoff_strike_price,
            option_type
        )

        return vega

    def supershare_price(self, lower_strike_price, upper_strike_price):
    # Expected value of the option at present

        # V = ( ( S * exp(-q*T) ) / E_d ) * ( Phi(d1(E_d)) - Phi(d1(E_u)) )
        price = ( ( self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) ) / lower_strike_price ) * ( norm.cdf(self._d1(lower_strike_price), 0, 1) - norm.cdf(self._d1(upper_strike_price), 0, 1) )
        return price


    def supershare_delta(self, lower_strike_price, upper_strike_price):
    # Delta measures how the option price will change with a change in the underlying asset price

        delta = self._supershare_greek(
            self.asset_or_nothing_delta,
            lower_strike_price,
            upper_strike_price
        )

        return delta

    def supershare_gamma(self, lower_strike_price, upper_strike_price):
    # Gamma measures the rate of change of Delta

        gamma = self._supershare_greek(
            self.asset_or_nothing_gamma,
            lower_strike_price,
            upper_strike_price
        )

        return gamma

    def supershare_rho(self, lower_strike_price, upper_strike_price):
    # Rho measures how the option price will change with a change in the risk-free interest rate

        rho = self._supershare_greek(
            self.asset_or_nothing_rho,
            lower_strike_price,
            upper_strike_price
        )

        return rho

    def supershare_theta(self, lower_strike_price, upper_strike_price):
    # Theta measures how the option price will change as time passes

        theta = self._supershare_greek(
            self.asset_or_nothing_theta,
            lower_strike_price,
            upper_strike_price
        )

        return theta

    def supershare_vega(self, lower_strike_price, upper_strike_price):
    # Vega measures how the option price will change with a change in the underlying asset volatility

        vega = self._supershare_greek(
            self.asset_or_nothing_vega,
            lower_strike_price,
            upper_strike_price
        )

        return vega
