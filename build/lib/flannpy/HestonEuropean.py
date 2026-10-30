
# A Python Library for Option Pricing and Sensitivities

# European Options (Heston)

import numpy as np

from .NumericalMethods import FiniteDifferenceMethods

# Heston Stochastic Volatility Model for European Options

class HestonEuropean:

    # S             :   underlying_price
    # E             :   strike_price
    # tau           :   time_to_maturity
    # r             :   risk_free_rate
    # q             :   dividend_yield
    # call / put    :   option_type
    # v0            :   initial_variance
    # kappa         :   mean_reversion_rate
    # theta         :   long_run_variance
    # xi            :   volatility_of_variance
    # rho           :   correlation
    # lambda        :   variance_risk_premium
    # phi           :   fourier_variable

    def __init__(self, underlying_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, option_type, initial_variance, mean_reversion_rate, long_run_variance, volatility_of_variance, correlation, variance_risk_premium):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.option_type = option_type
        self.initial_variance = initial_variance
        self.mean_reversion_rate = mean_reversion_rate
        self.long_run_variance = long_run_variance
        self.volatility_of_variance = volatility_of_variance
        self.correlation = correlation
        self.variance_risk_premium = variance_risk_premium
    # Constructor which intialises the object's instance variables

    def __str__(self):
        return (
            f"Price : {self.price():>8.4f}\n"
            f"Delta : {self.delta():>8.4f}\n"
            f"Gamma : {self.gamma():>8.4f}\n"
            f"Rho   : {self.rho():>8.4f}\n"
            f"Theta : {self.theta():>8.4f}\n"
            f"Vanna : {self.vanna():>8.4f}\n"
            f"Vega  : {self.vega():>8.4f}\n"
            f"Volga : {self.volga():>8.4f}"
        )
    # String to return a summary

    def _characteristic_function(self, fourier_variable):
    # Characteristic function defined via the inverse Fourier transform

        # Calculate constants
        a = self.mean_reversion_rate * self.long_run_variance
        b = self.mean_reversion_rate + self.variance_risk_premium

        # Define rxpi for easier computation since it is a common variable
        rxpi = self.correlation * self.volatility_of_variance * fourier_variable * 1j

        # Define d parameter given phi and b
        d = np.sqrt( ( rxpi - b )**2 + ( fourier_variable * 1j + fourier_variable**2) * self.volatility_of_variance**2 )

        # Define g parameter given rxpi, b, and d
        g = ( b - rxpi - d ) / ( b - rxpi + d )

        # Calculate characteristic function by components
        component1 = np.exp( (self.risk_free_rate - self.dividend_yield) * fourier_variable * 1j * self.time_to_maturity)
        component2 = self.underlying_price**( fourier_variable * 1j ) * ( ( 1 - g * np.exp(-d * self.time_to_maturity) ) / ( 1 - g ) )**(- (2 * a) / (self.volatility_of_variance**2) )
        component3 = np.exp( a * self.time_to_maturity * ( b - rxpi - d ) / self.volatility_of_variance**2 + self.initial_variance * ( b - rxpi - d ) * ( ( 1 - np.exp(-d * self.time_to_maturity) ) / ( 1 - g * np.exp(-d * self.time_to_maturity) ) ) / self.volatility_of_variance**2)

        varphi = component1 * component2 * component3

        return varphi

    def _rectangular_integration(self):
    # Determine Heston price via rectangular integration

        integral, umax, N = 0, 100, 20000 # 100 is taken compared to infinity as it will compute approximately
        dphi=umax/N # dphi is width

        for i in range(1,N):
            # Rectangular integration
            phi = dphi * ( 2 * i + 1 ) / 2 # Midpoint to calculate height
            numerator = np.exp( - self.risk_free_rate * self.time_to_maturity ) * ( self._characteristic_function(phi - 1j) - self.strike_price * self._characteristic_function(phi))
            denominator = 1j * phi * self.strike_price**(1j * phi)

            integral += dphi * numerator / denominator # Calculate f(midpoint) and multiply by width over the integration space

        return integral

    def price(self):
    # Expected value of the option at present
    # option_type is Call = True, Put = False

        if ( self.option_type == True ):

            # C = 0.5 * ( S * exp(-q*T) - E * exp(-r*T) ) + ( 1 / pi ) * integral (as defined above)
            price = np.real( ( self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) - self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) ) / 2 + self._rectangular_integration() / np.pi)
        else:

            # P = 0.5 * ( E * exp(-r*T) - S * exp(-q*T) ) + ( 1 / pi ) * integral (as defined above)
            price = np.real( ( self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) - self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) ) / 2 + self._rectangular_integration() / np.pi)
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

    def vanna(self):
    # Sensitivity of option vega to a change in the underlying price

        # Convert initial variance to volatility
        initial_volatility = np.sqrt(self.initial_variance)

        mixed = FiniteDifferenceMethods(
            self,
            self.price,
            "underlying_price",
            h = 1e-2
        ).mixed_derivative(
            "initial_variance",
            h2 = 1e-4
        )

        # Apply chain rule
        vanna = 2 * initial_volatility * mixed

        return vanna

    def vega(self):
    # Sensitivity of option price to a change in the initial variance

        # Convert initial variance to volatility
        initial_volatility = np.sqrt(self.initial_variance)

        first = FiniteDifferenceMethods(
            self,
            self.price,
            "initial_variance",
            h = 1e-4
        ).first_derivative()

        # Apply chain rule
        vega = 2 * initial_volatility * first

        return vega

    def volga(self):
    # Sensitivity of option vega to a change in the initial variance

        # Create instance
        calculate_derivatives = FiniteDifferenceMethods(
            self,
            self.price,
            "initial_variance",
            h = 1e-4
        )

        first = calculate_derivatives.first_derivative()
        second = calculate_derivatives.second_derivative()

        # Apply chain rule
        volga = 2 * first + 4 * self.initial_variance * second

        return volga
