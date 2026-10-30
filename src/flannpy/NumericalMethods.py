
# A Python Library for Option Pricing and Sensitivities, and Volatility Modelling

# Numerical Methods

class NewtonRaphson:

    # loss_function
    # loss_gradient
    # initial_guess
    # tolerance
    # max_iteration

    def __init__(
            self,
            loss_function,
            loss_gradient,
            initial_guess,
            tolerance = 1e-3,
            max_iteration = 20
        ):
        self.loss_function = loss_function
        self.loss_gradient = loss_gradient
        self.initial_guess = initial_guess
        self.tolerance = tolerance
        self.max_iteration = max_iteration
    # Constructor which intialises the object's instance variables

    def newton_raphson(self):
    # Function to solve for an optimal value iteratively using the Newton-Raphson method

        # Track whether convergence criterion is satisfied before reaching max number of iteration
        converged = False

        # Make an initial guess
        x = self.initial_guess

        for _ in range(self.max_iteration):

            # Calculate the loss function
            loss_function_value = self.loss_function(x)

            # Check if the loss is less than the tolerance
            if abs(loss_function_value) < self.tolerance:
                converged = True
                break # If yes, then stop

            else: # If no, then continue another iteration

                # Calculate the gradient of the loss function
                loss_gradient_value = self.loss_gradient(x)

                # Update the guess using the Newton-Raphson formula
                x -= loss_function_value / loss_gradient_value

        # Warn user if the convergence criterion was never met after max number of iterations
        if not converged:
            print(f"Series not converged within tolerance using {self.max_iteration} iterations.")

        return x

class FiniteDifferenceMethods:

    # pricing_model is the instance of the underlying model class of which the sensitivity is calculated from
    # pricing_function is the function that calculates the option's value
    # attribute is the name of the parameter being varied
    # h is the small increment used in the finite difference approximation

    def __init__(
            self,
            pricing_model,
            pricing_function,
            attribute,
            h
        ):
        self.pricing_model = pricing_model
        self.pricing_function = pricing_function
        self.attribute = attribute
        self.h = h
    # Constructor which intialises the object's instance variables

    def _finite_difference_value(self, changes, *args):

        # Store the original values of all attributes that will be changed, these values are used to restore the object's state after pricing
        # Only reason dict is used is because there are two attributes that need changing in mixed_derivative
        original = {
            attr: getattr(self.pricing_model, attr)
            for attr in changes
        }

        try:
            # Temporarily apply the specified attribute changes
            for attr, value in changes.items():
                setattr(self.pricing_model, attr, value)

            # Calculate the option price using the modified attribute
            return self.pricing_function(*args)
    
        finally:
            # Restore the original attribute values
            for attr, value in original.items():
                setattr(self.pricing_model, attr, value)

    def first_derivative(self, *args):

        # Get the current value of the attribute
        original = getattr(self.pricing_model, self.attribute)

        # Calculate prices at a small positive and negative shift
        up = self._finite_difference_value({self.attribute: original + self.h}, *args)
        down = self._finite_difference_value({self.attribute: original - self.h}, *args)

        # Approximate the first derivative using the central difference approximation
        central_difference_approximation = ( up - down ) / ( 2 * self.h )

        return central_difference_approximation

    def second_derivative(self, *args):

        # Get the current value of the attribute
        original = getattr(self.pricing_model, self.attribute)

        # Calculate prices above and below the current value, and the unmodiefied value
        up = self._finite_difference_value({self.attribute: original + self.h}, *args)
        mid = self.pricing_function(*args)
        down = self._finite_difference_value({self.attribute: original - self.h}, *args)

        # Approximate the second derivative using the central difference approximation
        central_difference_approximation = ( up - 2 * mid + down ) / ( self.h**2 )

        return central_difference_approximation

    def mixed_derivative(self, attribute2, h2, *args):
    # self.attribute becomes attribute1
    # self.h becomes h1

        # Get the current values of the two attributes
        x0 = getattr(self.pricing_model, self.attribute)
        y0 = getattr(self.pricing_model, attribute2)

        # Calculate the option price at all four combinations of positive and negative shifts
        prices = {
            (sx, sy): self._finite_difference_value({
                self.attribute: x0 + sx * self.h,
                attribute2: y0 + sy * h2,
            }, *args)
            for sx in (1, -1)
            for sy in (1, -1)
        }

        # Retrieve the prices corresponding to each combination of shifts
        up_up = prices[(1, 1)]
        up_down = prices[(1, -1)]
        down_up = prices[(-1, 1)]
        down_down = prices[(-1, -1)]

        # Approximate the mixed second derivative using the central difference approximation
        central_difference_approximation = ( up_up - up_down - down_up + down_down ) / ( 4 * self.h * h2 )

        return central_difference_approximation
