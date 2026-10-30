
# A Python Library for Option Pricing and Sensitivities

# Infinite Series

class InfiniteSeries:

    # function_pointer
    # tolerance
    # max_iteration

    def __init__(
            self,
            function_pointer,
            tolerance = 1e-6,
            max_iteration = 100
        ):
        self.function_pointer = function_pointer
        self.tolerance = tolerance
        self.max_iteration = max_iteration
    # Constructor which intialises the object's instance variables

    def summation(self):
    # Sum_{n = -infinity}^{infinity} function_pointer

        # Track whether convergence criterion is satisfied before reaching max number of iterations
        converged = False

        # Start with central term n = 0
        total = self.function_pointer(0)

        # Evaluate terms symmetrically about zero
        for n in range(1, self.max_iteration + 1):

            positive = self.function_pointer(n)
            negative = self.function_pointer(-n)

            # Add pairs together to running total each iteration
            total += positive + negative

            # Relative convergence test, are the terms added sufficiently small relative to the accumulated total
            # Thus, the loop breaks if the terms are assumed negligible and summation stops
            if ( ( abs(positive) + abs(negative) ) < self.tolerance * max(abs(total), 1e-4) ):
                converged = True
                break

        # Warn user if the convergence criterion was never met after max number of iterations
        if not converged:
            print(f"Series not converged within tolerance using {self.max_iteration} iterations.")

        return total
