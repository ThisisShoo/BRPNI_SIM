"""Testing ground for new features, or to run a specific function and see what's wrong with it"""
import numpy as np

def power_series(x, a, b, c, d, e, f):
    """A 5-terms power series function"""
    output = a + b * x + c * x**2 + d * x**3 + e * x**4 + f * x**5
    return output

print(power_series(1, 1, 1, 1, 1, 1, 1))
