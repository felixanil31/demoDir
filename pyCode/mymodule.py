"""mymodule: a tiny demo module for Chapter 1."""

PI = 3.141592653589793

def square(x):
    """Return x squared (x*x)."""
    return x * x

def hypotenuse(a, b):
    """Return sqrt(a^2 + b^2) without importing math here."""
    return (a*a + b*b) ** 0.5

class Greeter:
    def __init__(self, name):
        self.name = name
    def hello(self):
        return f"Hello, {self.name}!"
