class Polynomial:
    def __init__(self, *coefficients_):
        self.coefficients = coefficients_

    def __call__(self, x):
        res = 0
        for coef in self.coefficients[:0:-1]:
            res += coef
            res *= x
        res += self.coefficients[0]
        return res
