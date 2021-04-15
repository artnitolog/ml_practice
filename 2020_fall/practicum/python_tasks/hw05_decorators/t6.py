# %%
from functools import wraps


def substitutive(f, prev_args=None):
    if prev_args is None:
        prev_args = tuple()
    @wraps(f)
    def wrap(*args):
        now_args = prev_args + args
        if f.__code__.co_argcount == len(now_args):
            return f(*now_args)
        elif f.__code__.co_argcount < len(now_args):
            raise TypeError
        else:
            return substitutive(f, now_args)
    return wrap
