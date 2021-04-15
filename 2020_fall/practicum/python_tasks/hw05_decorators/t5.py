# %%
from functools import wraps


def check_arguments(*types):
    def checker(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            # print(args, types)
            if len(types) > len(args):
                raise TypeError
            for obj, t in zip(args, types):
                if not isinstance(obj, t):
                    raise TypeError
            return f(*args, **kwargs)
        return wrap
    return checker
