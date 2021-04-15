# %%
def linearize(obj):
    for cur in obj:
        if cur == obj:
            yield obj
        else:
            try:
                for deeper_cur in linearize(cur):
                    yield deeper_cur
            except TypeError:
                yield cur
# %%
