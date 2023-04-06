"""Functions that check types."""


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_tuple_of_ints(x):
    if not isinstance(x, tuple):
        return False
    for e in x:
        if not is_int(e):
            return False
    return True


def is_list(x):
    return isinstance(x, list)


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False
