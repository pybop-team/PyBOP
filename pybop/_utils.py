import re

import numpy as np


def add_spaces(string):
    """
    Return the class name as a string with spaces before each new capitalised word.
    """
    re_outer = re.compile(r"([^A-Z ])([A-Z])")
    re_inner = re.compile(r"(?<!^)([A-Z])([^A-Z])")
    return re_outer.sub(r"\1 \2", re_inner.sub(r" \1\2", string))


def is_numeric(x):
    """
    Check if a variable is numeric.
    """
    return isinstance(x, int | float | np.number)
