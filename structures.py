import numpy as np
import inspect

from scipy.special import erfinv


def prior_structure(function=None, shape=(1,), label=None, periodic=False, reflective=False,**kwargs):
    """
    creates a dictionary with appropriate structure for the prior distribution

    inputs:
        function:   None | callable
            None:       no sampling performed, only value in params is returned
            callable:   function to be applied to the parameters
        kwargs:     dict
            key-value pairs of parameters to be passed to the function

        TODO:
            * should this be transformed to a class?
    """

    ### check if function satisfies a couple of conditions
    if function is None:
        ### watch out: name of variable is arbitrary!
        if len(kwargs) != 1:
            raise ValueError(
                "If function is None, kwargs must contain exactly one entry."
            )
    else:
        if not callable(function):
            raise ValueError("function must be callable if not None.")
        # Try to check if kwargs match the function signature
        sig = inspect.signature(function)
        params = set(sig.parameters.keys())
        missing = params - set(kwargs.keys()) - {"x"}
        if missing:
            raise ValueError(f"Missing required kwargs for function: {missing}")

    ### return dictionary of according structure
    return {
        # "hierarchical": hierarchical,
        "label": label,
        "has_meta": any(isinstance(val, dict) for val in kwargs.values()),
        "shape": shape,
        "sample": not (function is None),
        "function": function,
        "parameters": kwargs,
        "periodic": periodic,
        "reflective": reflective,
    }

def halfnorm_ppf(x, loc, scale): 
    return loc + scale * np.sqrt(2) * erfinv(x)


def norm_ppf(x, mean, sigma): 
    return mean + sigma * np.sqrt(2) * erfinv(2 * x - 1)


def bounded_flat(x, low, high):
    return x * (high - low) + low


def build_key(key, tag, tag_idx=0):
    return f"{key}_{tag}{tag_idx}"


import re
from typing import Iterable, List, Optional, Tuple


def parse_name_and_indices(
    s: str, literals: Iterable[str]
) -> Tuple[str, List[Optional[int]]]:
    """
    Returns (variable_name, [idx_or_None per literal in the same order]).
    Variable name = prefix before the first <literal><digits> token.
    """

    if isinstance(literals, str):
        lits = literals.split()
    else:
        lits = list(literals)

    alts = "|".join(map(re.escape, lits))
    # Don't match inside letter-words; allow underscores and punctuation as separators.
    rx = re.compile(rf"_(?<![A-Za-z])({alts})(\d+)(?![A-Za-z])")

    found = {}
    first_pos = None

    for m in rx.finditer(s):
        lit, num = m.group(1), int(m.group(2))
        if lit not in found:  # keep only first per literal
            found[lit] = (m.start(), num)
            if first_pos is None or m.start() < first_pos:
                first_pos = m.start()

    # Remove all matches from the string and output the remaining string
    name = rx.sub("", s)

    # name = s[: first_pos - 1] if first_pos is not None else s
    indices: List[Optional[int]] = [
        found[lit][1] if lit in found else None for lit in lits
    ]
    return name, indices


def build_distr_structure_from_params(
    params_tmp, literal, default_struct, dict_key=None
):

    dict_key = f"{literal}s" if dict_key is None else dict_key
    params = {dict_key: []}
    for key, val in params_tmp.items():

        name, indices = parse_name_and_indices(key, literal)
        idx = indices[0]

        if not isinstance(idx, int):
            params[key] = val
        else:
            if len(params[dict_key]) < idx + 1:
                params[dict_key].append(default_struct(0, 0, 0))
            setattr(params[dict_key][idx], name, val)
    return params
