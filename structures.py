import numpy as np
import inspect

from scipy.special import erfinv


def prior_structure(function=None, shape=(1,), label=None, **kwargs):
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
    }

def halfnorm_ppf(x, loc, scale): 
    return loc + scale * np.sqrt(2) * erfinv(x)

def norm_ppf(x, mean, sigma): 
    return mean + sigma * np.sqrt(2) * erfinv(2 * x - 1)
