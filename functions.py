import numpy as np
from scipy.special import erfinv, erf
from scipy.stats import norm


def halfnorm_ppf(x, loc, scale): 
    return loc + scale * np.sqrt(2) * erfinv(x)

def norm_ppf(x, mean, sigma): 
    return mean + sigma * np.sqrt(2) * erfinv(2 * x - 1)

def norm_cdf(x, mean, sigma):
    return 0.5 * (1 + erf((x - mean) / (sigma * np.sqrt(2))))


# def _phi(x):
#     return 1./np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

# def _psi(x):
#     """ Helper function for truncated normal ppf calculation """
#     return 0.5 * (1 + erf(x / np.sqrt(2)))

# def truncated_normal_pdf(x, mean, sigma, low, high):
#     """ Percent point function (inverse of cdf) for truncated normal distribution """
#     p_out = np.zeros_like(x)
    
#     mask_low = x < low
#     mask_high = x > high

#     p_out[mask_low] = 0
#     p_out[mask_high] = 0
#     if np.all(mask_low | mask_high):
#         return p_out

#     psi_low, psi_high = _psi(((low - mean) / sigma, (high - mean) / sigma))

#     mask_in = ~(mask_low | mask_high)
#     p_out[mask_in] = 1. / sigma * _phi((x[mask_in]-mean)/sigma) / (psi_high - psi_low)
#     return p_out
#     # alpha, beta = norm.cdf(a), norm.cdf(b)
#     # scaled_x = alpha + x * (beta - alpha)
#     # return norm.ppf(scaled_x) * sigma + mean

# def truncated_normal_ppf(x, mean, sigma, low, high):
#     """ Percent point function (inverse of cdf) for truncated normal distribution """
    
#     psi_low, psi_high = _psi(((low - mean) / sigma, (high - mean) / sigma))

#     scaled_x = psi_low + x * (psi_high - psi_low)
#     return mean + sigma * np.sqrt(2) * erfinv(2 * scaled_x - 1)

def bounded_flat(x, low, high):
    return x * (high - low) + low



def _circfuncs_common(samples, high, low):
    # xp = array_namespace(samples) if xp is None else xp

    # if xp.isdtype(samples.dtype, "integral"):
    #     dtype = xp.asarray(1.0).dtype  # get default float type
    #     samples = xp.asarray(samples, dtype=dtype)

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low) * 2.0 * np.pi / (high - low))
    cos_samp = np.cos((samples - low) * 2.0 * np.pi / (high - low))

    return samples, sin_samp, cos_samp


def circmean(
    samples, weights=1.0, high=2 * np.pi, low=0, axis=None, nan_policy="propagate"
):
    # xp = array_namespace(samples)
    # Needed for non-NumPy arrays to get appropriate NaN result
    # Apparently atan2(0, 0) is 0, even though it is mathematically undefined
    # if (samples. == 0:
    #     return xp.mean(samples, axis=axis)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = np.sum(sin_samp * weights, axis=axis)
    cos_sum = np.sum(cos_samp * weights, axis=axis)
    res = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

    res = res[()] if res.ndim == 0 else res
    return res * (high - low) / 2.0 / np.pi + low


def modulo_with_offset(x,low=0,high=None):
    if high is None:
        return x
    diff = high - low
    return np.mod(x - low, diff) + low
