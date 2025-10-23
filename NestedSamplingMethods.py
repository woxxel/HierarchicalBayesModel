import logging
from matplotlib import pyplot as plt
import numpy as np
from .functions import circmean as weighted_circmean, modulo_with_offset
from scipy.ndimage import gaussian_filter1d as gauss_filter
from scipy.interpolate import interp1d


def run_sampling(
    prior_transform,
    loglikelihood,
    parameter_names,
    periodic=None,
    reflective=None,
    mode="dynesty",
    n_live=100,
    nP=1,
    logLevel=logging.ERROR,
):
    n_params = len(parameter_names)

    if mode=='dynesty':

        from dynesty import NestedSampler, pool as dypool

        # print("running nested sampling")

        options = {
            "ndim": n_params,
            "nlive": n_live,
            "bound": "single",
            "sample": "rslice",
            # reflective=[BM.priors["p"]["idx"]] if two_pop else False,
            "periodic": np.where(periodic)[0].tolist() if (periodic and np.any(periodic)) else False,
        }
        # print(f"nP={nP}")
        if nP>1:
            with dypool.Pool(nP, loglikelihood, prior_transform) as pool:
                sampler = NestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    pool=pool,
                    **options,
                )
                sampler.run_nested(dlogz=1.,print_progress=False)
        else:
            sampler = NestedSampler(
                loglikelihood,
                prior_transform,
                **options,
            )
            sampler.run_nested(dlogz=1.,print_progress=False)
        sampling_result = sampler.results

        return sampling_result, sampler
    else:

        import ultranest
        from ultranest.stepsampler import RegionSliceSampler

        from ultranest.popstepsampler import (
            PopulationSliceSampler,
            generate_region_oriented_direction,
        )
        from ultranest.mlfriends import RobustEllipsoidRegion

        NS_parameters = {
            "min_num_live_points": n_live,
            "max_num_improvement_loops": 2,
            "max_iters": 50000,
            "cluster_num_live_points": 20,
        }

        logger = logging.getLogger("ultranest")
        logger.setLevel(logLevel)

        sampling_result = None
        show_status = True
        n_steps = 10

        sampler = ultranest.ReactiveNestedSampler(
            parameter_names,
            loglikelihood,
            prior_transform,
            # wrapped_params=BM.wrap,
            vectorized=True,
            num_bootstraps=20,
            ndraw_min=512,
        )

        while True:
            try:
                # sampler.stepsampler = PopulationSliceSampler(
                sampler.stepsampler = RegionSliceSampler(
                    # popsize=2**4,
                    nsteps=n_steps,
                    # generate_direction=generate_region_oriented_direction,
                )

                sampling_result = sampler.run(
                    **NS_parameters,
                    region_class=RobustEllipsoidRegion,
                    update_interval_volume_fraction=0.01,
                    show_status=show_status,
                    viz_callback=False,#"auto",
                )
            except Exception as exc:
                if type(exc) == KeyboardInterrupt:
                    break
                if type(exc) == TimeoutException:
                    raise TimeoutException("Sampling took too long")
                n_steps *= 2
                print(f"increasing step size to {n_steps=}")
                if n_steps > 100:
                    break
        return sampling_result, sampler


def get_samples_from_results(results, mode="dynesty"):

    return {
        "samples": (
            results.samples
            if mode == "dynesty"
            else results["weighted_samples"]["points"]
        ),
        "weights": (
            results.importance_weights()
            if mode == "dynesty"
            else results["weighted_samples"]["weights"]
        ),
    }


def get_single_posterior_from_samples(
    samp,
    weights,
    periodic=False,
    x=None,
    qs=[0.05, 0.341, 0.5, 0.841, 0.95],
    smooth_sigma=1.0,
):  # , parameter_names, mode="dynesty", output="dict"):

    qs = [0.001, 0.05, 0.341, 0.5, 0.841, 0.95, 0.999]

    samp = np.array(samp).copy()

    posterior = {}
    if isinstance(periodic, list):
        # print("this is periodic: ", periodic)
        low, high = periodic
        diff = high - low

        posterior["mean"] = weighted_circmean(samp, weights=weights, low=low, high=high)
        shift_from_center = (
            posterior["mean"] - low - diff / 2.0
        )  # shift field to the center

        samp[
            samp < np.mod(shift_from_center - low, diff)
        ] += diff  # move lower part above the wrap point to allow proper posterior
    else:
        low, high = None, None
        posterior["mean"] = (samp * weights).sum()

    idx_sorted = np.argsort(samp)
    samples_sorted = samp[idx_sorted]

    # get corresponding weights
    sw = weights[idx_sorted]

    cumsw = np.cumsum(sw)
    quants = np.interp(qs, cumsw, samples_sorted)

    posterior["CI"] = modulo_with_offset(quants[1:-1], low, high)
    posterior["stdev"] = np.sqrt((weights * (samp - posterior["mean"]) ** 2).sum())

    if x is not None:
        f = interp1d(
            modulo_with_offset(samples_sorted, low, high),
            cumsw,
            bounds_error=False,
            fill_value="extrapolate",
        )

        # cdf makes jump at wrap-point, resulting in a single negative value. "Maximum" fixes this, but is not ideal
        posterior["p_x"] = np.maximum(
            0,
            (
                gauss_filter(f(x[1:]) - f(x[:-1]), smooth_sigma)
                if smooth_sigma > 0
                else f(x[1:]) - f(x[:-1])
            ),
        )

    return posterior


def get_posterior_from_samples(
    samp, weights, parameter_names=None, periodic=None, **kwargs
):

    iterator = range(samp.shape[1]) if parameter_names is None else parameter_names

    if periodic is None:
        periodic = [False] * samp.shape[1]

    posterior = {}
    for i, key in enumerate(iterator):

        # print(f"computing mean for parameter {key}, periodic={periodic[i]}")
        posterior[key] = get_single_posterior_from_samples(
            samp[:, i], weights, periodic=periodic[i], **kwargs
        )

    return posterior


def plot_results(BM,results,mode="dynesty",truths=None):

    # priors = {key:prior for key,prior in BM.priors.items() if prior["transform"] is not None}

    samples = get_samples_from_results(results, mode=mode)
    posterior = get_posterior_from_samples(
        samples["samples"],
        samples["weights"],
        BM.parameter_names_all,
        BM.periodic_boundaries,
        qs=[0.341, 0.5, 0.841, 0.999],
    )

    samples_dict = {
        key: samples["samples"][:, i] for i, key in enumerate(BM.parameter_names_all)
    }

    plot_params = [key for key in BM.parameter_names if BM.priors[key]["transform"] is not None]
    fig,axes = plt.subplots(nrows=len(plot_params), ncols=1, figsize=(10, 1.5*len(plot_params)),sharex=True)
    for i,key in enumerate(plot_params):
        prior = BM.priors[key]
        if not prior["transform"]:
            continue

        title_str = prior["label"] if prior["label"] is not None else key
        if prior["n"] == 1:
            # print("Single population model")
            plot_prior(axes[i], prior, color="tab:green", label="prior")
            plot_posterior(
                axes[i],
                samples_dict[key],
                samples["weights"],
                posterior[key],
                alpha=0.5,
            )
            title_str += f" [${posterior[key]['mean']:.3f}\pm{posterior[key]['stdev']:.3f}$]"

            if truths and key in truths:
                axes[i].axhline(truths[key], color="tab:red", linestyle="--", label="truth")
        else:
            # print("Multi-population model")

            y_max = 0.

            colors = ["tab:green", "tab:blue"]
            if prior["has_meta"]:
                for var,col in zip(prior["input_vars"],colors):
                    key_hierarchy = f"{key}_{var}"
                    # print(key_hierarchy, BM.priors[key_hierarchy])
                    plot_prior(axes[i], BM.priors[key_hierarchy], color=col, label="prior")
                    plot_posterior(
                        axes[i],
                        samples_dict[key_hierarchy],
                        samples["weights"],
                        posterior[key_hierarchy],
                        color=col,
                    )
                    axes[i].axhline(posterior[key_hierarchy]["mean"], color=col, linestyle="--", label="posterior (meta)")

                    if var=="mean":
                        y_max = max(y_max, posterior[key_hierarchy]["CI"][-1])

                        title_str += f" [${posterior[key_hierarchy]['mean']:.3f}\pm{posterior[key_hierarchy]['stdev']:.3f}$]"

                if truths and key in truths:
                    axes[i].axhline(truths[key], color="tab:red", linestyle="--", label="truth")
            else:
                plot_prior(axes[i], prior, color="tab:green", label="prior")

            for n in range(prior["n"]):
                bottom = 1.+n/2
                plot_posterior(
                    axes[i],
                    samples_dict[f"{key}_{n}"],
                    samples["weights"],
                    posterior[f"{key}_{n}"],
                    bottom=bottom,
                    color="grey",
                    alpha=0.5,
                    label="posterior (pop)" if n == 0 else None,
                )

                y_max = max(y_max, posterior[f"{key}_{n}"]["CI"][-1])

                if not prior["has_meta"]:
                    axes[i].text(bottom,posterior[f"{key}_{n}"]["mean"],f"${posterior[f'{key}_{n}']['mean']:.3f}\pm{posterior[f'{key}_{n}']['stdev']:.3f}$",va="bottom")

                    if truths and (key in truths):
                        if isinstance(truths[key], (list, np.ndarray)):
                            truth = truths[key][n] if n<len(truths[key]) else truths[key][0]
                        else:
                            truth = truths[key]

                        axes[i].plot([bottom,bottom+1./2],[truth]*2,color="tab:red", linestyle="--", label="truth")

            plt.setp(axes[i], ylim=(0, y_max))

        axes[i].set_title(title_str)
        axes[i].spines[["top","right"]].set_visible(False)
        if i==0:
            axes[i].legend()
    plt.setp(axes[-1],
            xticks=(0,)+tuple(np.linspace(1,1+BM.dimensions["shape"][0]/2-0.5,BM.dimensions["shape"][0])),
            xticklabels=("meta",) + tuple(f"n={n+1}" for n in range(BM.dimensions["shape"][0]))
        )

    # plt.setp(axes[0],xlim=(0,4))
    plt.tight_layout()


from scipy.ndimage import gaussian_filter


def plot_posterior(ax, samples, weights, posterior, **kwargs):

    offset = kwargs.get("bottom", 0.0)

    # print(samples_sorted)
    # x = posterior["samples"]
    # weights = posterior["weights"]
    try:
        span = np.percentile(
            samples, [0.1, 99.9], weights=weights, method="inverted_cdf"
        )
    except:
        span = np.percentile(samples, [0.1, 99.9])
    span[0] = 0
    sx = 0.02
    bins = int(round(10. / sx))

    n, b = np.histogram(samples, bins=bins, weights=weights, range=np.sort(span))

    n = gaussian_filter(n, 10.)
    n /= n.max()*2 * 1.1

    ax.plot(offset + n,b[:-1],color=kwargs.get("color","tab:grey"),label=kwargs.get("label"))
    ax.fill_betweenx(b[:-1], offset, offset + n, color=kwargs.get("color", "tab:grey"), alpha=kwargs.get("alpha", 0.7))

    ax.plot(offset,posterior["mean"],"o",color=kwargs.get("color","tab:grey"),markersize=5)


def plot_prior(ax, prior, **kwargs):

    one_range = np.linspace(0, 1, 1001)
    y = prior["transform"](one_range)
    x = np.gradient(one_range, y)
    x /= np.nanmax(x)

    color = kwargs.get("color", "tab:red")
    ax.plot(x, y, **(kwargs | {"color": color}))
    ax.fill_betweenx(y, 0, x, color=color, alpha=kwargs.get("alpha", 0.2))


# def compare_results(BM, sampler, mP, mode="ultranest", biological=False):

#     print('data in:')
#     for key in mP.params.keys():
#         print(f'{key} = {mP.params[key]}')

#     paraKeys = []

#     if biological:
#         paraKeys.extend(["nu_bar", "alpha_0", "tau_A", "tau_N", "r_N"])
#         truth_values = None
#     else:
#         if mP.two_pop:
#             paraKeys.extend("p")

#         for i in range(2 if mP.two_pop else 1):
#             paraKeys.extend([f"gamma_{i}", f"delta_{i}", f"nu_max_{i}"])
#         # paraKeys.extend(["gamma", "delta_1", "nu_max_1"])
#         # if mP.two_pop:
#         #     paraKeys.extend(["gamma_2", "delta_2"])

#         truth_values = []
#         for distributions in mP.params["distr"]:
#             # for key in paraKeys:
#             for key in distributions:
#                 # if key in distributions.keys():
#                 truth_values.append(distributions[key])
#             # break
#             # truth_values.append(mP.params[key])

#     mean = get_mean_from_sampler(sampler.results, paraKeys, mode=mode)

#     if mode=='dynesty':

#         from dynesty import utils as dyfunc, plotting as dyplot

#         dyplot.traceplot(
#             sampler.results,
#             # truths=truth_values,
#             truth_color="black",
#             show_titles=True,
#             trace_cmap="viridis",
#         )
#         plt.show(block=False)

#         dyplot.cornerplot(
#             sampler.results,
#             color="dodgerblue",
#             # truths=truth_values,
#             show_titles=True,
#         )
#         plt.show(block=False)

#         print("\nresults", mean)
#     else:

#         sampler.plot_trace()
#         sampler.plot_corner()

#         print('\nresults')
#         for i,key in enumerate(sampler.results['paramnames']):
#             print(
#                 f"{key} = {sampler.results['posterior']['mean'][i]} \pm {sampler.results['posterior']['stdev'][i]}"
#             )
#         plt.show(block=False)

#     mP.plot_rates(param_in=mP.params)

#     if biological:
#         dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

#         distribution_mean = {}
#         distribution_mean["nu_max"] = get_nu_max(**mean)
#         distribution_mean["gamma"] = get_gamma(**mean)
#         distribution_mean["delta"] = get_delta(**mean)
#         nu_bar_in = get_nu_bar(**mP.params["distr"][0])
#         nu_bar_out = get_nu_bar(**distribution_mean)

#         tau_I_in = get_tau_I(nu_max=mP.params["distr"][0]["nu_max"])
#         tau_I_out = get_tau_I(nu_max=distribution_mean["nu_max"])

#         alpha_0_in = get_alpha_0(**mP.params["distr"][0])
#         alpha_0_out = get_alpha_0(**distribution_mean)

#         print(f"{nu_bar_in=}, {nu_bar_out=}")
#         print(f"{tau_I_in=}, {tau_I_out=}")
#         print(f"{alpha_0_in=}, {alpha_0_out=}")

#     results_inferred = {"distr": [{}] * (2 if mP.two_pop else 1)}
#     for key in mean.keys():
#         print(f"{key} = {mean[key]}")

#         if key == "p":
#             results_inferred[key] = mean[key]
#             continue

#         # if key.startswith("nu"):
#         #     var = key
#         # else:
#         key_split = key.split("_")
#         idx = int(key_split[-1]) if len(key_split) > 1 else np.nan
#         var = ("_").join(key_split[:-1]) if np.isfinite(idx) else key

#         print(var, idx)
#         results_inferred["distr"][idx][var] = mean[key]

#     return results_inferred


class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass
