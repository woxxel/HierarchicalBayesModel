import logging
from matplotlib import pyplot as plt
import numpy as np

from dynesty import NestedSampler, pool as dypool
import ultranest
import ultranest.stepsampler

from ultranest.popstepsampler import (
    PopulationSliceSampler,
    generate_region_oriented_direction,
)
from ultranest.mlfriends import RobustEllipsoidRegion


def run_sampling(
    prior_transform,
    loglikelihood,
    parameter_names,
    mode="ultranest",
    n_live=100,
    nP=1,
    logLevel=logging.ERROR,
):

    n_params = len(parameter_names)

    if mode=='dynesty':

        print("running nested sampling")

        options = {
            "ndim": n_params,
            "nlive": n_live,
            "bound": "single",
            "sample": "rslice",
            # reflective=[BM.priors["p"]["idx"]] if two_pop else False,
            # periodic=np.where(BM.wrap)[0],
        }

        if nP>1:
            with dypool.Pool(nP, loglikelihood, prior_transform) as pool:
                sampler = NestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    pool=pool,
                    **options,
                )
                sampler.run_nested(dlogz=1.)
        else:
            sampler = NestedSampler(
                loglikelihood,
                prior_transform,
                **options,
            )
            sampler.run_nested(dlogz=1.)
        sampling_result = sampler.results

        return sampling_result, sampler
    else:

        NS_parameters = {
            "min_num_live_points": n_live,
            "max_num_improvement_loops": 3,
            "max_iters": 50000,
            "cluster_num_live_points": 20,
        }

        sampler = ultranest.ReactiveNestedSampler(
            parameter_names,
            loglikelihood,
            prior_transform,
            # wrapped_params=BM.wrap,
            vectorized=True,
            num_bootstraps=20,
            ndraw_min=512,
        )

        logger = logging.getLogger("ultranest")
        logger.setLevel(logLevel)

        show_status = True
        n_steps = 10
        sampler.stepsampler = PopulationSliceSampler(
            popsize=2**4,
            nsteps=n_steps,
            generate_direction=generate_region_oriented_direction,
        )

        sampling_result = sampler.run(
            **NS_parameters,
            region_class=RobustEllipsoidRegion,
            update_interval_volume_fraction=0.01,
            show_status=show_status,
            viz_callback="auto",
        )

        return sampling_result, sampler


def get_mean_from_sampler(results, paramNames, mode="ultranest", output="dict"):

    mean = {} if output == "dict" else []
    for i, key in enumerate(paramNames):
        if mode == "dynesty":
            samp = results.samples[:, i]
            weights = results.importance_weights()
        else:
            samp = results["weighted_samples"]["points"][:, i]
            weights = results["weighted_samples"]["weights"]

        if output == "dict":
            mean[key] = (samp * weights).sum()
        elif output == "list":
            mean.append((samp * weights).sum())
        # print(f"{key} mean: {mean[key]:.3f}")
    return mean


def get_posterior_statistics(results,parameter_names,mode="dynesty"):

    posterior = {}
    for i, key in enumerate(parameter_names):

        post = {}
        if mode == "dynesty":
            post["samples"] = results.samples[:, i]
            post["weights"] = results.importance_weights()
        else:
            post["samples"] = results["weighted_samples"]["points"][:, i]
            post["weights"] = results["weighted_samples"]["weights"]

        post["mean"] = (post["samples"] * post["weights"]).sum()
        post["stdev"] = np.sqrt((post["weights"] * (post["samples"] - post["mean"])**2).sum())

        posterior[key] = post

    return posterior


def plot_results(BM,results,mode="dynesty",truths=None):

    # priors = {key:prior for key,prior in BM.priors.items() if prior["transform"] is not None}

    max_percentile = 99.9
    
    posterior = get_posterior_statistics(results,BM.parameter_names_all,mode=mode)
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
            plot_posterior(axes[i],posterior[key], alpha=0.5)
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
                    plot_posterior(axes[i],posterior[key_hierarchy], color=col)
                    axes[i].axhline(posterior[key_hierarchy]["mean"], color=col, linestyle="--", label="posterior (meta)")

                    if var=="mean":
                        y_max = max(y_max,np.percentile(posterior[key_hierarchy]["samples"], [max_percentile],method="inverted_cdf",weights=posterior[key_hierarchy]["weights"]))
                            # ylim=np.percentile(posterior[key_hierarchy]["samples"], [0.01,99.99],method="inverted_cdf",weights=posterior[key_hierarchy]["weights"]))
                        title_str += f" [${posterior[key_hierarchy]['mean']:.3f}\pm{posterior[key_hierarchy]['stdev']:.3f}$]"
                
                if truths and key in truths:
                    axes[i].axhline(truths[key], color="tab:red", linestyle="--", label="truth")
            else:
                plot_prior(axes[i], prior, color="tab:green", label="prior")

            for n in range(prior["n"]):
                bottom = 1.+n/2
                plot_posterior(axes[i],posterior[f"{key}_{n}"], bottom=bottom, color="grey", alpha=0.5, label="posterior (pop)" if n==0 else None)

                y_max = max(y_max,np.percentile(posterior[f"{key}_{n}"]["samples"], [max_percentile],method="inverted_cdf",weights=posterior[f"{key}_{n}"]["weights"]))

                if not prior["has_meta"]:
                    axes[i].text(bottom,posterior[f"{key}_{n}"]["mean"],f"${posterior[f'{key}_{n}']['mean']:.3f}\pm{posterior[f'{key}_{n}']['stdev']:.3f}$",va="bottom")

                    if truths and (key in truths):
                        if isinstance(truths[key], (list, np.ndarray)):
                            truth = truths[key][n] if n<len(truths[key]) else truths[key][0]
                        else:
                            truth = truths[key]

                        axes[i].plot([bottom,bottom+1./2],[truth]*2,color="tab:red", linestyle="--", label="truth")

            plt.setp(axes[i],ylim=(0,y_max))
            
            

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

def plot_posterior(ax,posterior,**kwargs):

    offset = kwargs.get("bottom", 0.0)

    # print(samples_sorted)
    x = posterior["samples"]
    weights = posterior["weights"]
    span = np.percentile(posterior["samples"], [0.1, 99.9], weights=weights, method="inverted_cdf")
    span[0] = 0
    sx = 0.02
    bins = int(round(10. / sx))

    n, b = np.histogram(x,
        bins=bins,
        weights=weights,
        range=np.sort(span)
    )
    
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
