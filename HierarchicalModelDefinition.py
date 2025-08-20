import numpy as np
import logging, os
from pprint import pprint
import itertools
import warnings

class HierarchicalModel:
    """
        Defines a general class for setting up a hierarchical model for bayesian inference. Has to be inherited by a specific model class, which then further specifies the loglikelihood etc
    """

    def __init__(self, logLevel=logging.ERROR):
        '''
            initialize the class
        '''

        self.log = logging.getLogger("nestLogger")
        self.set_logLevel(logLevel)

        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

        # self.dims["n_samples"] = 0   ## requires to be set by method


    def set_logLevel(self,logLevel):
        self.log.setLevel(logLevel)
    

    def prepare_data(self,event_counts,T,dimension_names=None):
        """
            Preprocesses the data, provided as an n_samples x n_conditions x max(n_neuron) array, containing the spike counts of each neuron
            Dimensionality of event_counts might differ and assumes 1 for each missing dimension

            n_neuron might differ between animals, so the data is usually padded with NaNs

            INPUT:
                * event_counts     [int] n_animals x n_conditions x n_datapoints
                    number of observed events per stimulus during time T
                * T     [int]
                    time period of measurement

            DIMENSIONS:
                n_samples:      number of animals in datapool

                n_conditions:   could be: drive by different input current, e.g. different orientations

                n_datapoints:   here: # neurons; could also be: different stimuli (e.g. places, for place fields)
        """
        
        event_counts = np.array(event_counts)
        self.T = T

        self.data = {
            "event_counts":     event_counts,
            "T":                T,
        }

        dims = self.data["event_counts"].shape
        self.dimensions = {
            "shape":        dims,
            "n":            len(dims),
            "names":        dimension_names if dimension_names else [f"dimension_{i}_x{dim}" for i,dim in enumerate(dims)],
        }

        self.dimensions["iterator"] = list(itertools.product(
            *[range(s) for s in self.dimensions["shape"][:-1]]
        ))

        self.data["n_neurons"] = np.array(
            [
                np.isfinite(events).sum(axis=-1)
                for events in self.data["event_counts"]
            ]
        )
        # pprint(f"{self.data=}")
        # pprint(f"{self.dimensions=}")



    def set_priors(self, priors_init):
        """
            Set the priors for the model. The priors are defined in the priors_init dictionary, which has to follow the following structure:

            All parameters appearing in "hierarchical" will be treated as hierarchical parameters. If a parameter is hierarchical, the mean and **2nd_key** parameter define the meta distribution. If a parameter is not hierarchical, only the mean parameter is used.
        """

        self.paramNames = []
        self.paramIn = list(priors_init.keys())
        self.priors = {}

        self.n_params = 0

        for prior_key, prior in priors_init.items():

            if prior.get("has_meta", False):
                
                if np.all(self.dimensions["shape"][:-1]==1):
                    logging.warning(
                        f"{prior_key} is set as hierarchical, but only one sample is available. \nThis doesn't make much sense. Consider using non-hierarchical priors instead."
                    )

                ## add the parameters for the hierarchical prior
                for sub_key, sub_prior in prior["parameters"].items():
                    if isinstance(sub_prior,dict):
                        self.set_prior_param(
                            sub_prior,
                            prior_key,
                            sub_key,
                        )

            ## then, add the actual parameters for the hierarchical prior
            self.set_prior_param(prior, prior_key, has_meta=prior.get("has_meta", False))

        self.wrap = np.zeros(self.n_params).astype('bool')


    def set_prior_param(
        self, priors_init, param, key=None, has_meta=False
    ):
        """
            sets a single prior variable
        TODO
        * description of the function
        """

        paramName = param + (f"_{key}" if key else "")

        self.priors[paramName] = {}
        self.priors[paramName]["idx"] = self.n_params

        # print(f"pre:  {priors_init['shape']} vs {self.dimensions['shape']}")
        
        ## check for proper shapes of priors and align shape
        shape = priors_init["shape"] + (1,)
        assert len(shape)<=self.dimensions["n"], f"prior for {paramName} should have {self.dimensions['n']-1} dimensions, but {shape} is provided"
        for i,dim in enumerate(self.dimensions["shape"][::-1],start=1):
            if i>len(shape):
                shape = (1,) + shape
            assert shape[-i]==1 or shape[-i]==dim, f"prior shape {shape} is not broadcastable to {self.dimensions['shape']}"
        # print(f"post:  {shape} vs {self.dimensions['shape']}")
        
        self.priors[paramName]["shape"] = shape
        self.priors[paramName]["n"] = np.prod(priors_init["shape"])
        
        self.priors[paramName]["has_meta"] = has_meta

        if priors_init["function"] is None:
            ### None function means that the value is constant and not sampled
            self.priors[paramName]["value"] = np.broadcast_to(
                list(priors_init["parameters"].values())[0], priors_init["shape"]
            )
            self.priors[paramName]["transform"] = None
            return None

        elif has_meta:

            # get indexes of hierarchical parameters for quick access later
            self.priors[paramName]["input_vars"] = []
            self.priors[paramName]["input_constants"] = {}

            for var in priors_init["parameters"].keys():
                if self.priors.get(f"{param}_{var}") is None:
                    self.priors[paramName]["input_constants"][var] = priors_init["parameters"][var]
                else:
                    self.priors[paramName][f"idx_{var}"] = self.priors[f"{param}_{var}"][
                        "idx"
                    ]
                    self.priors[paramName][f"n_{var}"] = self.priors[f"{param}_{var}"]["n"]
                    self.priors[paramName]["input_vars"].append(var)

            self.priors[paramName]["transform"] = lambda x, params, fun=priors_init[
                "function"
            ]: fun(x, **params)

        else:
            self.priors[paramName]["transform"] = (
                lambda x, params=priors_init["parameters"], fun=priors_init[
                    "function"
                ]: fun(x, **params)
            )
        self.n_params += self.priors[paramName]["n"]
        self.paramNames.append(paramName)
        

    def set_prior_transform(self,vectorized=True):
        '''
            sets the prior transform function for the model

            only takes as input the mode, which can be either of
            - 'vectorized': vectorized prior transform function
            - 'scalar': scalar prior transform function
            - 'tensor': tensor prior transform function
        '''

        def prior_transform(p_in):

            """
                The actual prior transform function, which transforms the random variables from the unit hypercube to the actual priors
            """

            # print(p_in.shape)
            # print(self.dimensions)

            if len(p_in.shape)==1:
                p_in = p_in[np.newaxis,...]
            n_chain = p_in.shape[0]
            # print(p_in.shape,p_in)
            p_out = np.zeros_like(p_in)

            for key, prior in self.priors.items():
                # print(key,prior)
                if prior["transform"] is None:
                    continue
                
                input_keys = {}                
                if prior.get("has_meta", False):
                    ## get input variables and constants for input to hierarchical prior
                    input_keys["params"] = {}

                    for var in prior["input_vars"]:
                        # print(f"idx_{var}",prior[f"idx_{var}"], prior[f"n_{var}"])
                        input_keys["params"][var] = p_out[:, prior[f"idx_{var}"]:prior[f"idx_{var}"] + prior[f"n_{var}"]].reshape((n_chain,)+self.priors[f"{key}_{var}"]["shape"][:-1])
                        # print(input_keys["params"][var].shape)
                    for var in prior["input_constants"]:
                        input_keys["params"][var] = prior["input_constants"][var]

                # print("params:",input_keys)
                ## transform the prior parameters
                p_out[:, prior["idx"] : prior["idx"] + prior["n"]] = \
                    prior["transform"](
                        p_in[:, prior["idx"] : prior["idx"] + prior["n"]].reshape((n_chain,)+prior["shape"][:-1]), 
                        **input_keys
                    ).reshape((n_chain,-1))

            if vectorized:
                return p_out
            else:
                return p_out[0,:]

        return prior_transform

    def get_params_from_p(self, p_in, idx_chain=None, idx=None):
        """
        obtains a human readable, structured dictionary of parameters from the input p_in
        """
        if len(p_in.shape) == 1:
            p_in = p_in[np.newaxis, :]
        n_chains = p_in.shape[0]

        # assert (
        #     idx_chain is None or idx_chain < nChains
        # ), "idx_chain must be smaller than the number of chains in p_in"
        # assert (
        #     idx_sample is None or idx_sample < self.dims["n_samples"]
        # ), "idx_sample must be smaller than the number of samples in p_in"

        slice_chain = slice(None) if idx_chain is None else idx_chain

        params = {}
        for var in self.paramIn:
            params[var] = np.zeros(
                ((n_chains,) if idx_chain is None else ()) +
                (self.priors[var]["shape"][:-1] if idx is None else ())
            )

            if self.priors[var].get("transform"):

                ## build appropriate slice
                if idx is None:
                    slice_sample = slice(self.priors[var]["idx"],self.priors[var]["idx"] + self.priors[var]["n"])

                else:

                    offset_effective_nd = tuple(idx_ if self.priors[var]["shape"][i] > 1 else 0 for i, idx_ in enumerate(idx))
                    offset_effective = np.ravel_multi_index(offset_effective_nd, self.priors[var]["shape"][:-1])

                    idx_effective = self.priors[var]["idx"] + offset_effective
                    slice_sample = slice(idx_effective, idx_effective + 1)

                # Get the sliced values from p_in
                sliced = np.squeeze(p_in[slice_chain, slice_sample])
            else:
                sliced = np.squeeze(self.priors[var]["value"])#[slice_chain])

            # Fill params[var] with the sliced values, handling both scalar and array cases
            if params[var].shape == ():
                params[var] = sliced
            else:
                params[var][...] = sliced

        # print(params)

        return params
        # if idx_chain is None and idx_sample is None:
        # if no specific chain or sample is requested, return all
        # else:
        # return np.squeeze(params)