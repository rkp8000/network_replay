import numpy as np


class LIFNtwk(object):
    """
    Network of leaky integrate-and-fire (LIF) neurons. All parameters should be given in SI units
    (i.e., time constants in seconds, potentials in volts). This simulation uses exponential
    synapses for all synapse types.

    In all weight matrices, rows index target, cols index source.
    
    :param tau_m: membrane time constant
    :param e_leak: leak reversal potential
    :param v_th: firing threshold potential
    :param v_reset: reset potential
    :param tau_r: refractory time
    :param es_rev: synaptic reversal potentials (dict with keys naming
        synapse types, e.g., 'AMPA', 'NMDA', ...)
    :param taus_syn: synaptic time constants (dict)
    :param ws_rcr: recurrent synaptic weight matrices (dict with keys
        naming synapse types)
    :param ws_up: input synaptic weight matrices from upstream inputs (dict)
    :param dt: integration time step for dynamics simulation
    """
    
    def __init__(self, tau_m, e_leak, v_th, v_reset, tau_r, es_rev, taus_syn, ws_rcr, ws_up, dt):
        """Constructor."""

        # validate arguments

        # check syn. dicts have same keys
        if not set(es_rev) == set(taus_syn) == set(ws_rcr) == set(ws_up):
            raise ValueError(
                'All synaptic dicts ("es_rev", "taus_syn", '
                '"ws_rcr", "ws_inp") must have same keys.'
            )

        self.syns = es_rev.keys()

        # check weight matrices have correct dims
        self.n = list(ws_rcr.values())[0].shape[1]

        if not all([w.shape[0] == w.shape[1] == self.n for w in ws_rcr.values()]):
            raise ValueError('All recurrent weight matrices must be square.')

        # check input matrices' have correct dims
        self.n_up = list(ws_up.values())[0].shape[1]

        if not all([w.shape[0] == self.n for w in ws_up.values()]):
            raise ValueError('Upstream weight matrices must have one row per neuron.')

        if not all([w.shape[1] == self.n_up for w in ws_up.values()]):
            raise ValueError('All upstream weight matrices must have same number of columns.')

        # store network params
        self.tau_m = tau_m
        self.e_leak = e_leak
        self.v_th = v_th
        self.v_reset = v_reset
        self.tau_r = tau_r
        self.es_rev = es_rev
        self.taus_syn = taus_syn
        self.ws_rcr = ws_rcr
        self.ws_up = ws_up
        self.dt = dt

    def run(self, spks_up, vs_init, gs_init):
        """
        Run a simulation of the network.

        :param spks_up: upstream spiking inputs (rows are time points, cols are neurons)
            (should be non-negative integers)
        :param vs_init: initial vs
        :param gs_init: initial gs (dict of 1-D arrays)

        :return: network response object
        """

        # validate arguments
        if type(spks_up) != np.ndarray or spks_up.ndim != 2:
            raise TypeError('"inps_upstream" must be a 2D array.')

        if not spks_up.shape[1] == self.n_up:
            raise ValueError('Upstream input size does not match size of input weight matrix.')

        if not vs_init.shape == (self.n,):
            raise ValueError('"vs_init" must be 1-D array with one element per neuron.')

        if not all([gs.shape == (self.n,) for gs in gs_init.values()]):
            raise ValueError(
                'All elements of "gs_init" should be 1-D with one element per neuron.')

        ts = np.arange(len(spks_up))

        # allocate space for results
        sim_shape = (len(ts), self.n)
        vs = np.nan * np.zeros(sim_shape)
        spks = np.zeros(sim_shape, dtype=bool)
        gs = {syn: np.nan * np.zeros(sim_shape) for syn in self.syns}

        # initialize membrane potentials, conductances, and refractory counters
        vs[0, :] = vs_init
        for syn in self.syns:
            gs[syn][0, :] = gs_init[syn]
        rp_ctrs = np.zeros(self.n)

        # run simulation
        for step in range(1, len(ts)):

            # loop over all synapse types and calculate new conductances
            for syn in self.syns:

                w_up = self.ws_up[syn]
                w_rcr = self.ws_rcr[syn]
                tau_syn = self.taus_syn[syn]

                # calculate upstream and recurrent inputs to conductances
                inps_up = w_up.dot(spks_up[step])
                inps_rcr = w_rcr.dot(spks[step-1].astype(float))

                # decay conductances and add any positive inputs
                dgs = -(self.dt/tau_syn) * gs[syn][step-1] + inps_up + inps_rcr

                # store conductances
                gs[syn][step] = gs[syn][step-1] + dgs

            # calculate current input resulting from conductances
            is_g = [gs[syn][step]*(self.es_rev[syn]-vs[step-1]) for syn in self.syns]

            # update membrane potential
            dvs = -(self.dt/self.tau_m) * (vs[step-1] - self.e_leak) + np.sum(is_g, axis=0)
            vs[step] = vs[step-1] + dvs

            # force refractory neurons to reset potential
            vs[step][rp_ctrs > 0] = self.v_reset

            # identify spikes
            spks[step] = vs[step] >= self.v_th
            # reset membrane potentials of spiking neurons
            vs[step][spks[step]] = self.v_reset

            # set refractory counters for spiking neurons
            rp_ctrs[spks[step]] = self.tau_r
            # decrement refractory counters for all neurons
            rp_ctrs -= self.dt
            # adjust negative refractory counters up to zero
            rp_ctrs[rp_ctrs < 0] = 0

        # return NtwkResponse object
        return NtwkResponse(ts=ts, vs=vs, spks=spks, gs=gs, ws_rcr=self.ws_rcr, ws_up=self.ws_up)


class NtwkResponse(object):
    """
    Class for storing network response parameters.

    :param ts: time vector
    :param vs: membrane potentials
    :param spks: spk times
    :param gs: conductances
    :param ws_rcr: recurrent weight matrices
    :param ws_up: upstream weight matrices
    """

    def __init__(self, ts, vs, spks, gs, ws_rcr, ws_up):
        """Constructor."""
        self.ts = ts
        self.vs = vs
        self.spks = spks
        self.gs = gs
        self.ws_rcr = ws_rcr
        self.ws_up = ws_up

    def save(self, time_file, ntwk_file, save_vs=True, save_spks=True, save_gs=False, save_ws=True):
        """
        Save network response to file.

        :param save_file: path of file to save it to (do not include .db extension)
        :param save_vs: whether to save membrane potentials
        :param save_sps: whether to save spikes
        :param save_gs: whether to save conductances
        :param save_ws: whether to save connectivity matrices
        """
        pass
