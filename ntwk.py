from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix
import os

from aux import save


# CONNECTIVITY FUNCTIONS

def cxns_pcs_rcr(pfs, z_pc, l_pc):
    """
    Generate a recurrent connectivity matrix with preferential
    attachment between pyramidal cells with nearby place fields.
    
    :param pfs: (2 x N) array of place fields (cells without place fields
        should have nans in their place)
    :param z_pc: normalization factor for connections
    :param l_pc: length scale of preferential attachment (m)
    :return: N x N boolean cxn matrix
    """
    # check args
    if len(pfs) != 2:
        raise ValueError('Arg "pfs" must have two rows.')
    if l_pc <= 0:
        raise ValueError('Arg "l_pc" must be > 0.')
    
    # get number of cells
    n = pfs.shape[1]
    
    # build distance matrix
    dx = np.tile(pfs[0][None, :], (n, 1)) - np.tile(pfs[0][:, None], (1, n))
    dy = np.tile(pfs[1][None, :], (n, 1)) - np.tile(pfs[1][:, None], (1, n))
    d = np.sqrt(dx**2 + dy**2)
    
    # build cxn probability matrix
    p = z_pc*np.exp(-d/l_pc)
    
    # set nans and diagonal to zero
    p[np.eye(n, dtype=bool)] = 0
    p[np.isnan(p)] = 0
    
    # build cxn matrix
    cxns = np.random.rand(n, n) < p
    
    return cxns


def ridge_h(shape, dens, hx):
    """
    Randomly sample PCs from along a horizontal "ridge", assigning them each a place-field
    center and an EC->PC cxn weight.
    
    :param shape: tuple specifying ridge shape (m)
    :param dens: number of place fields per m^2
    :param hx: dict with keys:
        'dists': array of uniformly spaced distances used to sample weights
        'weights': distribution of EC->PC NMDA cxn weights to sample from as fn of dists
    
    :return: place field centers, EC->PC cxn weights
    """
    # sample number of nrns
    n_pcs = np.random.poisson(shape[0] * shape[1] * dens)
    
    # sample place field positions
    pfcs = np.random.uniform((-shape[0]/2, -shape[1]/2), (shape[0]/2, shape[1]/2), (n_pcs, 2)).T
    
    # sample EC->PC cxn weights according to dists to centerline
    dd = np.mean(np.diff(hx['dists']))
    dist_idxs = np.round(np.abs(pfcs[1]) / dd).astype(int)
    
    ws = []
    for dist_idx in dist_idxs:
        w_dstr = hx['ws_n_ec_pc'][:, min(dist_idx, len(hx['dists'])-1)]
        ws.append(np.random.choice(w_dstr))
    
    return pfcs, ws


# INITIAL CONDITIONS

def sample_vs_gs_init(ws_n_pc_ec, v_g_init):
    """
    Return an initial membrane voltage and NMDA conductance for each
    of several PCs, depending on their EC->PC cxn weight.
    """
    
    dw = np.mean(np.diff(v_g_init['weights']))
    w_idxs = np.round((ws_n_pc_ec - v_g_init['weights'][0])/ dw).astype(int)
    
    vs = np.nan * np.zeros(len(ws_n_pc_ec))
    gs = {'NMDA': np.nan * np.zeros(len(ws_n_pc_ec))}
    
    for nrn_ctr, w_idx in enumerate(w_idxs):
        vs[nrn_ctr] = np.random.choice(v_g_init['vs'][:, w_idx])
        gs['NMDA'][nrn_ctr] = np.random.choice(v_g_init['gs'][:, w_idx])
        
    return vs, gs


# NETWORK CLASSES AND FUNCTIONS

class LIFNtwk(object):
    """
    Network of leaky integrate-and-fire (LIF) neurons. All parameters should be given in SI units
    (i.e., time constants in seconds, potentials in volts). This simulation uses exponential
    synapses for all synapse types.

    In all weight matrices, rows index target, cols index source.
    
    :param t_m: membrane time constant (or 1D array)
    :param e_l: leak reversal potential (or 1D array)
    :param v_th: firing threshold potential (or 1D array)
    :param v_reset: reset potential (or 1D array)
    :param e_ahp: afterhyperpolarization (potassium) reversal potential
    :param t_ahp: afterhyperpolarization time constant
    :param w_ahp: afterhyperpolarization magnitude 
    :param t_r: refractory time
    :param es_syn: synaptic reversal potentials (dict with keys naming
        synapse types, e.g., 'AMPA', 'NMDA', ...)
    :param ts_syn: synaptic time constants (dict)
    :param ws_rcr: recurrent synaptic weight matrices (dict with keys
        naming synapse types)
    :param ws_up: input synaptic weight matrices from upstream inputs (dict)
    :param plasticity: dict of plasticity params with the following keys:
        'masks': synaptic dict of boolean arrays indicating which synapses in 
            ws_up are plastic, i.e., which synapses correspond to ECII-CA3 cxns
        'w_ec_ca3_maxs': synaptic dict of max values for plastic weights
        'T_W': timescale of activity-dependent plasticity
        'T_C': timescale of CA3 spike-counter auxiliary variable
        'C_S': threshold for spike-count-based plasticity activation
        'BETA_C': slope of spike-count nonlinearity
    :param sparse: whether to convert weight matrices to sparse matrices for
        more efficient processing
    """
    
    def __init__(self,
            t_m, e_l, v_th, v_reset, t_r,
            e_ahp=0, t_ahp=np.inf, w_ahp=0,
            es_syn=None, ts_syn=None, ws_up=None, ws_rcr=None, 
            plasticity=None, sparse=True):
        """Constructor."""

        # validate arguments
        if es_syn is None:
            es_syn = {}
        if ts_syn is None:
            ts_syn = {}
        if ws_up is None:
            ws_up = {}
        if ws_rcr is None:
            ws_rcr = {}

        # check syn. dicts have same keys
        if not set(es_syn) == set(ts_syn) == set(ws_rcr) == set(ws_up):
            raise ValueError(
                'All synaptic dicts ("es_syn", "ts_syn", '
                '"ws_rcr", "ws_inp") must have same keys.'
            )

        self.syns = es_syn.keys()

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

        # check plasticity parameters
        if plasticity is not None:
            # make sure all parameters are given
            if set(plasticity) != {'masks', 'w_ec_ca3_maxs', 'T_W', 'T_C', 'C_S', 'BETA_C'}:
                raise KeyError(
                    'Argument "plasticity" must contain the correct keys '
                    '(see LIFNtwk documentation).')
            # make sure there is one plasticity matrix for each synapse type
            if set(plasticity['masks']) != set(ws_up):
                raise KeyError(
                    'Argument "plasticity[\'masks\']" must contain same keys '
                    '(synapse types) as argument "ws_up".')
            # make sure plasticity matrices are boolean and same size as ws_up
            for w in plasticity['masks'].values():
                if w.shape != (self.n, self.n_up):
                    raise ValueError(
                        'All matrices in "plasticity[\'masks\']" must have same '
                        'shape as matrices in "ws_up".')
                if w.dtype != bool:
                    raise TypeError(
                        'All matrices in "plasticity[\'masks\']" must be '
                        'logical arrays.')
            # make sure max weight values dict has correct synaptic keys
            if set(plasticity['w_ec_ca3_maxs']) != set(ws_up):
                raise KeyError(
                    'Argument "plasticity[\'w_ec_ca3_maxs\']" must contain same '
                    'keys (synapse types) as argument "ws_up".')
        
        # make sure v_reset is actually an array
        if isinstance(v_reset, (int, float, complex)):
            v_reset = v_reset * np.ones(self.n)
            
        # store network params
        self.t_m = t_m
        self.e_l = e_l
        self.v_th = v_th
        self.v_reset = v_reset
        self.t_r = t_r
        self.e_ahp = e_ahp
        self.t_ahp = t_ahp
        self.w_ahp = w_ahp
        
        self.es_syn = es_syn
        self.ts_syn = ts_syn
        
        self.plasticity = plasticity
        if plasticity is not None:
            self.ns_plastic = {syn: w.sum() for syn, w in plasticity['masks'].items()}
         
        if sparse:
            ws_rcr = {syn: csc_matrix(w) for syn, w in ws_rcr.items()}
            ws_up = {syn: csc_matrix(w) for syn, w in ws_up.items()}
            
        self.ws_rcr = ws_rcr
        self.ws_up_init = ws_up

    def run(self, spks_up, dt, vs_init=None, gs_init=None, g_ahp_init=None):
        """
        Run a simulation of the network.

        :param spks_up: upstream spiking inputs (rows are time points, cols are neurons)
            (should be non-negative integers)
        :param dt: integration time step for dynamics simulation
        :param vs_init: initial vs
        :param gs_init: initial gs (dict of 1-D arrays)
        :param g_ahp_init: initial g_ahp (1-D array)

        :return: network response object
        """

        # validate arguments
        if vs_init is None:
            vs_init = self.e_l * np.ones(self.n)
        if gs_init is None:
            gs_init = {syn: np.zeros(self.n) for syn in self.syns}
        if g_ahp_init is None:
            g_ahp_init = np.zeros(self.n)
        
        if type(spks_up) != np.ndarray or spks_up.ndim != 2:
            raise TypeError('"inps_upstream" must be a 2D array.')

        if not spks_up.shape[1] == self.n_up:
            raise ValueError('Upstream input size does not match size of input weight matrix.')

        if not vs_init.shape == (self.n,):
            raise ValueError('"vs_init" must be 1-D array with one element per neuron.')

        if not all([gs.shape == (self.n,) for gs in gs_init.values()]):
            raise ValueError(
                'All elements of "gs_init" should be 1-D array with one element per neuron.')
        if not g_ahp_init.shape == (self.n,):
            raise ValueError(
                '"g_ahp_init" should be 1-D array with one element per neuron.')

        ts = np.arange(len(spks_up)) * dt

        # allocate space for dynamics results
        sim_shape = (len(ts), self.n)
        vs = np.nan * np.zeros(sim_shape)
        spks = np.zeros(sim_shape, dtype=bool)
        gs = {syn: np.nan * np.zeros(sim_shape) for syn in self.syns}
        g_ahp = np.nan * np.zeros(sim_shape)
        
        # initialize membrane potentials, conductances, and refractory counters
        vs[0, :] = vs_init
        
        for syn in self.syns:
            gs[syn][0, :] = gs_init[syn]
        g_ahp[0, :] = g_ahp_init
        
        rp_ctrs = np.zeros(self.n)
        
        # initialize plasticity variables
        if self.plasticity is not None:
            
            # rename variables to make them more accessible
            masks_plastic = self.plasticity['masks']
            w_ec_ca3_maxs = self.plasticity['w_ec_ca3_maxs']
            t_w = self.plasticity['T_W']
            t_c = self.plasticity['T_C']
            c_s = self.plasticity['C_S']
            beta_c = self.plasticity['BETA_C']
            
            # allocate space for plasticity variables
            # NOTE: ws_plastic values are time-series of just the plastic weights
            # in a 2D array where rows are time points and cols are weights
            cs = np.zeros(sim_shape)
            ws_plastic = {
                syn: np.nan * np.zeros((len(ts), n_plastic))
                for syn, n_plastic in self.ns_plastic.items()
            }
        
            # set spike counter to 0 and plastic weights to initial weights
            cs[0] = 0
            for syn, mask in masks_plastic.items():
                ws_plastic[syn][0] = self.ws_up_init[syn][mask].copy()
        else:
            masks_plastic = None
            ws_plastic = None
            cs = None
        
        # run simulation
        ws_up = deepcopy(self.ws_up_init)
        
        for step in range(1, len(ts)):

            ## update dynamics
            for syn in self.syns:
                
                # calculate new conductances for all synapse types
                w_up = ws_up[syn]
                w_rcr = self.ws_rcr[syn]
                t_syn = self.ts_syn[syn]

                # calculate upstream and recurrent inputs to conductances
                inps_up = w_up.dot(spks_up[step])
                inps_rcr = w_rcr.dot(spks[step-1].astype(float))

                # decay conductances and add any positive inputs
                dg = -(dt/t_syn) * gs[syn][step-1] + inps_up + inps_rcr
                # store conductances
                gs[syn][step] = gs[syn][step-1] + dg
                
            # calculate new AHP conductance
            inps_ahp = self.w_ahp * spks[step-1]
            
            # decay ahp conductance and add new inputs
            dg_ahp = (-dt/self.t_ahp) * g_ahp[step-1] + inps_ahp
            # store ahp conductance
            g_ahp[step] = g_ahp[step-1] + dg_ahp

            # calculate current input resulting from synaptic conductances
            is_g = [gs[syn][step] * (self.es_syn[syn] - vs[step-1]) for syn in self.syns]
            # add in AHP current
            is_g.append(g_ahp[step] * (self.e_ahp - vs[step-1]))

            # update membrane potential
            dvs = -(dt/self.t_m) * (vs[step-1] - self.e_l) + np.sum(is_g, axis=0)
            vs[step] = vs[step-1] + dvs

            # force refractory neurons to reset potential
            vs[step][rp_ctrs > 0] = self.v_reset[rp_ctrs > 0]

            # identify spks
            spks[step] = vs[step] >= self.v_th
            # reset membrane potentials of spiking neurons
            vs[step][spks[step]] = self.v_reset[spks[step]]

            # set refractory counters for spiking neurons
            rp_ctrs[spks[step]] = self.t_r
            # decrement refractory counters for all neurons
            rp_ctrs -= dt
            # adjust negative refractory counters up to zero
            rp_ctrs[rp_ctrs < 0] = 0
            
            ## update plastic weights
            if self.plasticity is not None:
                
                # calculate and store updated spk-ctr
                cs_next = update_spk_ctr(spks=spks[step], cs_prev=cs[step-1], t_c=t_c, dt=dt)
                cs[step] = cs_next
                
                # calculate new weight values for each syn type
                for syn in self.syns:
                    ws_prev = ws_plastic[syn][step-1]
                    w_ec_ca3_max = w_ec_ca3_maxs[syn]
                    
                    # reshape spk-ctr variable to align with updated weights
                    cs_syn = cs_next[masks_plastic[syn].nonzero()[0]]
                    ws_next = update_plastic_weights(
                        cs=cs_syn, ws_prev=ws_prev, c_s=c_s, beta_c=beta_c,
                        t_w=t_w, w_ec_ca3_max=w_ec_ca3_max, dt=dt)
               
                    # store updated weight values
                    ws_plastic[syn][step] = ws_next

                # insert updated weights into ws_up
                for syn, mask in masks_plastic.items():
                    ws_up[syn][mask] = ws_plastic[syn][step]

        # return NtwkResponse object
        return NtwkResponse(
            vs=vs, spks=spks, v_rest=self.e_l, v_th=self.v_th,
            gs=gs, g_ahp=g_ahp, ws_rcr=self.ws_rcr, ws_up=self.ws_up_init,
            cs=cs, ws_plastic=ws_plastic, masks_plastic=masks_plastic)


def z(c, c_s, beta_c):
    return 1 / (1 + np.exp(-(c - c_s)/beta_c))


def update_spk_ctr(spks, cs_prev, t_c, dt):
    """
    Update the spk-ctr auxiliary variable.
    :param spks: multi-unit spk vector from current time step
    :param cs_prev: spk-ctrs for all cells at previous time step
    :param t_c: spk-ctr time constant (see parameters.ipynb)
    :param dt: numerical integration time step
    """
    dc = -cs_prev * dt / t_c + spks.astype(float)

    return cs_prev + dc


def update_plastic_weights(cs, ws_prev, c_s, beta_c, t_w, w_ec_ca3_max, dt):
    """
    Update the plastic cxns from EC to CA3.
    
    :param cs: spk-ctrs for all cells at current time step
    :param ws_prev: 1-D array of plastic weight values at previous timestep
    :param c_s: spk-ctr threshold (see parameters.ipynb)
    :param beta_c: spk-ctr nonlinearity slope (see parameters.ipynb)
    :param t_w: weight change timescale (see parameters.ipynb)
    :param w_ec_ca3_max: syn-dict of maximum EC->CA3 weight values
    :param dt: numerical integration time step
    """
    if cs.shape != ws_prev.shape:
        raise ValueError('Spk-ctr "cs" and plastic weights "ws_prev" must have same shape.')
        
    dw = z(cs, c_s, beta_c) * (w_ec_ca3_max - ws_prev) * dt / t_w 
    return ws_prev + dw


class NtwkResponse(object):
    """
    Class for storing network response parameters.

    :param vs: membrane potentials
    :param spks: spk times
    :param gs: syn-dict of conductances
    :param ws_rcr: syn-dict of recurrent weight matrices
    :param ws_up: syn-dict upstream weight matrices
    :param cell_types: array-like of cell-types
    :param cs: spk ctr variables for each cell
    :param ws_plastic: syn-dict of time-courses of plastic weights
    :param masks_plastic: syn-dict of masks specifying which weights the plastic
        ones correspond to
    :param place_field_centers: array of cell place field centers
    """

    def __init__(
            self, vs, spks, v_rest, v_th, gs, g_ahp, ws_rcr, ws_up, cell_types=None,
            cs=None, ws_plastic=None, masks_plastic=None, place_field_centers=None):
        """Constructor."""
        # check args
        if (cell_types is not None) and (len(cell_types) != vs.shape[1]):
            raise ValueError(
                'If "cell_types" is provided, all cells must have a type.')
            
        self.vs = vs
        self.spks = spks
        self.v_rest = v_rest
        self.v_th = v_th
        self.gs = gs
        self.g_ahp = g_ahp
        self.ws_rcr = ws_rcr
        self.ws_up = ws_up
        self.cell_types = cell_types
        self.cs = cs
        self.ws_plastic = ws_plastic
        self.masks_plastic = masks_plastic
        self.place_field_centers = place_field_centers

    def save(self, save_file, save_gs=False, save_ws=True, save_place_fields=True):
        """
        Save network response to file.

        :param save_file: path of file to save it to (do not include .db extension)
        :param save_gs: whether to save conductances
        :param save_ws: whether to save connectivity matrices
        :param save_positions: whether to save positions
        """
        data = {
            'vs': self.vs,
            'spks': self.spks,
            'v_rest': self.v_rest,
            'v_th': self.v_th,
            'cell_types': self.cell_types,
        }

        if save_gs:
            data['gs'] = self.gs
            data['g_ahp'] = self.g_ahp

        if save_ws:
            data['ws_rcr'] = self.ws_rcr
            data['ws_up'] = self.ws_up
            data['ws_plastic'] = self.ws_plastic
            data['masks_plastic'] = self.masks_plastic

        if save_place_fields:
            data['place_field_centers'] = self.place_field_centers

        return save(save_file, data)
