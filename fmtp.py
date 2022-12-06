"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file implements the "fMTP" classes used to simulate fMTP: 
 - "fMTP" is the model class that implements the model. Primarily, it contains 
   a function for Trace Formation given a set of trials (foreperiods) and a 
   function for Trace Expression that yields the amount of preparation on each 
   trial.
 - "FPexp" is the class that describes most experiments (blocks with FPs), and
   has a function to run (simulate) this experiment with an fMTP object, and
   return the preparatory state on each trial.
 - "FPgonogo" extends this class to implement the go/nogo task
 - "FPtransfer" is another extension  to simulate transfer experiments.
 
Authors: Josh Manu Salet & Wouter Kruijne
Email: saletjm@gmail.com
"""

# Import packages
import numpy as np 
import scipy.ndimage
import pandas as pd



class fMTP(object):  
    """
    This class implements our fMTP model.
    """
    
    def __init__(self, tau, tau_constant, k, 
                 tau_min = 0.05, tau_max = 5.0, N = 50, dt = 0.001):
        """
        Initializes the attributes of the class:
            tau_min, tau_max: the time range of the time cell layer
            N: number of time cells
            k: temporal smear
            dt: precision
        """
        
        # Define a set of time nodes
        self.dt = dt
        self.t, self.timecells = self._define_time_shankarhoward(
                                tau_min, tau_max, N, k, self.dt)
        # forgetting curve
        self.tau = float(tau) # rate of forgetting
        self.tau_constant = float(tau_constant) # memory persistence
        # Start forgetting from n-1 
        self.decay = lambda x, tau, tau_constant : np.r_[                       
                0, np.arange(2, x+1)**tau + tau_constant]   
                
        return


    def _t2i(self, t):
        """
        Helper function to turn time t into the closest index (column)
        in the time-cell matrix
        """
        return (np.abs(self.t - t)).argmin()


    @staticmethod
    def _define_time_shankarhoward(taumin, taumax, N, k, dt, tmax = 8.0):
        """
        The following returns the activation pattern of TILT/time cells over 
        an interval [0, tmax]. The spacing of time cells is here linear-spaced.
        The critical equation (get activation pattern at each t given a 
        set of tau-star values) is taken from Shankar & Howard (2012)
        A Scale-Invariant Internal Representation of Time (Equation 3.2),
        doi: 10.1162/NECO_a_00212. But unlike that paper, we assume that t
        and tau_star are positive, emphasizing their preparation rather than
        memory.
        """
        
        # physical time:
        t = np.arange(0, tmax, dt)[np.newaxis, :] + dt
        # space time cells' peaks linearly
        tau_star = np.linspace(taumin, taumax, N)                               
        tau_star = tau_star[:,np.newaxis]        

        # The big T representation evoked by an impulse at t=0 
        # here labeled 'A' for 'activation':
        # this is (equation 3.2) in Shankar & Howard, 2012   
        #-- but with negative -tau and negative -t
        fact = np.math.factorial
        A = ( (1/t) * (k ** (k+1))/fact(k) * (-t/-tau_star)**(k+1) * 
                np.exp(-k*(-t/-tau_star)) )

        return t.squeeze(), A


    def trace_formation(self, trs, catch_dur = 3.25):
        """
        For every FP, lay down a trace; effectively, this establishes a set of 
        Hebbian associations between each time cell and the activation process,
        and associations between each time cell and the inhibition process.
        defined as follows:
            - I = INTEGRAL(T [0 -> FP - 0.050])
            - A = INTEGRAL(T [FP -> FP + 0.3])
        i.e.: activity from 0 -50ms before FP are associated with inhibition,
        and 0-300ms after FP (response times) related to activation.

        trs are assumed to be a list of the FP on each trial; if one is None
        it is assumed to be a catch trial, where only inhibition is built up
        for the duration "catch_dur".
        """
        
        dt = self.dt
        
        # excitation 'zone' runs from FP - FP + 300ms
        zone_a = lambda fp: np.r_[fp, fp + 0.3]
        # inhibition 'zone' runs from start to FP - 50ms
        zone_i = lambda fp: np.r_[0, fp + 0.05]

        # initiallize the weight matrix
        self.W = np.zeros((self.timecells.shape[0], np.array(trs).size, 2))
        # Trace _formation_ is not dependent on the past, only on FP. so we can
        # define traces for every unique FP, and lay those down wherever needed

        catch_done = False
        for tr in np.unique(trs):
            # define z_i, z_a
            z_i, z_a = zone_i(tr), zone_a(tr)

            try:
                # find nearest time-index:
                tidx = self._t2i(tr) # NB: this WON'T fail for nan or bounds
                # make_sure all indices are is in the time-cells matrix
                assert z_a[1] <= self.t.max()
                assert z_i[1] >= self.t.min()
                # make sure the estimate is not more than 1.5 dt off:
                # = more than 1dt
                assert np.abs(self.t[tidx] - tr) < 1.5 * dt

                ## Now, start making traces:
                # inhibition zone is sum up to t-50ms 
                ii = self._t2i(z_i[0]), self._t2i(z_i[1])
                inhib = np.sum(self.timecells[:, ii[0]:ii[1] ], axis = 1) * dt

                # trace for activation: avitvation zone is t to t+300ms
                ia = self._t2i(z_a[0]), self._t2i(z_a[1])
                activ = np.sum(self.timecells[:, ia[0]:ia[1] ], axis = 1) * dt

                # also, find the  trial indices where this trace is formed.
                trial_indices = np.where(trs == tr)[0]

            except AssertionError as e:
                ## either catch trial or boundary error
                if np.isnan(tr): # catch trial; only compute once:
                    if catch_done:
                        continue

                    # catch_inhib: sum of activity up to 'catch_dur'
                    icatch = self._t2i(catch_dur)
                    inhib = np.sum(self.timecells[:, :icatch], axis = 1) * dt
                    # no activation on catch trials (0.0)
                    activ = np.zeros(self.timecells.shape[0])
                    trial_indices  = np.where(np.isnan(trs))[0]
                    catch_done = True
                else: 
                    raise e
            # now, lay down this trace in the association matrix:
            self.W[:, trial_indices, 0] = activ[:, np.newaxis]
            self.W[:, trial_indices, 1] = inhib[:, np.newaxis]
        return


    def trace_formation_gonogo(self, trs):
        """
        Los, 2013, has go / nogo trials with different FPs
        Simulate these as if they were response-trials, (self.trace_formation)
        but subsequently turn the excitation these trials to zero
        """
        gng = [tr[1] for tr in trs]
        FPs = np.array([tr[0] for tr in trs])
        # first, pretend every trial = go:
        self.trace_formation(trs=FPs)
        # this lays down traces in W
        for i,g in enumerate(gng):
            if g == 'nogo':
                self.W[:,i,1] *= 1. 
                self.W[:,i,0] *= 0.
            elif g == 'relax':
                self.W[:,i,1] *= 0.0
                self.W[:,i,0] *= 0.0
        return


    def trace_expression(self, trs, inverse_relation = False):
        """
        Get a decay-weighted history of W on each trial, 
        and use this to arrive at a measure for preparation over time
        ...except for the first trial (has no history)
        So here:
        - make a w_filter with self.decay(trials, tau)
        - use convolution to get a history weighted memory trace
        - Compute weighted A and I over time within each trial.
        - Combine those (ratio) into a preparation measure related to RT 
        """

        # higher tau means faster decay -- less n-1 effect!
        # forgetting curve, i.e. filter
        w_filter = self.decay(trs.size, self.tau, self.tau_constant)            
        
        # padding; assume weights for each FP are zero before experiment
        pad = np.zeros_like(self.W)[:, :int(trs.size / 2), :]
        W_ = np.concatenate([pad, self.W, pad] ,axis = 1 )

        # W_p = past-weighted Weight-matrix
        self.W_p = scipy.ndimage.convolve1d(
            W_,w_filter,axis = 1,mode = 'constant', cval = 0)[:, :trs.size, :]

        # now we have, for every tr, the current state of Weights
        # use that to get I and A over time:
        A = np.einsum('nr,nt->rt',self.W_p[:, :, 0], self.timecells) 
        I = np.einsum('nr,nt->rt',self.W_p[:, :, 1], self.timecells) 

        # I/A ratio-- no computation possible for first trial (0/0)
        self.Mprep = np.ones_like(I) * np.nan
        self.Mprep[1:, :]  = I[1:, :]  / A[1:, :] 
               
        # Now, get per trial, the 'prep' at the critical moment:
        # start with nans:
        self.prep = np.ones_like(trs) * np.nan
        for tr in np.unique(trs):
            # Catch trials; no prep, no data -- nans:
            if np.isnan(tr):
                trial_indices = np.where(np.isnan(trs))[0]
                self.prep[trial_indices] = np.nan
                continue
            # else:
            trial_indices = np.where(trs == tr)[0]
            tidx = self._t2i(tr)
            preps = self.Mprep[trial_indices, tidx]
            # now, shift back again (trial [0] has prep=nan)
            self.prep[trial_indices] = preps

        return I, A # experiment has been run -- results are in self.prep



class FPexp(object):
    """
    Experiment object -- makes Foreperiod Paradigm Experiments.
    This structure assumes only 1 block of a single type of distrib.
    Transfer experiments use a different class, defined below.
    """
    
    def __init__(self, FPs, distribution = 'uni', tr_per_block = 500):
        """
        Initializes the attributes of the class: defines distribution and 
        list of FP trials. The class contains some build in distribution like
        the uniform ('uni'), anti-exponential ('anti'), and exponential 
        ('exp') FP distribution.
        """
        
        # Initialize pseudo-random generator; 
        # Guaranteeing reproducibility of the results
        np.random.seed(1)

        self.tr_per_block = tr_per_block
        # some 'famous' distributions:
        distrs = dict(
            uni = np.repeat(1, len(FPs)), # uniform distribution
            exp = [8,4,2,1], # exponential FP distribution
            anti = [1,2,4,8], # anti-exponential FP distribution
            exp_ = [8,4,1], # exponential FP distribution (3 FPs)
            anti_ = [1, 4, 8], # anti-exponential FP distribution (3 FPs)
            gauss   = [1,5,1,1], #c.f. Trillenberg, last one is for catch;
            constant = [1], # constant blocks only have a single FP
        )

        # if it's a 'famous' distribution, use predefined ratios
        if distribution in list(distrs.keys()):
            self.distribution_ = distrs[distribution]
        else:
            self.distribution_ = distribution

        # Take some liberties fitting the precise distributions into the
        # given tr_per_block; just treat these as proportions 
        self.distribution = np.array(self.distribution_, dtype=float)
        self.distribution /= self.distribution.sum() # proportions

        # Assume FPs are in seconds.
        self.FPs = np.array(FPs, dtype = float) # turns catch (None) into nan
        assert np.array(FPs).size == self.distribution.size

        # Make exp as a list of (1) block
        self.full_exp = [ self.make_block(tr_per_block) ]
        
        return


    def make_block(self, tr_per_block = 500):
        """
        Proportions * tr_per_block as an int is used to determine number of 
        trials for each FP -- might not adhere to the precise ratios given
        by distrib if it doesn't fit.
        """
        
        props = np.round(self.distribution * tr_per_block).astype(int)
        trs = np.repeat(self.FPs, props).tolist()
        np.random.shuffle(trs)

        # do not start with catch trial:
        while np.isnan(trs[0]):
            np.random.shuffle(trs)
            
        return trs


    def run_exp(self, model, inv_map = False, catch_dur = None):
        """
        Given an fMTP instance, run the experiment (self) and return a pandas 
        df with all the trials and their prescribed prep.
        """
        
        # run the exp, get the trace expressions back:
        all_trs = np.array(self.full_exp).flatten()
        if catch_dur is None:
            model.trace_formation(all_trs)
        else:
            model.trace_formation(all_trs, catch_dur = catch_dur)
        model.trace_expression(all_trs, inv_map)
        
        # now, store each trial in a pandas structure, for analysis
        # return FP, prep
        df = pd.DataFrame(zip(all_trs, model.prep))
        df.columns = ['FP', 'prep']
        df['FPn_1'] = np.r_[np.nan, all_trs[:-1]] # FPn-1
        df['factor2'] = np.r_[[np.nan, np.nan], all_trs[:-2]] # FPn-2
        df_Mprep = pd.DataFrame(model.Mprep)
        
        return df, df_Mprep

    

class FPgonogo(FPexp):
    """
    Extend the Foreperiod experiment with go-nogo trials. This effectively
    alters the run_exp function as it uses a different fMTP function
    """
    
    def __init__(self, relax = 1, *args, **kwargs):
        """
        Initializes the attributes of the class: defines list of FP trials
        """
        
        super(FPgonogo, self).__init__(*args, **kwargs)
        trs = np.array(self.full_exp).flatten()
        trs = np.sort(trs)
        
        # Alter the blocks in such a way that half are go and half are nogo
        if relax == 0:  
            assert trs.size % 2 == 0
            gng = np.tile(['go', 'nogo'], trs.size/2)
            self.full_exp = zip(trs, gng)
            np.random.shuffle(self.full_exp)
            while self.full_exp[0][1] == 'nogo':
                np.random.shuffle(self.full_exp)
        elif relax == 1:
            gng = np.tile(['go', 'go', 'nogo', 'relax'], int(trs.size/4))
            self.full_exp = list(zip(trs, gng))         
            np.random.shuffle(self.full_exp)
            while self.full_exp[0][1] != 'go':
                np.random.shuffle(self.full_exp)
        
        return
        
    
    def run_exp(self, model, inv_map = False, catch_dur=None):
        """
        Run the exp, get the trace expressions back:
        """
        
        all_trs = self.full_exp
        model.trace_formation_gonogo(all_trs)
        FPs = np.array([tr[0] for tr in all_trs])
        model.trace_expression(FPs, inv_map)

        # Store each trial in a pandas structure:
        # return FP, prep
        FPs, gng = zip(*all_trs)
        df = pd.DataFrame(zip(FPs, gng, model.prep))
        df.columns = ['FP','gng','prep']
        df['FPn_1'] = np.r_[np.nan, df.FP.iloc[:-1].values]
        df['factor2'] = np.r_[np.nan, df.gng.iloc[:-1].values]    
        df_Mprep = pd.DataFrame(model.Mprep)
        
        return (df, df_Mprep) 


class FPtransfer(object):
    """
    Transfer experiment -- multiple blocks with different distributions
    run_exp should run these blocks as if they're one -- but keep track
    of which block is what -- so that we can plot transfer effects
    Note -- these blocks typically only get _meaning_ when different 
    groups are run -- so 2 different sequences with different fMTP instances
    """   
    
    def __init__(self, FPs, transfer_group = 'exp_anti_group', 
                 tr_per_block = 500):
        """
        Initializes the attributes of the class: defines list of FP trials Aand
        the two 'particpant' groups
        """
        
        self.tr_per_block = tr_per_block
        self.FPs = FPs
        
        # Create gropus
        exp_group = dict(name = 'exp_group', 
                         distrib = ['uni', 'exp', 'exp', 'uni', 'uni'])
        anti_group = dict(name = 'anti_group',
                          distrib=['uni', 'anti', 'anti', 'uni', 'uni'])
        groups = dict(exp_group = [exp_group],
                      anti_group = [anti_group],
                      exp_anti_group = [exp_group, anti_group])
                 
        if transfer_group in groups.keys():
            groups = groups[transfer_group]
        else:
            groups = transfer_group
 
        # make experiments for both groups:
        for grp in groups:
            for dist in grp['distrib']:
                newblock = FPexp(FPs = self.FPs, distribution = dist,
                                 tr_per_block = tr_per_block)
                try:
                    fullexp.append(newblock)
                except NameError as e:
                    fullexp = [newblock]
            grp['blocks'] = fullexp
            del fullexp
        self.groups = groups
        
        return

    
    def run_exp(self, model, inv_map = False):
        """
        Run the exp, get the trace expressions back:
        """
        
        # for each group:
        for gr in self.groups:
            # make a dummy FPexp
            dummy = FPexp(self.FPs)
            dummy.full_exp = []
            # Merge all trials into one sequence:
            for bl in gr['blocks']:
                dummy.full_exp += bl.full_exp
            dummy.run_exp(model, inv_map = inv_map)
            gr['prep'] = model.prep # prep state at discrete FP
            gr['prep_con'] = model.Mprep # continuous prep. state
            gr['FP'] = np.array(dummy.full_exp).flatten()
            
        # format output:
        for gr in self.groups:
            df = pd.DataFrame(zip(gr['FP'], gr['prep'], gr['prep_con']))
            df.columns = ['FP', 'prep', 'prep_con']
            df['group'] = gr['name']
            df['block_index'] = np.repeat(np.arange(1, len(gr['blocks']) + 1), 
                                          self.tr_per_block)
            df['distrib'] = np.repeat(gr['distrib'], 
                                      self.tr_per_block).tolist()
            gr['dataframe'] = df
        
        # return 'full dataset'
        return (pd.concat([g['dataframe'] for g in self.groups]))