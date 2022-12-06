"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file implements the classes used to simulate different versions of
hazard accounts:
    
 - "fMTPhz" is a class that implements a subtype of fMTP: "fMTPhz". This class
   inhertis functionalities of its parent class 'fMTP'. 
 - "HazardStatic" is a class that simulates the static (i.e. no memory) 
   hazard models. That is the classical hazard and subjective hazard. 

Authors: Josh Manu Salet
Email: saletjm@gmail.com
"""

# Read in libaries & classes
import numpy as np 
import pandas as pd
from scipy.stats import norm
import scipy.ndimage
from fmtp import fMTP


class fMTPhz(fMTP):  
    """
    This classes builds on all properties of fMTP, but instead only defines an
    activation unit (instead of an additional inhibition unit) that reflects
    the probability densitiy function that is used for the hazard computation 
    (see trace expression)
    """
    
    def trace_formation(self, trs, catch_dur = 3.25):
        """
        For every FP, lay down a trace; effectively, this establishes a set of 
        Hebbian associations between each time cell and the Activation process
        'trs' are assumed to be a list of the FP on each trial.
        """
    
        # excitation 'zone' runs from FP and FP + 300ms
        zone_a = lambda fp: np.r_[fp, fp + 0.3]

        # initiallize the weight matrix
        self.W = np.zeros((self.timecells.shape[0], trs.size))
        # Trace _formation_ is not dependent on the past, only on FP. so we can
        # define traces for every unique FP, and lay those down wherever needed
        for tr in np.unique(trs):
            # define z_a
            z_a = zone_a(tr)
            # find the  trial indices where this trace is formed
            trial_indices = np.where(trs == tr)[0]
            # if catch trial define 'FP length' as trial duration (3.25 s)
            if np.isnan(tr): 
                z_a = zone_a(catch_dur)
                trial_indices = np.where(np.isnan(trs))[0]
                tr = catch_dur #re-define trial to 3.25 s
            # find nearest time-index:
            tidx = self._t2i(tr) # NB: this WON'T fail for nan or bounds
            # make_sure all indices are is in the time-cells matrix
            assert z_a[1] <= self.t.max()
            # make sure the estimate is not more than 1.5 dt off: 
            # = more than 1dt
            assert np.abs(self.t[tidx] - tr) < 1.5 * self.dt
            
            ## start making traces:
            ia = self._t2i(z_a[0]), self._t2i(z_a[1])
            activ = np.sum(self.timecells[:, ia[0]:ia[1] ], axis=1) * self.dt
            # lay down this trace in the association matrix:
            self.W[:, trial_indices] = activ[:,np.newaxis]
        
        return
    
    
    def trace_formation_gonogo(self, trs):
        """
        Los, 2013, has go / nogo trials with different FPs. Simulate these as
        if they were response-trials, but subsequently turn the excitation 
        these trials to zero
        """
        gng = [tr[1] for tr in trs]
        FPs = np.array([tr[0] for tr in trs])
        # first, pretend every trial = go:
        self.trace_formation(trs = FPs)
        # this lays down traces in W
        for i,g in enumerate(gng):
            if (g == 'nogo') or (g == 'relax'):
                self.W[:,i] *= 0.
                
        return
    
    
    def trace_expression(self, trs, inv_mapping = False):
        """
        Decay-weighted history of W on each trial, and use this to arrive at a 
        measure for preparation over time ...except for the first trial (has 
        no history). So here:
            - make a w_filter with self.decay(trials, tau)
            - use convolution to get a history weighted memory trace
            - normalize activation function of traces
            - get conditional probability of those activation functions
        """

        # higher tau means faster decay -- less n-1 effect!
        # forgetting curve, i.e. filter
        w_filter = self.decay(trs.size, self.tau, self.tau_constant)            
        
        # padding; assume weights for each FP are zero before experiment
        pad = np.zeros_like(self.W)[:, :int(trs.size / 2)]
        W_ = np.concatenate([pad, self.W, pad], axis = 1)

        # W_p = past-weighted weight-matrix
        self.W_p = scipy.ndimage.convolve1d(
            W_, w_filter, axis = 1, mode = 'constant', cval = 0)[:, :trs.size]

        # now we have, for every tr, the current state of weights
        # use that to get I and A over time:
        A = np.einsum('nr,nt->rt', self.W_p, self.timecells) 
        
        # make sure that the surface under the activation curves equals 1
        # note that these activation curves represent the pdf's of which we 
        # calculate the hazard 
        maxA = np.max(np.cumsum(A, axis = 1), axis = 1)
        maxA = np.array(maxA[:, np.newaxis], dtype = float)
        # First trial there is no preparation
        A[1:, :] = A[1:, :] / maxA[1:, :]
        # calculate hazard: first time point cdf must be zero
        pdf_fail = np.roll(A, 1, axis = 1)
        pdf_fail[:, 0] = 0
        cdf = np.cumsum(pdf_fail, axis = 1)
        # hazard function
        self.Mprep = np.ones_like(A) * np.nan
        if (inv_mapping):
            self.Mprep[1:, :] = 1. / (A[1:, :] / (1. - cdf[1:, :]))
        else:
            self.Mprep[1:, :] = A[1:, :] / (1. - cdf[1:, :]) 
            
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
            
        return # experiment has been run -- results are in self.prep
    
    
    def classic_hazard(self, densities):
        """
        Obtain the classic linear hazard of the discrete timepoints (= FP)
        """
        
        pdf = np.array(densities, dtype = float)
        pdf /= pdf.sum()
        # Calculate the hazard
        haz = self.haz(pdf)
        
        return haz
        
            
    def get_con_pdf(self, FPs, pdf_id, densities):
        """
        Computes the continuous probability density function from the discrete
        FPs. 
        """
        
        pdf = np.zeros_like(self.t)        
        if pdf_id == "uni":
            pdf[self._t2i(FPs[0]) : self._t2i(FPs[-1]) + 1] = 1.
        elif pdf_id == "constant":
            pdf[self._t2i(FPs[0])] = 1.
        elif (pdf_id == "exp") | (pdf_id == "anti"): 
            # Find the corresponding exponential function 
            FP_step_size = FPs[1] - FPs[0] 
            step_size = densities[1] / densities[0]
            growth_constant = step_size**(1. / FP_step_size)
            B = densities[0] / growth_constant**FP_step_size
            # Get pdf
            pdf[self._t2i(FPs[0]) : self._t2i(FPs[-1]) + 1] = \
            B*growth_constant**self.t[self._t2i(FPs[0]):self._t2i(FPs[-1]) + 1]
        elif pdf_id == "gauss":
            # These values are estimated in file "gauss_fit_trill.py"
            pdf = norm.pdf(self.t, 1.963, 0.685) * self.dt
        elif pdf_id == "bimodal":
            gauss = lambda x, mu, std : (
                1. / (std * np.sqrt(2. * np.pi))) * np.exp(
                    -0.5 * ((x - mu) / std)**2.)
            # Bimodal FP-Distributions
            pdf1 = gauss(self.t, FPs[0], self.dt)
            pdf2 = gauss(self.t, FPs[1], self.dt)
            pdf = pdf1 + pdf2
            
        return pdf
    
    
    def get_dis_pdf(self, FPs, pdf_id, densities, catch_dur = 3.25):
        """
        Get the 'discrete'probability density function according the
        distribution from which the FPs are drawn
        """
        
        # If replication Trillenberg add catch duration
        if pdf_id == 'gauss':
            FPs = np.r_[FPs[FPs != None], catch_dur]
            
        # Get FP indices 
        ifps = []
        for i in range(len(FPs)):
            ifps.append(self._t2i(FPs[i]))
        
        # Get pdf 
        pdf = np.zeros_like(self.t)   
        pdf[ifps] = densities

        return pdf
        
    
    def obj_hazard(self, FPs, pdf):
        """
        Obtain the "objective" hazard (no blurring)
        """
        
        # Scale the pdf such that surface of pdf equals 1
        pdf_scale = pdf / max(np.cumsum(pdf))
        # Calculate the hazard 
        obj_haz = self.haz(pdf_scale)
    
        return obj_haz


    def sub_hazard(self, FPs, pdf, phi):
        """
        Obtain the "subjective" hazard by blurring the probability densitiy 
        function with a normal distribution whose standard deviation is 
        proportional to elapsed time.
        """
        
        # initialize convolution
        t_ = self.t[:,np.newaxis]
        tau = t_.T
        
        # temporal blur (gaussian with increasing std)
        constant_term = (1. / (phi * t_ * np.sqrt(2. * np.pi)))
        gauss_kernel = constant_term * np.exp(-(tau - t_)**2.
                                              /(2. * phi**2. * t_**2.))
        
        # convolve the pdf with the gaussian kernel
        pdf_blur = gauss_kernel.dot(pdf)
        pdf_blur = np.squeeze(pdf_blur)        
        # scale such that surface of blurred pdf equals 1
        pdf_blur /= max(np.cumsum(pdf_blur))

        # calculate the hazard of the blurred pdf
        sub_haz = self.haz(pdf_blur)
        
        return pdf_blur, sub_haz


    @staticmethod
    def haz(pdf):        
        """
        calculate the hazard by implementing its classical function
        """
        
        # for the first time point the cdf must be zero         
        pdf_fail = np.roll(pdf, 1)                                                      
        pdf_fail[0] = 0
        cdf = np.cumsum(pdf_fail)
        haz = pdf / (1. - cdf) 
 
        return haz 
    
  
    
class HazardStatic(object):
    """
    Simulate the static (i.e. no memory) hazard models. That is the classical
    hazard and subjective hazard. 
    """
    
    def __init__(self, FPs, pdf_id = 'uni'):
        """
        Defines the probability density function according the foreperiod 
        distribution
        """
        
        self.FPs = FPs
        self.pdf_id = pdf_id
        
        # Some 'famous' distributions:
        pdfs = dict(
            exp     = [8.,4.,2.,1.],
            anti = [1., 2., 4., 8.], 
            exp_ = [8,4,1],
            anti_ = [1, 4, 8],
            uni = np.repeat(1, len(FPs)),
            constant = [1],
            gauss   = [1,5,1,1]
        )

        # If it's a 'famous' distribution, use predefined ratios
        if pdf_id in pdfs.keys():
            self.densities = pdfs[pdf_id]
        else:
            self.densities = pdf_id
                
        return
    
    
    def run_exp(self, hazard, model, inv_map = False, phi = 0.29, con_pdf = 0):
        """
        Get the hazard function given the FP-distribution properties
        """ 
        
        # if 'type_' is linear, get the discrete "classic" linear hazard 
        if model == "classical":
            hazard = hazard.classic_hazard(self.densities)
            if (inv_map):
                hazard = 1. / hazard
            # Return as pandas Dataframe object
            df = pd.DataFrame(zip(self.densities, hazard))
            df.columns = ['pdf', 'prep']  
        # Else, get the continuous linear and subjective hazard
        elif model == "subjective":
            if con_pdf == 1:
                pdf = hazard.get_con_pdf(self.FPs, self.pdf_id, self.densities)
            elif con_pdf == 0:
                pdf = hazard.get_dis_pdf(self.FPs, self.pdf_id, self.densities)
            obj_hz = hazard.obj_hazard(self.FPs, pdf)
            pdf_blur, prep = hazard.sub_hazard(self.FPs, pdf, phi)
            if (inv_map):
                prep = 1. / prep
            # Return as pandas Dataframe object
            df = pd.DataFrame(zip(pdf, pdf_blur, obj_hz, prep))
            df.columns = ['pdf', 'pdf_blur', 'obj_hz', 'prep']
            
        return df