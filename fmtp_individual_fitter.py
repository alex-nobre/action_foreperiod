"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file implements the individual fitting procedure to obtain the
participant unique estimates for temporal smear (k), rate of forgetting (r),
memory persistence (c), and non-prepartion specific intercept (b). The fitting
procedure is applied on the empirical data of Los et al. 2017 experiment 
1. 

Note that the running time for this fitting procedure is long and probably 
cannot be performed on a personal computer. The fitting procedure has been
run on a external computer cluster an took longer than 1 day. 

The resulting parameter estimates can be viewed in the file
 "fmtp_individual_runner".

Josh Manu Salet
Email: saletjm@gmail.com
"""


## Read in libaries & classes
import numpy as np 
import pandas as pd
import pickle

# import optimizer
from scipy import optimize as op

# import classes fMTP and hazard-based models
from fmtp import fMTP, FPexp, FPtransfer


class fit_subs (object): 
    """ 
    This class performs the individual subjects fit of the Los et al. 2017 
    set. In this fitting procedure we search for subject specific forgetting
    parameters (r & c), temporal precision (k), offset (b). We leave the 
    scaling parameter (a) fixed across subjects
    """
    
    
    def __init__ (self, k_range, dist, seq, transf):
        """
        Initializes the attributes of the class:
        > k_range: discrete values reflecting temporal precision 
        > dist: empirical distribution effects
        > seq: empirical sequential effects
        > transf: empirical transfer effects
        """
        
        self.k_range = k_range
        self.dist = dist
        self.seq = seq
        self.transf = transf
        self.FP = np.unique(dist.FP) / 1000.
        
        return


    def fit_sub (self, p_val, p_across, p_unique, subs, bounds_opm, 
                 return_flag = False):
        """
        This function searches for a set of parameters that minimizes the 
        sum of squared errors across participants. The input parameter settings
        defines parameters that are fitted across or between participants. 
        
        Parameters
        ----------
        p_val : 
            value of parameter that is fixed across participants
        p_across : 
            dictionary containing the identities/keys of the parameters that 
            are fitted across participants
        p_unique : 
            dictionary containing the identities/keys of the parameters that 
            are fitted separately for each (unique) participant
        subs: 
            the participant of which the data is fitted
        bounds_opm :
            bounds that values of forgetting curve parameters can take
        return_flag : 
            If true returns values of the fit values for each subject
            If false returns sum squared error used to perform optimization 
            routine
        """  
        
        # Initialise dataframe to store subject specific parameters in
        coefs = pd.DataFrame(index = range(0, len(subs)), 
                             columns = ['sub', 'group', 'sse', 'R2',
                                        'r', 'c', 'a', 'b', 'k'])
        # Assign fit values to dictionary
        for i, ikey in enumerate(p_across.keys()): 
            p_across[ikey] = p_val[i]
        # Run for each subject 
        SSE = 0
        for i, isub in enumerate(subs):
            
            print(i, isub)
            # Select data according subject number
            sub_dist = self.dist[self.dist.Subject == isub]
            sub_seq = self.seq[self.seq.Subject == isub]
            sub_transf = self.transf[self.transf.Subject == isub]
            # Find best partiticpant unique parameters
            pvals = list(p_unique.values()) # within participant fit values
            
            # k only fitting parameter
            if (list(p_unique.keys()) == ['k']) & (return_flag == False): 
                sse = self.fit_unique(pvals, p_unique, p_across, sub_dist, 
                                      sub_seq, sub_transf)
            else: # search for optimal fitting values within participants
                fit_vals = op.minimize(self.fit_unique, pvals,
                                       method = 'L-BFGS-B', 
                                       args = (p_unique, p_across, sub_dist, 
                                               sub_seq, sub_transf), 
                                       bounds = bounds_opm, options = {
                                           'disp': False, 'ftol': 1e-3}) 
                sse = fit_vals.fun # subject specific sse
                pvals = fit_vals.x
            # Get SSE across participants (the thing to be optimised)
            SSE += sse
            
            if return_flag: # get back fitting values (don't optimise)
                    # Run function with the fitted parameters
                    sse, k, coef = self.fit_unique(pvals, p_unique, p_across, 
                                                   sub_dist, sub_seq, sub_transf, 
                                                   return_flag = True)
                    # Sort values in dataframe
                    p_coefs = {**coef, **p_across, **p_unique}
                    coefs['sub'].iloc[i] = isub
                    coefs['group'].iloc[i] = np.unique(sub_transf.group)
                    coefs['sse'].iloc[i] = sse
                    coefs['R2'].iloc[i] = p_coefs['R2'] 
                    coefs['k'].iloc[i] = k
                    coefs['r'].iloc[i] = p_coefs['r']
                    coefs['c'].iloc[i] = p_coefs['c']
                    coefs['a'].iloc[i] = 6.
                    coefs['b'].iloc[i] = p_coefs['b']
        
        if return_flag:
            return coefs
        else:
            return SSE
    
    
    def fit_unique (self, p_val, p_unique, p_across, sub_dist, sub_seq, 
                    sub_transf, return_flag = False):
        """
        This function search for the value of the temporal precision parameter 
        'k' that minimizes sum squared error. Since 'k' only takes on discrete
        values; permutate through the possible discrete values of k 
        (self.k_range) and run fMTP with the fitting values (r, c, and k).

        Parameters
        ----------
        p_val : 
            value of parameter unique to each subject
        p_unique : 
            dictionary containing the identities/keys of the parameters that 
            are fitted separately for each (unique) participant
        p_across : 
            dictionary containing the identities/keys of the parameters that 
            are fitted across participants
            p0_a : 
            scaling parameter to map preparation to RT (fixed across subjects)
        sub_dist : 
            subject specific distribution effect
        sub_seq : 
            subject specific sequential effect
        sub_transf : 
            subject specific transfer effect
        return_flag : TYPE, optional
            If true returns values of the participant's unique parameter set 
            minimizing sum of squared errors
            If false returns sum squared error used to perform optimization 
            routine
        """
        
        # Assign fit values to dictionary
        for i, ikey in enumerate(p_unique.keys()): 
            p_unique[ikey] = p_val[i]
        # Merge unique- and across parameters   
        p_all = {**p_unique, **p_across}
        # Permutate through k integers
        sse = 1e100        
        k = 0
        for ki in self.k_range:
            # Call fMTP class with subject-specific parameters
            fmtp = fMTP(p_all['r'], p_all['c'], ki) 
            # Run fMTP subject-specific class
            sse_, coef_ = self.run_fmtp(fmtp, p_all, sub_dist, sub_seq, 
                                        sub_transf)
            if sse_ < sse:
                sse = sse_
                k = ki
                coef = coef_

        if return_flag: 
            return sse, k, coef
        else: 
            return sse    
        
        
    def run_fmtp (self, fmtp_state, parms, sub_dist, sub_seq, sub_transf):
        """
        Runs the participant-specific fMTP class. It simulates the experiments

        Parameters
        ----------
        fmtp : 
            the fMTP class
        parms :
             the set of parameter values used to run fMTP
        
        """
        
        # Preparation effects for each distribution 
        distribs = np.unique(sub_dist.distrib)
        for distr in distribs:
            # Call experiment object to set up experiment
            exp = FPexp(FPs = self.FP, distribution = distr)           
            sim, prep = exp.run_exp(fmtp_state) # run experiment
            sim['distrib'] = distr 
            # Omit first trial, there is no preparation
            sim = sim.iloc[1:]
            sim['FPn_1'] = np.r_[np.nan, sim.FP.iloc[:-1].values]
            # Concatenate distributions in single dataframe
            try:
                all_sim = pd.concat([all_sim, sim]) # simulation sequece
            except NameError as e:
                all_sim = sim
        
        # Distribution effect
        avg_dist = all_sim.groupby(['distrib', 'FP']).mean()
        # Sequential effect
        avg_seq = all_sim.loc[all_sim['distrib'] == 'uni'].groupby([
            'FP', 'FPn_1']).mean()     
        # Transfer effect
        transfer_group = np.unique(self.transf.group)[0]
        transfer = FPtransfer(self.FP, transfer_group = transfer_group)                                                   
        sim_transf = transfer.run_exp(fmtp_state) # run experiment 
        sim_transf = sim_transf.iloc[1:, :]  
        # Sort output simulation
        sim_transf = sim_transf.loc[sim_transf['block_index'].isin([1, 4, 5])]
        sim_transf.loc[sim_transf['block_index'] == 1, 'distrib'] = 'uni_pre'
        sim_transf.loc[sim_transf['block_index'].isin([4, 5]),
                       'distrib'] = 'uni_post'
        avg_transf = sim_transf.groupby(['group', 'distrib', 'FP']).mean()
    
        # Concatenate sequential-, distribution-, and transfer effects
        sim = np.r_[avg_dist.prep.values, avg_seq.prep.values, 
                    avg_transf.prep.values]
        emp = np.r_[sub_dist.RT.values, sub_seq.RT.values, 
                    sub_transf.RT.values]  
    
        # Find parameters (a and b) mapping preparation onto RT 
        p0_coef = {'b': 375.}
        bounds_opm = [(200.0, 450.0)]

        fit_val = op.minimize(self.fit_coef, list(p0_coef.values()), 
                              method ='L-BFGS-B', args = (p0_coef, parms, sim, emp), 
                              bounds = bounds_opm,
                              options = {'disp': False, 'ftol': 1e-3})                               
        # Assign fit values to dictionary
        for i, key in enumerate(p0_coef):
            p0_coef[key] = fit_val.x[i]

        # Attach coefficient of determination to the coefficient dictionary
        p0_coef['R2'] = np.corrcoef(emp, sim)[0][1] ** 2.

        return (fit_val.fun, p0_coef)
    
    
    def fit_coef (self, coefs_val, coefs_dict, parms_dict, prep, emp):    
        """
        This functions maps preparation on RT using the linear mapping values
        a and b. Takes into account whether a and/or b are fitted across
        or within participants. Note that we constained a to a parameter 
        values of 6 (see line 306).

        Parameters
        ----------
        coefs_val : 
            value of linear parameters unique to each subject
        coefs : 
            dictionary containing the identities/keys of the linear values 
            unique for each participant
        parms:
            dictionary containing the identities/keys of linear values across
            participants
        prep : 
            fmtp's (discrete) preparation
        emp : 
            empirical RT grand averages

        """

        # Assign fit values to dictionary
        for i, key in enumerate(coefs_dict):
            coefs_dict[key] = coefs_val[i]

        pall_vals = {**coefs_dict, **parms_dict}
        # Convert preparatory state to RT
        RTmod = (6. * prep) + pall_vals['b']
        SSE = sum(pow(emp - RTmod, 2))    
                                          
        return SSE



if __name__ == '__main__':
        
    # Sort empirical data
    dat = pd.read_csv('empirical_data/sub_avg.csv')
    dat.loc[dat.Group == 'anti_exp', 'Group'] = 'anti_group'
    dat.loc[dat.Group == 'exp', 'Group'] = 'exp_group'
    dat = dat.rename(columns = {'Group' : 'group'})
    
    # Distribution effects
    emp_dist = dat.loc[dat.effect_type == 'distribution']
    emp_dist = emp_dist.rename(columns = {'FP_factor' : 'distrib'})
    emp_dist.loc[emp_dist.distrib == 'uniform_pre', 'distrib'] = 'uni'
    emp_dist = emp_dist.sort_values(by = ['distrib', 'FP'])
   
    # Sequential effects
    emp_seq = dat.loc[dat.effect_type == 'sequential']
    emp_seq = emp_seq.rename(columns = {'FP_factor' : 'FPn_1'})
    emp_seq['FPn_1'] = emp_seq['FPn_1'].astype(float)
    
    # Transfer effects
    emp_transf = dat.loc[dat.effect_type == 'transfer']
    emp_transf = emp_transf.rename(columns = {'FP_factor' : 'distrib'})
    emp_transf.loc[emp_transf.distrib == 'uniform_pre', 'distrib'] = 'uni_pre'
    emp_transf.loc[emp_transf.distrib == 'uniform_post', 'distrib'] = 'uni_post'
    emp_transf = emp_transf.sort_values(by = ['distrib', 'FP'])   
    
    
    # Fitting all parameters free between participants
    
    # Initialise fitting class
    k_range = np.arange(1, 13)
    fit = fit_subs(k_range, emp_dist, emp_seq, emp_transf)
    # Set starting parameters that can only vary across participants
    p_across = {'none': None}
    # Set starting participant unique values
    p_unique = {'r': -3. , 'c': 1e-4}
    bounds_unique =  ((-8., 0.), (0., 1e-2))
    # Select participants to fit
    subs = np.unique(emp_dist.Subject)
    coefs = fit.fit_sub([''], p_across, p_unique, subs, bounds_unique, 
                        return_flag = True)

    # Store estimates
    #with open('parameter_estimates', 'wb') as f:
    #    pickle.dump(coefs, f)