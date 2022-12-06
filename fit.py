"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file implements the "fit" classes used to execute the fitting-procedure
of different preparation models:
    - "sort_fit" is a class that is used to sort the output of the simulations 
      of the preparation models alongside the empirical data across all time
      scales.
    - "show_fit" is a class that is used to display the resulting fit of the 
      preparation models together with the empirical data. 
    - "get_fit" is a class that is used to fit a specified preparation model 
      onto the empirical data across the different time scales  

Authors: Josh Manu Salet 
Email: saletjm@gmail.com
"""

# Import packages
import numpy as np 
import pandas as pd

from scipy import optimize as op

# Import plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import classes
from fmtp import FPexp, FPtransfer
from hazard import HazardStatic

# Default plotting styles  
plt_emp = dict(s = 20., marker = 'o', facecolor = 'w') # empirical data   
plt_hz = dict(marker = 'd', ms = 4, markerfacecolor = 'w')  # hazard 
plt_emp_gauss = dict(s = 17.5, marker = 'o', facecolor = 'w') # gaussian  
   
# Legend 
mpl.rcParams["legend.frameon"] = False
mpl.rcParams['legend.fontsize'] = 8

# Axes 
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.linewidth']= 1.
mpl.rcParams['axes.labelsize'] = 8 
mpl.rcParams['axes.titlesize'] = 10

# Size ticks
mpl.rcParams['ytick.major.size'] = 5.
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.major.size'] = 5.
mpl.rcParams['xtick.major.width'] = .5

# Font size of the xtick labels
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8

# Line and marker properties
mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['lines.markersize'] = 5.
mpl.rcParams['lines.markeredgewidth'] = 1.



class sort_fit(object):
    """
    This class is used to sort the output of the simulations of the preparation
    models.
    """   
    
    def __init__(self, fmtp, hz_lin, hz_inv):
        """
        Initialize the attributes for the class to display results
        """
        
        self.fmtp = fmtp
        self.hz_lin = hz_lin
        self.hz_inv = hz_inv

        return
    
    
    def run_dist (self, FP_con, FP_var, factor):
        """ 
        This function runs all preparation models for the FP effects across 
        experimental blocks: distribution effects ('factor') for the
        constant- ('FP_con') and/or variable FP paradigm ('FP_var'). Returns 
        two dataframes: (1) discrete preparation points at the FPs and (2) 
        continuous preparation curves.    
        """
        
        from itertools import zip_longest  
        # For each experimental block
        for FPi, fac in zip_longest(FP_con, factor):
            # Determine whether constant FP or variable
            if FPi:
                FP = np.array([FPi])
            else:
                FP = FP_var
            # Memory fMTP models
            sim, prep = self.sim_dist(FP, fac)
            # Static (no memory) hazard models
            hz = HazardStatic(FPs = FP, pdf_id = fac)
            # Classical hazard
            prep_chz = hz.run_exp(self.hz_lin, 'classical') 
            prep_chz_inv = hz.run_exp(self.hz_inv, 'classical', inv_map = True) 
            # Subjective hazard
            prep_sub = hz.run_exp(self.hz_lin, 'subjective', phi = 0.5) 
            prep_sub_inv = hz.run_exp(self.hz_inv, 'subjective', 
                                      inv_map = True, phi = 0.29) 
            # Merge continuous static preparation with fMTP memory preparation
            prep['sub_hz'] = prep_sub.prep
            prep['sub_hz_inv'] = prep_sub_inv.prep
            prep['c_hz'] = np.nan
            prep['c_hz_inv'] = np.nan
            # Merge discrete static preparation with fMTP memory preparation
            if len(FP[FP == None]) > 0: # if catch trial
                sim['c_hz'] = prep_chz.prep.values[:-1]
                sim['c_hz_inv'] = prep_chz_inv.prep.values[:-1]
            else:
                sim['c_hz'] = prep_chz.prep.values
                sim['c_hz_inv'] = prep_chz_inv.prep.values
            sim['sub_hz'] = prep_sub.prep[
                np.array(FP[FP != None]*1000 - 1, dtype = int)].values
            sim['sub_hz_inv'] = prep_sub_inv.prep[
                np.array(FP[FP != None]*1000 - 1, dtype = int)].values
            try:
                allsim = pd.concat([allsim, sim])  
                allprep = pd.concat([allprep, prep])                                  
            except NameError as e:
                allsim = sim
                allprep = prep
        
        return allsim, allprep
    
    
    def sim_seq (self, FP, distr = 'uni'):
        """ 
        This function runs fMTP (and its subtype 'fMTPhz') and two dataframes
        sorted on the sequential effect containting the (1) average 
        preparation at discrete FPs ('sim') and (2) average preparation 
        curves ('prep').
        """
        
        # Run fMTP
        exp = FPexp(FPs = FP, distribution = distr, tr_per_block = 500)                             
        sim, prep_fmtp = exp.run_exp(self.fmtp) # fMTP 
        sim.rename(columns = {'prep' : 'fMTP'}, inplace = True)
        sim_hz_lin, prep_hz_lin = exp.run_exp(self.hz_lin) # fMTPhz
        sim['fMTPhz'] = sim_hz_lin.prep
        sim_hz_inv, prep_hz_inv = exp.run_exp(self.hz_inv, inv_map = True)
        sim['fMTPhz_inv'] = sim_hz_inv.prep # inverse fMTPhz
        
        # First trial has no preparation
        sim = sim.iloc[1:, :]
        prep_fmtp = prep_fmtp.iloc[1:, :]
        prep_hz_lin = prep_hz_lin.iloc[1:, :]
        prep_hz_inv = prep_hz_inv.iloc[1:, :]
        
        # Average preparation curves sort on FPn and FPn-1
        for FP1 in FP:
            prep = pd.DataFrame()
            index = np.where(np.array(sim.FPn_1) == FP1)[0]
            prep['fMTP'] = np.mean(prep_fmtp.iloc[index, :])
            prep['fMTPhz'] = np.mean(prep_hz_lin.iloc[index, :])
            prep['fMTPhz_inv'] = np.mean(prep_hz_inv.iloc[index, :])
            prep['FPn_1'] = FP1    
            try:
                allprep = pd.concat([allprep, prep])
            except NameError as e:
                allprep = prep
        
        # Average preparation sort on FPn and FPn-1
        sim = sim.groupby(['FPn_1', 'FP']).mean().reset_index()
        sim['distrib'] = distr
        
        return (sim, allprep)
    
    
    def sim_seq2 (self, FP, exp, distr = 'uni'):         
        """ 
        This function runs fMTP (and its subtype 'fMTPhz') and returns two 
        dataframes sorted on "second order" sequential effects (e.g., split 
        on "n-2" or type of trial) containting the (1) average preparation at  
        discrete FPs ('sim') and (2) the average preparation curves ('prep').
        """
        
        # Run fMTP            
        sim, prep_fmtp = exp.run_exp(self.fmtp)
        sim.rename(columns = {'prep' : 'fMTP'}, inplace = True)
        sim_hz_lin, prep_hz_lin = exp.run_exp(self.hz_lin) # fMTPhz
        sim['fMTPhz'] = sim_hz_lin.prep
        # fMTPhz inverse 
        sim_hz_inv, prep_hz_inv = exp.run_exp(self.hz_inv, inv_map = True) 
        sim['fMTPhz_inv'] = sim_hz_inv.prep 
        
        # First trial has no preparation
        sim = sim.iloc[1:, :]
        prep_fmtp = prep_fmtp.iloc[1:, :]
        prep_hz_lin = prep_hz_lin.iloc[1:, :]
        prep_hz_inv = prep_hz_inv.iloc[1:, :]
        
        # Average preparation curves sort on FPn, FPn-1, and FPn-2
        for FP1 in FP:
            prep_n1 = pd.DataFrame()
            index_n1 = np.where(np.array(sim.FPn_1) == FP1)[0]
            prep_n1['fMTP'] = np.mean(prep_fmtp.iloc[index_n1, :])
            prep_n1['fMTPhz'] = np.mean(prep_hz_lin.iloc[index_n1, :])
            prep_n1['fMTPhz_inv'] = np.mean(prep_hz_inv.iloc[index_n1, :])
            prep_n1['FPn_1'] = FP1
            prep_n1['factor2'] = np.nan
            try:
                allprep = pd.concat([allprep, prep_n1])
            except NameError as e:
                allprep = prep_n1
            for fac2 in np.unique(sim.factor2):
                prep_f2 = pd.DataFrame()
                index_f2 = np.where((np.array(sim.FPn_1) == FP1) & 
                                    (np.array(sim.factor2) == fac2))[0]
                prep_f2['fMTP'] = np.mean(prep_fmtp.iloc[index_f2, :])
                prep_f2['fMTPhz'] = np.mean(prep_hz_lin.iloc[index_f2, :])
                prep_f2['fMTPhz_inv'] = np.mean(prep_hz_inv.iloc[index_f2, :])
                prep_f2['FPn_1'] = FP1
                prep_f2['factor2'] = fac2
                allprep = pd.concat([allprep, prep_f2])
        
        # Average preparation sort on FPn and FPn-1
        sim = sim.groupby(['factor2', 'FPn_1', 'FP']).mean().reset_index()
        sim['distrib'] = distr
        
        return sim, allprep


    def sim_dist (self, FP, distr):
        """ 
        This function runs fMTP (and its subtype 'fMTPhz') and returns two 
        dataframes containting the (1) average preparation at discrete FPs 
        ('sim') and (2) average preparation curves ('prep') sort for the 
        distribution effects.
        """
        
        # Run fMTP 
        exp = FPexp(FPs = FP, distribution = distr, tr_per_block = 500)                          
        sim, prep_fmtp = exp.run_exp(self.fmtp) # fMTP 
        sim.rename(columns = {'prep' : 'fMTP'}, inplace = True)
        sim_hz, prep_hz = exp.run_exp(self.hz_lin) # fMTPhz
        sim['fMTPhz'] = sim_hz.prep
        sim_hz_inv, prep_hz_inv = exp.run_exp(self.hz_inv, inv_map = True) 
        sim['fMTPhz_inv'] = sim_hz_inv.prep # inverse fMTPhz
        
        # Average preparation at discrete FPs
        sim = sim.iloc[1:, :] # first trial has no preparation
        sim = sim.groupby('FP').mean().reset_index()
        sim['distrib'] = distr
    
        # Average preparation curves
        prep = pd.DataFrame()
        prep['fMTP'] = np.mean(prep_fmtp.iloc[1:, :]) # again remove 1st trial
        prep['fMTPhz'] = np.mean(prep_hz.iloc[1:, :])  
        prep['fMTPhz_inv'] = np.mean(prep_hz_inv.iloc[1:, :])
        prep['distrib'] = distr
        prep['FP'] = np.unique(sim.FP)[0]
        
        return(sim, prep)


    def sim_transf (self, FP, transf_gr = 'exp_anti_group'):
        """ 
        This function runs fMTP (and its subtype 'fMTPhz') and two dataframes
        containting the (1) average preparation at the discrete FPs ('sim') 
        and (2) the average preparation curves ('prep') sort for transfer 
        effects.
        """
        
        # Run fMTP
        exp = FPtransfer(FP, transfer_group = transf_gr, tr_per_block = 500)                         
        sim = exp.run_exp(self.fmtp) # fMTP
        sim.rename(columns = {'prep' : 'fMTP'}, inplace = True)
        sim_hz = exp.run_exp(self.hz_lin) # fMTPhz
        sim['fMTPhz'] = sim_hz.prep
        sim_hz_inv = exp.run_exp(self. hz_inv, inv_map = True) # inverse fMTPhz
        sim['fMTPhz_inv'] = sim_hz_inv.prep
    
        # First trial has no preparation
        sim = sim.iloc[1:, :]
        sim_hz = sim_hz.iloc[1:, :]
        sim_hz_inv = sim_hz_inv.iloc[1:, :]
        
        # Sort preparation curves
        for ig in np.unique(sim.group):
            gr = sim[sim.group == ig]
            gr_hz = sim_hz[sim_hz.group == ig]
            gr_hz_inv = sim_hz_inv[sim_hz_inv.group == ig]
            for ib in np.unique(sim.block_index):
                prep = pd.DataFrame()
                # Memory models
                prep['fMTP'] = np.mean(gr[gr.block_index == ib].prep_con)
                prep['fMTPhz'] = np.mean(
                    gr_hz[gr_hz.block_index == ib].prep_con)  
                prep['fMTPhz_inv'] = np.mean(
                    gr_hz_inv[gr_hz_inv.block_index == ib].prep_con) 
                # Grouping information
                prep['group'] = ig
                prep['block_index'] = ib
                prep['distrib'] = gr[gr.block_index == ib].distrib.values[0]
                try:
                    allprep = pd.concat([allprep, prep])
                except NameError as e:
                    allprep = prep      
    
        # Average preparation
        sim = sim.groupby(['group', 'block_index', 'distrib', 'FP']
                          ).mean().reset_index()
        sim.loc[sim.block_index == 1, 'distrib'] = 'uni_pre'
        sim.loc[sim.block_index.isin([4, 5]), 'distrib'] = 'uni_post'
        sim = sim.groupby(['group', 'distrib', 'FP']).mean().reset_index()    
        allprep = allprep.reset_index()
        allprep.loc[allprep.block_index == 1, 'distrib'] = 'uni_pre'
        allprep.loc[allprep.block_index.isin([4, 5]), 'distrib'] = 'uni_post'
        allprep = allprep.groupby(
            ['group', 'distrib', 'index']).mean().reset_index()
        
        return sim, allprep
        
 
        
class show_fit(object):
    """
    This class is used to display the resulting fit of all preparation 
    models together with the empirical data.
    """   
    
    def __init__(self, xlim, ylim, xticks, yticks,
                 xlabel = 'Foreperiod (ms)', ylabel = 'Reaction Time (ms)', 
                 clr_emp = ['#1f78b4cc','#1b9e77cc','#d95f02cc', '#7570b3cc', 
                            '#4ecdc4cc', '#ef476fcc'],
                 clr_sim = ['#1f78b4', '#1b9e77','#d95f02', '#7570b3', 
                            '#4ecdc4', '#ef476f']):

        """
        Initialize the attributes for the class to display the results
        """
                
        # Setting axes
        self.xlabel = xlabel 
        self.ylabel = ylabel 
        self.xlim = xlim
        self.ylim = ylim
        self.xticks = xticks
        self.yticks = yticks
        # Color settings (data and simulated)
        self.clr_emp = clr_emp 
        self.clr_sim = clr_sim
        
        return
    
    
    def cm2inch (self, value):
        """
        Convert cm to inches to set the correct size of the plot 
        """ 
        
        return value/2.54


    def init_plot (self, title = '', 
                  figwidth = 5.25, fighight = 6.25):
        """ 
        The plot initializer, sets all type of properties of the plot such as 
        size, title, labels, ticks, and limits
        """
        
        self.f, self.ax = plt.subplots(figsize = (self.cm2inch(figwidth), 
                                                  self.cm2inch(fighight)))
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_xticks(self.xticks)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_ylim(self.ylim)
        self.ax.set_yticks(self.yticks)
        self.ax.set_title(title)
        
        return (self.f, self.ax)
    

    def show_hazard (self, FP, sim, mod, dist, coef, l_style, clr):
        """
        Displays the classical hazard as a function of FP
        """

        b = coef[1]
        if (len(coef) > 2) & (dist == 'uni'):
            # Fitting constant 'a' was fixed across factors
            b = coef[2]
            
        sim2plot = sim[sim.distrib == dist]
        self.ax.plot(FP, (coef[0] * sim2plot[mod]) + b, 
                        color = clr, ls = l_style)
        return

        
    def show_pure (self, FP, prep, mod, coef, l_style, clr):
        """
        Displays results from constant FP paradigm. In contrast to variable FP 
        paradigm, displaying within-block variations of the RT-FP function, 
        the RT-FP function is here displayed across experimental blocks
        """
        
        FP_prep = np.unique(prep.FP)
        # Concatenate preparation across blocks 
        prepall = []
        for iFP in FP_prep:
            prepall.append(prep.loc[prep['FP'] == iFP][mod][int(iFP * 1000)])
        self.ax.plot(FP, (coef[0] * np.array(prepall)) + coef[1], 
                color = clr, ls = l_style)
        
        return
    
    
    def show_var (self, FP, prep, mod, coef, l_style, clr):
        """
        Displays RT-FP functions resulting from variable FP-paradigm.
        """  
        
        # Take in account constrains on fitting procedure. 
        b = coef[1]
        if len(coef) > 2:
            # Fitting constant 'a' was fixed across factors
            b = coef[2]
        t = np.linspace(FP[0], FP[-1], int(FP[-1] - FP[0])) 
        prep = prep[mod][int(FP[0] - 1):int(FP[-1] - 1)]        
        self.ax.plot(t, (coef[0] * prep) + b, color = clr, ls = l_style)      
            
        return
    
    
    def show_dist (self, mod, coef, emp, sim, prep, distr, R, 
                   clr_it = None, l_style = '-'):
        """ 
        This function returns the distribution effects (figures and quantitative
        information) from both the constant- and variable FP-paradigm. The 
        simulated preparation models are displayed as preparation curve 
        instead of average point estimates.
        """
        
        if mod[-3:] == 'inv':
            l_style = (0, (2., 2.))
        # Collapse over experimental blocks / groups with same FP distribution
        emp = emp.groupby(['distrib', 'FP']).mean().reset_index()
        # Distributions to be displayed
        distribs = prep[distr].unique()
        # Change order for consistency in color of plotting the distributions
        if np.array_equal(distribs, np.array(['anti', 'exp', 'uni_pre'])): 
            distribs = ['uni_pre', 'anti', 'exp']
        # For each distribution plot the RT-FP functions
        for i, dist in enumerate(distribs):  
            if not clr_it or (i > 0):
                clr_it = i
            # Empirical data
            FP = np.unique(emp.loc[emp[distr] == dist].FP) * 1000
            self.ax.scatter(FP, emp.loc[emp[distr] == dist].RT,
                            color = self.clr_emp[clr_it], **plt_emp)
            # Simulated 
            prep_dist = prep[prep[distr] == dist]
            # Classical (discrete hazard)
            if (mod == 'c_hz') or (mod == 'c_hz_inv'):
                sim = sim.groupby(['distrib', 'FP']).mean().reset_index()
                self.show_hazard(FP, sim, mod, dist, coef, l_style,
                                 self.clr_sim[clr_it])
            else:  # Constant-FP
                if dist == 'constant':
                    self.show_pure(FP, prep_dist, mod, coef, l_style, 
                                   self.clr_sim[clr_it])
                else: # Variable-FP
                    self.show_var(FP, prep_dist, mod, coef, l_style, 
                                  self.clr_sim[clr_it])
        self.f.tight_layout()
        self.f.show()         
        # Print results       
        self.print_results(title = '', model = mod, coef = coef, R = R)
    
        return 
         
    
    def show_seq (self, mod, coef, emp, sim, prep, R, l_style = '-'):
        """ 
        This function returns the sequential effects (figures and quantitative
        information). The simulated preparation models are displayed as
        preparation curve instead of average point estimates. 
        """
        
        # linestyle
        if mod[-3:] == 'inv':
            l_style = (0, (2., 2.)) 
        # Collapse over experimental blocks / groups with same FP distribution
        sim = sim.groupby(['FPn_1', 'FP']).mean().reset_index()
        emp = emp.groupby(['FPn_1', 'FP']).mean().reset_index()
        # Unique FPs
        FP = np.unique(prep.FPn_1)
        FP_ = FP * 1000.
        t = np.linspace(FP_[0], FP_[-1], int(FP_[-1] - FP_[0])) 
        # Plot for each grouped FPn-1 the RT-FP function and preparation curves
        for i, FP1 in enumerate(FP):                                                      
            # Empirical data
            emp_ = emp.loc[emp.FPn_1 == FP1]
            self.ax.scatter(FP_, emp_.RT.values, color = self.clr_emp[i],
                            **plt_emp)             
            # Simulated
            prep_FP1 = prep.loc[prep.FPn_1 == FP1]
            prep_FP1 = prep_FP1[mod][int(FP_[0] - 1):int(FP_[-1] - 1)]        
            self.ax.plot(t, (coef[0] * prep_FP1) + coef[1],
                         color = self.clr_sim[i], ls = l_style)   
        self.f.tight_layout()
        self.f.show()
        # Print results
        self.print_results(title = '', model = mod, coef = coef, R = R)

        return


    def show_seq2 (self, mod, coef, emp, sim, prep, R, 
                   figs, axes, l_style = '-'):
        """ 
        This function returns "second order" sequential effects (figures and 
        quantitative information). That is, the sequential effects split on an 
        additional factor of interest. For example, split on "n-2" or split on 
        type of trial (no/go-trials). The simulated preparation models are 
        displayed as preparation curve instead of averages point estimates. 
        """
        
        # linestyle
        if mod[-3:] == 'inv':
            l_style = (0, (2., 2.)) 
        FP = np.unique(prep.FPn_1)
        FP_ = FP * 1000.
        t = np.linspace(FP_[0], FP_[-1], int(FP_[-1] - FP_[0])) 
        prep = prep.dropna(subset = ['factor2'])
        # Plot for each grouped FPn-1 the RT-FP function
        for ifac, fac in enumerate(np.unique(sim.factor2)):                                                    
            emp_fac2 = emp.loc[emp.factor2 == fac]
            prep_fac2 = prep.loc[prep.factor2 == fac]
            for i, FP1 in enumerate(np.unique(sim.FPn_1)):
                # Empirical data
                emp_FP1 = emp_fac2.loc[emp_fac2.FPn_1 == FP1]
                axes[ifac].scatter(FP_, emp_FP1.RT.values, 
                                   color = self.clr_emp[i], **plt_emp)             
                # Simulated
                prep_FP1 = prep_fac2.loc[prep_fac2.FPn_1 == FP1]
                prep_FP1 = prep_FP1[mod][
                    int(FP_[0] - 1):int(FP_[-1] - 1)].values 
                axes[ifac].plot(t, (coef[0] * prep_FP1) + coef[1], 
                             color = self.clr_sim[i], ls = l_style)   
            figs[ifac].tight_layout()
            figs[ifac].show()
        # Print results
        self.print_results(title = '', model = mod, coef = coef, R = R)
        
        return axes


    def show_transf (self, mod, coef, emp, sim, prep, R, l_style = '-'):
        """ 
        This function returns the transfer effects: the pre-acquisition phase,
        acquisition phase, and test phase, for different group of participants
        The simulated preparation models are displayed as preparation curve 
        instead of average point estimates.
        """
        
        # linestyle
        if mod[-3:] == 'inv':
            l_style = (0, (2., 2.))
        # Distributions to be displayed
        distribs = np.unique(prep.distrib)
        for i, dist in enumerate(distribs):
            if (dist != 'anti') and (dist != 'exp'):
                self.init_plot(title = mod)
            emp_dist = emp.loc[emp.distrib == dist]
            prep_dist = prep.loc[prep.distrib == dist]
            FP = np.unique(sim.FP) * 1000
            for ig, gr in enumerate(np.unique(prep_dist.group)):
                if gr == 'exp_group':
                    clr_group = self.clr_sim[0]
                else:
                    clr_group = self.clr_sim[1]
                # Empirical data
                self.ax.scatter(FP, emp_dist.loc[emp_dist.group == gr].RT,
                                color = clr_group, **plt_emp)
                # Variable-FP
                self.show_var(FP, prep_dist[prep_dist.group == gr], mod, coef, 
                              l_style, clr_group)
            self.f.tight_layout()
            self.f.show()         
        # Print results       
        self.print_results(title = '', model = mod, coef = coef, R = R)
    
        return


    def show_gauss_rep (self, mod, coef, emp_clock, sim, prep, R, 
                       emp_noclock = [], clr_it = None, l_style = '-'):
        """ 
        This function returns the distribution effects from our gaussian
        replication assessments (figures and quantitative information). The 
        simulated preparation models are displayed as preparation curve 
        instead of average point estimates.
        """
        # linestyle
        if mod[-3:] == 'inv':
            l_style = (0, (2., 2.))
        if not clr_it:
                clr_it = 4
        # Empirical data
        FP = np.unique(sim.FP) * 1000
        self.ax.scatter(FP - 50, emp_clock.RT, color = self.clr_emp[clr_it],
                        **plt_emp_gauss)
        se = emp_clock.RT - emp_clock.lower
        self.ax.errorbar(FP - 50, emp_clock.RT, yerr = se, ls = 'none', 
                         elinewidth = 1, ecolor = self.clr_emp[clr_it],
                         zorder = 0)
        if len(emp_noclock) > 0: 
            self.ax.scatter(FP + 50, emp_noclock.RT, 
                            color = self.clr_emp[clr_it + 1], **plt_emp_gauss)
            se = emp_noclock.RT - emp_noclock.lower
            self.ax.errorbar(FP + 50, emp_noclock.RT, yerr = se, ls = 'none', 
                             elinewidth = 1, ecolor = self.clr_emp[clr_it + 1],
                             zorder = 0)
        # Simulated 
        if len(mod) > 0:
            if (mod == 'c_hz') or (mod == 'c_hz_inv'):
                self.show_hazard(FP, sim, mod, 'gauss', coef, l_style, 
                                 self.clr_sim[clr_it])
            else:
                self.show_var(FP, prep, mod, coef,
                              l_style, self.clr_sim[clr_it])
        self.f.tight_layout()
        self.f.show()         
        # Print results       
        self.print_results(title = '', model = mod, coef = coef, R = R)
        
        return
    
    
    def print_results (self, title, model, coef, R):
        """ 
        Print some of the key results (fitting parameters 'a' and 'b' and 
        the coefficient of determination (R2). 
        """
        
        # Title of study
        if title:
            print('')
            print('-' * len(title))
            print(title)
            print('-' * len(title)) 
        # Results from different subtypes of preparation models
        elif len(coef) > 0: 
            print('     ' + model + ": ")
            if len(coef) > 2:
                print('       ' + 'a = ', "%.2f" %coef[0] + ', ', 'b = ', 
                      "%.2f" %coef[1] + ', ', 'b2 = ', "%.2f" %coef[2])
            else:            
                print('       ' + 'a = ', "%.2f" %coef[0] + ', ', 
                      'b = ', "%.2f" %coef[1]),
            print('       ' +  'R^2 = ' "%.2f" %R)
    
        return

    

class get_fit (object):  
    """      
    This class is used to fit a specified preparation models onto the 
    empirical preparation effects across the different time scales
    """
    
    def __init__ (self, emp, sim, prep, factor, p0 = [4., 375.]):
        """
        Initialize the attributes for the class that reflects the least-squares
        regression to come from simulated preparation to RT
        """
             
        self.factor = factor # conditions of experiment
        self.p0 = p0 # starting values optimilization
        self.sim = sim # simulated preparation as function of FP
        self.emp = emp # empirical data as function of FP
        self.prep = prep # average preparation curves
        
        return
    
    
    def temp2RT (self, value, a, b):
        """
        Helper function to map preparation linearly to RT
        """
    
        return (a * value) + b

    
    def do_fit (self, mod):
        """ 
        Runs the optimizer
        """
        
        fit = op.minimize(self.sse, self.p0, args = (mod), 
                          method = 'L-BFGS-B')
        
        return (fit)
    
    
    def sse (self, parm, mod):    
        """
        Finding the linear constants (a and b) is done by minimizing the sum of
        squared error (SSE). Input:
            - parm: linear constants (a and b)
            - prep: Preparation
            - emp: emperical data
        """
                   
        # Both 'a' and 'b' can vary between conditions
        if len(parm) == 2:
            prep_ = self.temp2RT(self.sim[mod].values, *parm)  
            SSE = sum(pow(self.emp.RT.values - prep_, 2))  
        # Fixate 'a': only 'b' can vary across conditions
        else:
            SSE = 0 
            # Fixate 'a' across the conditions ('factor'), leave 'b' free
            for i, fac in enumerate(np.unique(self.sim[self.factor])):
                prep_ = self.temp2RT(self.sim[
                    self.sim[self.factor] == fac][mod].values,
                    parm[0], parm[i + 1]) 
                emp_ = self.emp.loc[self.emp[self.factor] == fac].RT.values
                SSE += sum(pow(emp_ - prep_, 2))                                          
            
        return SSE


    def get_R (self, parm, mod):
        """
        Gets the coefficient of deterimination
        """ 
        
        # Get R from unconstrained fitting procedure
        if len(parm) == 2.:
            R = np.corrcoef((parm[0] * self.sim[mod].values) + parm[1],
                            self.emp.RT)[0][1] ** 2.
        else: # Get R from constrained fitting procedure
            sim_val = []
            emp_val = []
            for i, fac in enumerate(np.unique(self.sim[self.factor])):
                sim_val.append((parm[0] * self.sim.loc[
                    self.sim[self.factor] == fac][mod].values) + parm[i + 1])
                emp_val.append(self.emp.loc[self.emp[self.factor] == fac].RT)
            R = np.corrcoef(np.concatenate(sim_val), 
                            np.concatenate(emp_val))[0][1] ** 2.
            
        return R  
    
    
    def run_fit (self, name_study, models, 
                 show_fit = False, emp_ = [], clr_it = None):
        """
        Runs the fitting procedure and obtained the results (figures and 
        print key results) for a specified FP study ('name_study') and 
        specified preparation model.
        """
        
        if show_fit == False:
            # Perform least-square regression
            fit_map = self.do_fit(models)
            # Obtain coefficient of determination
            R = self.get_R(fit_map.x, models)
        else:
            show_fit.print_results(title = name_study, model = '', 
                                   coef = '', R = '')
            # Get resulting qualitative and quantitative fit for each model
            for mod in models:
                show_fit.init_plot(title = mod[0])
                # Do this for linear- and inverse mapping
                for sub_mod in mod:
                    fit_map = self.do_fit(sub_mod)
                    # Obtain coefficient of determination
                    R = self.get_R(fit_map.x, sub_mod)
                    if self.factor == 'distrib':
                        show_fit.show_dist(sub_mod, fit_map.x, self.emp, 
                                           self.sim, self.prep, self.factor,
                                           R, clr_it)
                    elif self.factor == 'seq': # Sequential effects
                        show_fit.show_seq(sub_mod, fit_map.x, self.emp, 
                                          self.sim, self.prep, R)
                    elif self.factor == 'seq2': # 2nd order sequential effects
                        show_fit.show_seq2(sub_mod, fit_map.x, self.emp, 
                                           self.sim, self.prep, R)
                    elif self.factor == 'transf': # Transfer effects
                        show_fit.show_transf(sub_mod, fit_map.x, self.emp, 
                                             self.sim, self.prep, R)
                    elif self.factor == 'gauss_rep':
                        show_fit.show_gauss_rep(sub_mod, fit_map.x, self.emp, 
                                                self.sim, self.prep, R, 
                                                emp_, clr_it)
        return (fit_map.x, R)