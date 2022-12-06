"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file provides insights in the effect of the value of fMTP parameters 
on its preparatory state:
    - k: temporal smear
    - r: rate of forgetting
    - c: memory persistence 

First, we display the effect on the parameters values on its internal dynamics.

Then we show their effect on preparation on all three canonical time scales:
the sequential, distribution, and transfer effects.

Josh Manu Salet
Email: saletjm@gmail.com
"""


# Import default pacakges
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Import classes
from fmtp import fMTP
from hazard import fMTPhz
from fit import sort_fit, show_fit 


# Read in and sort empirical data

# Distribution effects
emp = pd.read_csv('empirical_data/transfer.csv')
# Block 1 is the uniform distribution
coll_block1 = emp.loc[emp['block_index'] == 1]                      
uni = coll_block1.groupby(['FP']).mean()  
# Block 2 and 3 are the exponential and anti-exponential distributions
coll_block23 = emp.loc[emp['block_index'].isin([2,3])]              
anti_exp = coll_block23.groupby(['group_name', 'FP']).mean()  
# Reorganise
emp_uni = pd.DataFrame()
emp_uni['RT'] = uni.RT.values
emp_uni['distrib'] = 'uni'
emp_uni['FP'] = np.unique(emp.FP)
emp_anti = pd.DataFrame()
emp_anti['RT'] = anti_exp.loc['anti_exp_group'].RT.values
emp_anti['distrib'] = 'anti'
emp_anti['FP'] = np.unique(emp.FP)
emp_exp = pd.DataFrame()
emp_exp['RT'] = anti_exp.loc['exp_group'].RT.values
emp_exp['distrib'] = 'exp'
emp_exp['FP'] = np.unique(emp.FP)
# Concatenate
emp_dist = pd.concat([emp_uni, emp_anti, emp_exp])

# Sequential effects
emp = pd.read_csv('empirical_data/seq_effects.csv') 
emp = emp.loc[(emp['Block'] == 1)]
emp.FPn_1 /= 1000.
emp_seq = emp.groupby(['FPn_1', 'FP']).mean()

# Transfer effects
emp = pd.read_csv('empirical_data/transfer.csv')   
emp['distrib'] = np.nan
emp = emp.rename(columns = {'group_name' : 'group'})
emp.loc[emp['group'] == 'anti_exp_group', 'group'] = 'anti_group'
emp.loc[emp.group == 'anti_group', 'distrib'] = 'anti'
emp.loc[emp.group == 'exp_group', 'distrib'] = 'exp'
emp.loc[emp.block_index.isin([1]), 'distrib'] = 'uni_pre'
emp.loc[emp.block_index.isin([4, 5]), 'distrib'] = 'uni_post'
# Aggregate group by block, group and FP 
emp_transf = emp.groupby(['group', 'distrib', 'FP']).mean().reset_index()


def plot_parm (FP, parms, parm_change, show, plot_flag, 
               distribs =  ['uni', 'anti', 'exp']): 
    """
    This function displays fMTP's preparatory state for different parameters on
    the three canonical time scale; thereby providing insight in how each 
    parameter values affects fMTP's preparation. We do so by changing the 
    values of each unique value while leaving all other parameters constant
    """
    
    # General settings
    clr = ['#1f78b47d','#1b9e777d','#d95f027d', '#7570b37d', 
           '#dbb42c7d', '#7570b37d']

    # Time axis
    FP_ = FP * 1000.
    t = np.linspace(FP_[0] *1, FP_[-1], int(FP_[-1] - FP_[0])) 
    # Get values of parameter that is systematically changed
    pvals = parms[parm_change]
    
    for pval in pvals:
        # Initialise plot
        f, ax = show.init_plot(title = '', figwidth = 4.75, fighight = 4)
            
        # Run simulation
        parms[parm_change] = pval
        fmtp = fMTP(parms['r'], parms['c'], parms['k'])
        sorter = sort_fit(fmtp, hz_lin, hz_inv)

        # Sequential effects
        if plot_flag == 'sequential':
            sim, prep = sorter.sim_seq(FP)
            for i, FP1 in enumerate(FP):                                                      
                prep_FP1 = prep.loc[prep.FPn_1 == FP1]
                prep_FP1 = prep_FP1['fMTP'][int(FP_[0] - 1):int(FP_[-1] - 1)]        
                ax.plot(t, prep_FP1, color = clr[i], alpha = 1.) 
    
        # Distribution effects
        elif plot_flag == 'distribution':
            sim, prep = sorter.run_dist([None], FP, distribs)
            for i, dist in enumerate(distribs):  
                prep_dist = prep[prep['distrib'] == dist]
                prep_dist = prep_dist['fMTP'][int(FP_[0] - 1):int(FP_[-1] - 1)]        
                ax.plot(t, prep_dist, color = clr[i], alpha = 1.)   
        
        # Transfer effects  
        elif plot_flag == 'transfer':
            sim, prep = sorter.sim_transf(FP)
            prep = prep[prep.distrib == 'uni_post']
            for ig, gr in enumerate(np.unique(prep.group)[::-1]):
                prep_gr = prep[prep.group == gr]
                prep_gr = prep_gr['fMTP'][int(FP_[0] - 1):int(FP_[-1] - 1)]  
                ax.plot(t, prep_gr, color = clr[ig], alpha = 1.) 
    
        f.tight_layout()
        f.show()
                        
    return


if __name__ == '__main__':
    
    plt.close("all")
    
    # Dummy hazard models
    hz_lin = fMTPhz(-3.1, 5e-5, 4)    
    hz_inv = fMTPhz(-2.6, 2e-4, 4)
    
    # Foreperiods
    FP = np.array([0.4, 0.8, 1.2, 1.6])   
    # FP distributions
    distribs =  ['uni', 'anti', 'exp'] 
    
    # Parameter settings fMTP
    k = 4
    r = -2.81
    c = 0.0001
    
    
    # Display fMTP's internal dynamics for different parameter values
    
    
    # Temporal smear: k
    
    k_vals = [2, 4, 8] 
    
    xlim = [0, 3500]
    ylim = [0., 3.5]
    xticks = np.arange(500, 3000, 1000)
    yticks = [.75, 1.75, 2.75]
    show = show_fit(xlim, ylim, xticks, yticks, 
                    ylabel = 'Firing rate (a.u.)', xlabel = 'time (ms)')
    tau_min = 0.05
    tau_max = 5.0
    N = 50
    
    clrs = ['#a6a6a6','#118ab2', '#ef476f']
    f, ax = show.init_plot(title = '', figwidth = 7, fighight = 5.5)
    for ki in zip([4, 2, 8], clrs):
        for icell in [3, 10, 18]:
            t, A = fMTP._define_time_shankarhoward(tau_min, tau_max, N, ki[0], c)
            ax.plot(t[:35000] * 1000, A[icell, :35000], color = ki[1])
        f.tight_layout()
        f.show()
    
    
    # Rate of forgettign (r)
    
    r_vals = [-2.81, -4., -1.25] 
    
    n_trials = 40
    xticks = np.arange(5, n_trials, 10)
    xlim = [-0.5, n_trials]
    ylim = [-0.025, 0.45]
    yticks = np.arange(0, ylim[-1], .2) 
    show = show_fit(xlim, ylim, xticks, yticks, xlabel = 'Previous trial (n)',
                    ylabel = 'Memory strength')
    f, ax = show.init_plot(title = '', figwidth = 7., fighight = 5.)
    for ri, clr in list(zip(r_vals, clrs)):
        forget_curve = np.r_[0, np.arange(2, n_trials + 1)**ri + c]   
        ax.plot(np.arange(1, n_trials), forget_curve[1:], 
                marker = 'o', color = clr, 
                markevery = np.arange(2, 20, 2).tolist(),
                markerfacecolor = 'w', ms = 3.)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    f.tight_layout()
    f.show()
    
    
    # Memory persistence (c)
    
    c_vals = [1e-4, 0, 2e-4] 
    
    n_trials = 120
    xticks = np.arange(20, n_trials, 20)
    xlim = [0, n_trials]
    ylim = [-1e-4, 2.5e-3]
    yticks = np.arange(0, ylim[-1], 1e-3)
    show = show_fit(xlim, ylim, xticks, yticks, xlabel = 'Previous trial (n)',
                    ylabel = 'Memory strength')
    f, ax = show.init_plot(title = '', figwidth = 7., fighight = 5.)
    for ci, clr in list(zip(c_vals, clrs)):
        forget_curve = np.r_[0, np.arange(2, n_trials + 1)**r + ci]   
        ax.plot(np.arange(1, n_trials), forget_curve[1:], 
                marker = 'o', color = clr,
                markevery = np.arange(40, 100, 7).tolist(),
                markerfacecolor = 'w', ms = 3.)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    f.tight_layout()
    f.show()
    

    
    # Display effect of parameter values on fMTP's preparatory state
    
    FP_ = FP * 1000 
    xlim = [FP_[0] - 200, FP_[-1] + 200]
    xticks = np.arange(FP_[0], xlim[-1], 400)
    ylim = [0, 30]
    yticks = [5, 15, 25]

    # k values
    show = show_fit(xlim, ylim, xticks, yticks, ylabel = 'Preparation (a.u.)')
    
    k_lims = [2,8]
    parms_fmtp = {'r': r, 'c': c, 'k': k_lims}
    plot_parm(FP, parms_fmtp, 'k', show, 'sequential')
    parms_fmtp = {'r': r, 'c': c, 'k': k_lims}
    plot_parm(FP, parms_fmtp, 'k', show, 'distribution')
    parms_fmtp = {'r': r, 'c': c, 'k': k_lims}
    plot_parm(FP, parms_fmtp, 'k', show, 'transfer')
    
    # r values
    show = show_fit(xlim, ylim, xticks, yticks, ylabel = 'Preparation (a.u.)')
   
    r_lims = [-1.25, -4]
    parms_fmtp = {'r': r_lims, 'c': c, 'k': k}
    plot_parm(FP, parms_fmtp, 'r', show, 'sequential')
    parms_fmtp = {'r': r_lims, 'c': c, 'k': k}
    plot_parm(FP, parms_fmtp, 'r', show, 'distribution')
    parms_fmtp = {'r': r_lims, 'c': c, 'k': k}
    plot_parm(FP, parms_fmtp, 'r', show, 'transfer')
    
    # c values
    show = show_fit(xlim, ylim, xticks, yticks, ylabel = 'Preparation (a.u.)')
    
    c_lims = [2e-4, 0]
    parms_fmtp = {'r': r, 'c': c_lims, 'k': k}
    plot_parm(FP, parms_fmtp, 'c', show, 'sequential')
    parms_fmtp = {'r': r, 'c': c_lims, 'k': k}
    plot_parm(FP, parms_fmtp, 'c', show, 'distribution')
    parms_fmtp = {'r': r, 'c': c_lims, 'k': k}
    plot_parm(FP, parms_fmtp, 'c', show, 'transfer')