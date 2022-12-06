"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file displays the results of the individual fitting procedure in which 
we estimated participant's unique parameter estimates for different components
of fMTP: 
    - k: temporal smear
    - r: rate of forgetting
    - c: memory persistence 
    - b: intercept (preparation unrelated)

First, we overview all participant unique parameter estimates in scatter 
plot and the qualtity of the fits by examining the coefficient of 
determination.

We then select a subset of participants that illustrate different ways in
which participants' preparatory behavior can differ.

Lastly, we provide the grand-average preparation curves for both participant
groups to show that the individual fitting procedure still adequately accounts
for the canonical grand-average effects.  

Josh Manu Salet
Email: saletjm@gmail.com
"""


# Import packages 
import numpy as np 
import pandas as pd
import pickle

from scipy import optimize as op 

# Import for plotting
import matplotlib.pyplot as plt

# Import classes
from fmtp import fMTP # fMTP class
from hazard import fMTPhz # hazard models
from fit import sort_fit, show_fit, get_fit # fitting procedure 

# Define colors for plotting
clr = ['#1f78b47d','#1b9e777d','#d95f027d', '#7570b37d', '#db4267', 
       '#7570b37d']


def findab (parm, prep, emp):    
    """
    Finding the linear constants (a and b) is done by minimizing the sum of
    squared error (SSE).
    """
    prep_ = (parm[0] * prep) + parm[1]
    
    return (sum(pow(emp - prep_, 2)))      
        

def get_R2 (coefs, subs):
     """
     This function get the coefficient of determination for each of the 
     unique subject fits
     """
     
     coefs['R2'] = np.nan
     for i, isub in enumerate(subs):

         sub_dist = emp_dist[emp_dist.Subject == isub]
         sub_seq = emp_seq[emp_seq.Subject == isub]
         sub_transf = emp_transf[emp_transf.Subject == isub]
         
         coefs_sub = coefs.loc[coefs['sub'] == isub] 
         distribs = np.unique(sub_dist.reset_index().distrib)
         trf_group = np.unique(sub_transf.group)[0]
         
         # Simulate experiment
         FP = np.unique(emp_dist.FP)
         fmtp = fMTP(coefs_sub.r.values[0], coefs_sub.c.values[0], 
                     coefs_sub.k.values[0])        
         sorter = sort_fit(fmtp, hz_lin, hz_inv)
         
         # Sequential
         sim_seq, prep_seq = sorter.sim_seq(FP)
         # Distribution
         sim_dist, prep_dist = sorter.run_dist([None], FP, distribs)
         # Transfer 
         sim_transf, prep_transf = sorter.sim_transf(FP, trf_group)
         
         transf_dist = np.unique(sub_transf.distrib)
         sim = np.r_[sim_seq.sort_values(by = 'FP').fMTP.values, 
                     sim_dist.fMTP.values, sim_transf[
                         sim_transf.distrib.isin(transf_dist)].fMTP.values]
         emp = np.r_[sub_seq.RT.values, sub_dist.RT.values, 
                     sub_transf.RT.values]  
         
         fit = op.minimize(findab, [4., 375], args = (sim, emp), 
                           method = 'L-BFGS-B')
         coefs['a'].iloc[i] = fit.x[0]
         coefs['b'].iloc[i] = fit.x[1]
         coefs['R2'].iloc[i] = np.corrcoef(emp, sim)[0][1] ** 2.
         
     return (coefs)
 

def scatter_coef (coef, title, ylim, yticks, ylabel, xlim = [], xticks = [], 
                  xlabel = 'N sort on R2', x2plot = [], clr = clr[0]):
    """
    Displays the participant unique parameter estimates in scatter plots
    """
    
    if len(x2plot) == 0:
        x2plot = np.arange(len(coef))
        xlim_off = round(len(coef) * .15)
        xlim = [-xlim_off, len(coef) + xlim_off]
        xrange = xlim[-1] - xlim[0]
        xticks = np.arange(xlim_off - 5, len(coef), round(xrange * .25))
    
    show = show_fit(xlim, ylim, xticks, yticks, xlabel = xlabel, 
                ylabel = ylabel)
    f, ax = show.init_plot(title = title, figwidth = 4., fighight = 4)
    ax.scatter(x2plot, coef, alpha = .1, color = clr)     
      
    return (f, ax)
    

if __name__ == '__main__':
    
    plt.close("all")
    
    # Dummy hazard models
    hz_lin = fMTPhz(-3.1, 5e-5, 4)    
    hz_inv = fMTPhz(-2.6, 2e-4, 4)
    
    # Sort empirical data
    dat = pd.read_csv('empirical_data/sub_avg.csv')
    dat.loc[dat.Group == 'anti_exp', 'Group'] = 'anti_group'
    dat.loc[dat.Group == 'exp', 'Group'] = 'exp_group'
    dat = dat.rename(columns = {'Group' : 'group'})
    dat.FP /= 1000.
    
    # Distribution effects
    emp_dist = dat.loc[dat.effect_type == 'distribution']
    emp_dist = emp_dist.rename(columns = {'FP_factor' : 'distrib'})
    emp_dist.loc[emp_dist.distrib == 'uniform_pre', 'distrib'] = 'uni'
    emp_dist = emp_dist.sort_values(by = ['distrib', 'FP'])
   
    # Sequential effects
    emp_seq = dat.loc[dat.effect_type == 'sequential']
    emp_seq = emp_seq.rename(columns = {'FP_factor' : 'FPn_1'})
    emp_seq['FPn_1'] = emp_seq['FPn_1'].astype(float)
    emp_seq.FPn_1 /= 1000.
    
    # Transfer effects
    emp_transf = dat.loc[dat.effect_type == 'transfer']
    emp_transf = emp_transf.rename(columns = {'FP_factor' : 'distrib'})
    emp_transf.loc[emp_transf.distrib == 'uniform_pre', 'distrib'] = 'uni_pre'
    emp_transf.loc[emp_transf.distrib == 'uniform_post', 'distrib'] = 'uni_post'
    emp_transf = emp_transf.sort_values(by = ['distrib', 'FP'])
    
    
    # Read the parameter estimates for each participant
    
    with open('parm_estimates', 'rb') as f:
        coefs = pickle.load(f)    
        
    # Selection of subjects
          # sub32: prototypical data
          # sub30: steep preparation
          # sub22: attenuated preparation
          # sub48: strong transfer effects 
    sub_select = np.array([32, 30, 22, 48])
    
    plotx = np.arange(len(coefs.k))
    xlim = [-5, 67]
    xticks = np.arange(5, 65, 25)
    
    # Sort subject on R2
    coefs = coefs.iloc[np.argsort(coefs.R2)]    
    
    # Coefficient of determination (R2)
    f, ax = scatter_coef(coefs.R2, '', 
                         [min(coefs.R2) - .1, max(coefs.R2) + .1],
                         np.arange(0.25, 1.0, .25), 'R2', xlim, xticks, 
                         xlabel = 'pp (ranked)', x2plot = plotx)
    # Identify specific subjects of interest
    for i, isub in enumerate(sub_select): 
        coef_sub = coefs[coefs['sub'] == isub]
        loc_sub = np.where(coefs['sub'] == isub)[0][0]
        ax.scatter(loc_sub, coef_sub.R2, color = clr[i + 1], alpha = .75)  
    f.tight_layout()
    f.show()
    
    
    # Temporal precision 
    f, ax = scatter_coef(coefs.k, '', [-1, 13], np.arange(2, 14, 4), 
                         'Uncertainty (k)', xlim, xticks, x2plot = plotx)
    
    # Identify specific subjects of interest
    for i, isub in enumerate(sub_select): 
        coef_sub = coefs[coefs['sub'] == isub]
        loc_sub = np.where(coefs['sub'] == isub)[0][0]
        ax.scatter(loc_sub, coef_sub.k, color = clr[i + 1], alpha = .75)  
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    f.tight_layout()
    f.show()
    
    # Forgetting curve (r)
    f, ax = scatter_coef(coefs.r * -1., '', 
                         [min(coefs.r * -1.) - 1, max(coefs.r * -1.) + 1],
                         [1.0, 4.0, 7.0][::-1], 'Rate (r)', xlim, xticks, x2plot = plotx)
    # Identify specific subjects of interest
    for i, isub in enumerate(sub_select): 
        coef_sub = coefs[coefs['sub'] == isub]
        loc_sub = np.where(coefs['sub'] == isub)[0][0]
        ax.scatter(loc_sub, coef_sub.r * -1., color = clr[i + 1], alpha = .75)  
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    f.tight_layout()
    f.show()

    # Forgetting curve (c)
    
    # remove three 'big' outliers c for illustrating purposes
    coefs.c = coefs.c[coefs.c < 0.001]
    plotx_c = np.arange(len(coefs.c))
    f, ax = scatter_coef(coefs.c, '', 
                         [min(coefs.c) - .0001, max(coefs.c) + .0001],
                         [2e-4, 5e-4, 8e-4], 
                         'Intercept (c)', xlim, xticks, x2plot = plotx_c)
    # Identify specific subjects of interest
    for i, isub in enumerate(sub_select): 
        coef_sub = coefs[coefs['sub'] == isub]
        loc_sub = np.where(coefs['sub'] == isub)[0][0]
        ax.scatter(loc_sub, coef_sub.c, color = clr[i + 1], alpha = .75)  
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    f.tight_layout()
    f.show()
    
    # Linear constant (b)
    f, ax = scatter_coef(coefs.b, '', 
                         [min(coefs.b) - 50, max(coefs.b) + 50],
                         [250, 350, 450], 'Offset (b)', xlim, xticks, 
                         x2plot = plotx)
    # Identify specific subjects of interest
    for i, isub in enumerate(sub_select): 
        coef_sub = coefs[coefs['sub'] == isub]
        loc_sub = np.where(coefs['sub'] == isub)[0][0]
        ax.scatter(loc_sub, coef_sub.b, color = clr[i + 1], alpha = .75)  
    f.tight_layout()
    f.show()
    
    
    
    # Preparation curves according subject's of interest    
    for isub in  sub_select:
        sub_dist = emp_dist[emp_dist.Subject == isub]
        sub_seq = emp_seq[emp_seq.Subject == isub]
        sub_transf = emp_transf[emp_transf.Subject == isub]
        
        coefs_sub = coefs.loc[coefs['sub'] == isub] 
        distribs = np.unique(sub_dist.reset_index().distrib)
        trf_group = np.unique(sub_transf.group)[0]
        
        # Simulate experiment
        FP = np.unique(emp_dist.FP)
        fmtp = fMTP(coefs_sub.r.values[0], coefs_sub.c.values[0], 
                    coefs_sub.k.values[0])
        sorter = sort_fit(fmtp, hz_lin, hz_inv)
        # Sequential
        sim_seq, prep_seq = sorter.sim_seq(FP)
        # Distribution
        sim_dist, prep_dist = sorter.run_dist([None], FP, distribs)
        # Transfer 
        sim_transf, prep_transf = sorter.sim_transf(FP, trf_group)
        
        FP = np.unique(emp_dist.FP)
        fmtp = fMTP(coefs_sub.r.values[0], coefs_sub.c.values[0], 
                    coefs_sub.k.values[0])
        sorter = sort_fit(fmtp, hz_lin, hz_inv)

        # Plot settings
        FP_ = FP * 1000 
        xlim = [FP_[0] - 200, FP_[-1] + 200]
        xticks = np.arange(FP_[0], xlim[-1], 400)
        ylim = [240, 505]
        yticks = np.arange(270, 500, 100)
        
        # Fit models and display preparation curves together with empirical data
        models = ['fMTP']
        
        # Sequential effects 
        show = show_fit(xlim, ylim, xticks, yticks)
        f, ax = show.init_plot(title = '', figwidth = 4.75, fighight = 4)
        ax.plot(xlim, np.repeat(coefs_sub.b, 2), ls = ':', 
                color = 'k', alpha = .5)
        show.show_seq('fMTP', [coefs_sub.a.values, coefs_sub.b.values], 
                      sub_seq, sim_seq, prep_seq, 1)
        
        # Distribution effects
        show = show_fit(xlim, ylim, xticks, yticks)
        f, ax = show.init_plot(title = '', figwidth = 4.75, fighight = 4)
        ax.plot(xlim, np.repeat(coefs_sub.b, 2), ls = ':', 
                color = 'k', alpha = .5)
        show.show_dist('fMTP', [coefs_sub.a.values, coefs_sub.b.values], 
                       sub_dist, '', prep_dist, 'distrib', 1)
        
        # Transfer effects
        show = show_fit(xlim, ylim, xticks, yticks)
        f, ax = show.init_plot(title = '', figwidth = 4.75, fighight = 4)
        ax.plot(xlim, np.repeat(coefs_sub.b, 2), ls = ':', 
                color = 'k', alpha = .5)
        prep_transf = prep_transf.groupby(
            ['distrib', 'index']).mean().reset_index()
        prep_transf = prep_transf[prep_transf['distrib'].isin(
            ['uni_pre', 'uni_post'])]
        show.show_dist('fMTP', [coefs_sub.a.values, coefs_sub.b.values], 
                       sub_transf, '', prep_transf, 'distrib', 1)
        
    
    # Display outcome average fitting procedure for both groups
    
    
    for i, igr in enumerate (['anti_group', 'exp_group']):
        
        emp_dist_across = emp_dist.loc[
            emp_dist['group'] == igr].groupby(['distrib', 'FP']).mean()
        emp_seq_across = emp_seq.loc[
            emp_seq['group'] == igr].groupby(['FP', 'FPn_1']).mean()
        emp_transf_across = emp_transf.loc[
            emp_transf['group'] == igr].groupby([
                'group', 'distrib', 'FP']).mean()
        
        sub_id = np.unique(emp_transf.loc[emp_transf['group'] == igr].Subject)
        coefs_gr = coefs.loc[coefs['sub'].isin(sub_id)]
        mean_r = np.mean(coefs_gr.r)
        mean_c = np.mean(coefs_gr.c)
        mean_a = np.mean(coefs_gr.a)
        mean_b = np.mean(coefs_gr.b)
        mean_k = round(np.mean(coefs_gr.k))
        
        # Simulate experiment
        FP = np.unique(emp_dist.FP)
        fmtp = fMTP(mean_r, mean_c, mean_k)
        sorter = sort_fit(fmtp, hz_lin, hz_inv)
        # Sequential
        sim_seq, prep_seq = sorter.sim_seq(FP)
        # Distribution
        distribs = np.unique(emp_dist_across.reset_index().distrib)
        sim_dist, prep_dist = sorter.run_dist([None], FP, distribs)
        # Transfer 
        sim_transf, prep_transf = sorter.sim_transf(FP, igr)
            
        # Plot settings
        FP_ = FP * 1000 
        xlim = [FP_[0] - 200, FP_[-1] + 200]
        xticks = np.arange(FP_[0], xlim[-1], 400)
        ylim = [297, 400] #[285, 435]
        yticks = [297, 345, 395] #np.arange(310, 411, 50)    
       
        # Fit models and display preparation curves together with 
        # empirical data
        models = ['fMTP']
        
        # Sequential effects 
        fit = get_fit(emp_seq_across, sim_seq.sort_values(by = 'FP'), 
                      prep_seq, 'seq')
        coef_seq, R_seq = fit.run_fit('sequential', 'fMTP')
        show = show_fit(xlim, ylim, xticks, yticks)
        show.init_plot(title = 'indiv. fits')
        show.show_seq('fMTP', [mean_a, mean_b], emp_seq_across, sim_seq,
                       prep_seq, 1)
        
        # Distribution effects
        fit = get_fit(emp_dist_across, sim_dist, prep_dist, 'distrib')
        coef_dist, R_dist = fit.run_fit('distribution', 'fMTP')
        show = show_fit(xlim, ylim, xticks, yticks)
        show.init_plot(title = 'indiv. fits')
        show.show_dist('fMTP', [mean_a, mean_b], emp_dist_across, '',
                       prep_dist, 'distrib', 1)
        
        # Transfer effects
        sim_transf = sim_transf[sim_transf['distrib'].isin([
            'uni_pre', 'uni_post'])]
        fit = get_fit(emp_transf_across, sim_transf, prep_transf, 'transf')
        coef_transf, R_transf = fit.run_fit('transfer', 'fMTP')
        show = show_fit(xlim, ylim, xticks, yticks)
        show.init_plot(title = 'indiv. fits')
        prep_transf = prep_transf.groupby(
            ['distrib', 'index']).mean().reset_index()
        prep_transf = prep_transf[prep_transf['distrib'].isin(
            ['uni_pre', 'uni_post'])]
        show.show_dist('fMTP', [mean_a, mean_b], emp_transf_across, '',
                       prep_transf, 'distrib', 1)