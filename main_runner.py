"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

This file runs the simulations of the empirical studies reported in the 
manuscript.

We first define each simulation in a separate function. Running this file
as the main script runs all six simulations by default. The user can 
comment/uncomment these functions to select simulations of interest.

For each simulation different preparation models are fitted to the empirical
grand averages: classical hazard, subjective hazard, fMTPhz and fMTP. 

The results of the simulations are plotted alongside the empirical data of the
original studies being simulated. The functions in this file assume that these
are located in the same folder.

Author: Josh Manu Salet
Email: saletjm@gmail.com
"""


# Import packages 
import numpy as np 
import pandas as pd

# Import for plotting
import matplotlib.pyplot as plt

# Import classes
from fmtp import fMTP, FPexp, FPgonogo
from hazard import fMTPhz
from fit import sort_fit, show_fit, get_fit 

# Suppress warning for readibility 
import warnings
warnings.filterwarnings("ignore")

def convarFP_niemi1979 (sorter):
    """
    Here we simulate Niemi's (1979) study. We use the grand averages
    reported in Experiment 1 for the bright stimulus condition in the variable
    FP paradigm (uniform FP-distribution) and the constant-FP paradigm reported
    in Experiment 2.
    doi: 10.1016/0001-6918(79)90038-6
    """
    
    # Empirical data
    emp_dat = pd.read_csv('empirical_data/niemi1979.csv')
    # Discrete FPs 
    FP_con = np.array([0.5, 5.5]) # constant
    FP_uni = np.arange(0.5, 3.5, 0.5) # uniform
    # Sort empirical data
    emp_con = emp_dat.loc[emp_dat.condition == 'bright_pure'].RT.values        
    emp_uni = emp_dat.loc[emp_dat.condition == 'bright'].RT.values  
    emp = pd.DataFrame({'RT' : np.r_[emp_con, emp_uni], 'distrib': np.r_[
        ['constant'] * len(emp_con), ['uni'] * len(emp_uni)]})
    emp['FP'] = np.r_[FP_con, FP_uni]
    
    # Simulate experiment
    sim_con, prep_con = sorter.run_dist(FP_con, None, np.repeat(['constant'], 
                                                                  len(FP_con)))
    sim_uni, prep_uni = sorter.run_dist([None], FP_uni, ['uni'])
    sim = pd.concat([sim_con, sim_uni]).reset_index()
    
    # Plot settings
    sim.loc[sim.FP == 5.5, 'FP'] = 4.25 # change x-axis
    emp.loc[emp.FP == 5.5, 'FP'] = 4.25
    xlim = [0, 4250 + 500]
    xticks = np.arange(500, 5001, 1250)
    ylim = [200, 350]
    yticks = np.arange(225, 351, 50)
    
    # Fit models and display preparation curves together with empirical data
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv'), 
              ('sub_hz', 'sub_hz_inv'), ('c_hz', 'c_hz_inv')]
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # Fit constant and uniform paradigm separately
    emp_con = emp[emp.distrib == 'constant']
    emp_uni = emp[emp.distrib == 'uni']
    # Start fitting procedure looping through all the models
    show.print_results(title = 'Niemi (1979)', model = '', coef = '', R = '')
    for mod in models: 
        show.init_plot(title = mod)
        for sub_mod in mod:
            # Fit models
            fit = get_fit(emp_con, sim_con, prep_con, 'distrib') # constant
            coef_con, R_con = fit.run_fit('Niemi (1979)', sub_mod)

            fit = get_fit(emp_uni, sim_uni, prep_uni, 'distrib')  # uniform
            coef_uni, R_uni = fit.run_fit('Niemi (1979)', sub_mod)
            # Show plots 
            print('Constant FP paradigm: ')
            show.show_dist(sub_mod, coef_con, emp_con, sim_con, prep_con, 
                           'distrib', R_con, clr_it = 0)
            print('Uniform FP paradigm: ')
            show.show_dist(sub_mod, coef_uni, emp_uni, sim_uni, prep_uni, 
                           'distrib', R_uni, clr_it = 1)
            print('')
    
    # Fit models but now constrain "a" across the two experiments for fMTP
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv')]
    prep_all = pd.concat([prep_con, prep_uni])
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # Memory models: distribution: effects
    fit = get_fit(emp, sim, prep_all, 'distrib', p0 = [4., 375, 375]) 
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv')] 
    fit.run_fit('Niemi (1979): fixate scaling parm. across the two experiments', 
                models, show_fit = show, clr_it = 0)

    return


def convarFP_seq_los2001 (sorter):    
    """
    Here we simulate Los and Heuvel's (2001) study. We focus on those blocks 
    in which participants were not informed about the nature of the upcoming
    FP (neutral-cue condition) and average over different payout regimes 
    (reward vs. non-reward). From this dataset we retrieve distribution effects
    from the constant- and variable FP-paradigm (uniform FP distribution). 
    Additionally, we retrieve sequential effects from the variable FP-paradigm.
    doi: 10.1037/0096-1523.27.2.370
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/losheuvel_dist.csv')
    # Discrete FPs
    FP = np.unique(emp.FP)
    # Merge constant and variable FP paradigm
    emp_con = (emp.RTpure + emp.RTpure_reward) / 2. # constant
    emp_uni = (emp.RTuni + emp.RTuni_reward) / 2. # uniform
    # Re-sort data
    emp_block = pd.DataFrame({'RT' : np.r_[emp_con, emp_uni], 'distrib': 
                              np.repeat(['constant', 'uni'], len(emp_con))})
    emp_block['FP'] = np.r_[FP, FP]
    # Sequential effects
    emp_seq = pd.read_csv('empirical_data/losheuvel_seq.csv')
    emp_seq.RT = (emp_seq.RT + emp_seq.RTreward) / 2.
    emp_seq = emp_seq.drop('RTreward', axis = 1)
    # Merge RT across blocks with sequential RTs
    emp = pd.concat([emp_block, emp_seq])
    
    # Simulate experiment: distribution effects
    sim_con, prep_con = sorter.run_dist(FP, None, 
                                        np.repeat(['constant'], len(FP)))
    sim_uni, prep_uni = sorter.run_dist([None], FP, ['uni'])
    sim_block = pd.concat([sim_con, sim_uni])
    prep_block = pd.concat([prep_con, prep_uni])
    # Simulate experiment: sequential effects
    sim_seq, prep_seq = sorter.sim_seq(FP)
    # Merge preparation across blocks with sequential RTs
    sim = pd.concat([sim_block, sim_seq])
    
    # Plot settings
    FP = FP * 1000 
    xlim = [FP[0] - 500, FP[-1] + 500]
    xticks = np.arange(FP[0], xlim[-1], 500)
    ylim = [200, 280]
    yticks = np.arange(220, 280, 20)    
    
    # Fit models and display preparation curves together with empirical data
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv'), 
              ('sub_hz', 'sub_hz_inv'), ('c_hz', 'c_hz_inv')]
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # Memory models: distribution: effects
    fit = get_fit(emp, sim, prep_block, 'distrib')     
    coef_dist, R_dist = fit.run_fit('Los (2001)', models, show_fit = show)    
    # Static models: distribution effects
    models = [('sub_hz', 'sub_hz_inv'), ('c_hz', 'c_hz_inv')] 
    fit = get_fit(emp_block, sim_block, prep_block, 'distrib') 
    fit.run_fit('', models, show_fit = show)
    # Memory models: sequential: effects
    ylim = [215, 310]
    yticks = np.arange(225, 301, 25)  
    show = show_fit(xlim, ylim, xticks, yticks)
    fit = get_fit(emp, sim, prep_seq, 'seq') 
    models = [('fMTP', 'fMTPhz_inv')] 
    coef_seq, R_seq = fit.run_fit('', models, show_fit = show)
    
    # Get corrected R2 for fMTP models, a en b constrained across simulations,
    # but get R2 for this specific dataset (instead of "overall" R2)
    fit = get_fit(emp_block, sim_block, prep_block, 'distrib') 
    fitseq = get_fit(emp_seq, sim_seq, prep_seq, 'seq') 
    print("")
    print("-----------------------------------------")
    print("Corrected R2 for fMTP models: ")
    print("-----------------------------------------")
    print("")
    for imod in ['fMTP', 'fMTPhz', 'fMTPhz_inv'] :
        Rdist = fit.get_R(coef_dist, imod)
        Rseq = fitseq.get_R(coef_seq, imod)
        print(imod, 'distribution: corrected R: ', "%.2f" %Rdist)
        print(imod, 'sequential: corrected R: ', "%.2f" %Rseq)
        print("")
    return


def distrib_transfer_los2017 (sorter):
    """
    Here we simulate Los, Kruijne, and Meeter's (2017) study. In this
    simulation, we focus on the distribution effects for the three FP
    distributions (uniform, anti-exponential, and exponential). Additionally,
    we plot the transfer effects. We focus on the data of Experiment 1. 
    doi: 10.1037/xhp0000279
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/transfer.csv')
    FP = np.unique(emp.FP) # discrete FPs
    # Resort data
    emp['distrib'] = np.nan
    emp = emp.rename(columns = {'group_name' : 'group'})
    emp.loc[emp['group'] == 'anti_exp_group', 'group'] = 'anti_group'
    emp.loc[emp.group == 'anti_group', 'distrib'] = 'anti'
    emp.loc[emp.group == 'exp_group', 'distrib'] = 'exp'
    emp.loc[emp.block_index.isin([1]), 'distrib'] = 'uni_pre'
    emp.loc[emp.block_index.isin([4, 5]), 'distrib'] = 'uni_post'
    emp = emp.groupby(['group', 'distrib', 'FP']).mean().reset_index()
    
    # Simulate experiment
    sim, prep = sorter.sim_transf(FP)

    # Plot setting
    FP_ = FP * 1000 
    xlim = [FP_[0] - 200, FP_[-1] + 200]
    xticks = np.arange(FP_[0], xlim[-1], 400)
    ylim = [240, 435]
    yticks = np.arange(310, 411, 50)    
   
    # Fit models and display preparation curves together with empirical data
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv')]
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # Memory models: distribution effects
    prep_dist = prep.groupby(['distrib', 'index']).mean().reset_index()
    # Re-sort preparation and empirical data
    prep_dist = prep_dist[prep_dist.distrib != 'uni_post']
    # Fit memory models
    fit = get_fit(emp, sim, prep_dist, 'distrib') 
    fit.run_fit('Los et al. (2017): distribution effects', models, 
                show_fit = show)
    # Static hazard models: distribution effects
    emp_dist = emp.groupby(['distrib', 'FP']).mean().reset_index()
    emp_dist = emp_dist[emp_dist.distrib != 'uni_post']
    models = [('sub_hz', 'sub_hz_inv'), ('c_hz', 'c_hz_inv')] 
    sim_static, prep_static = sorter.run_dist([None], FP, 
                                              ['anti', 'exp','uni'])
    sim_static.loc[sim_static.distrib == 'uni', 'distrib'] = 'uni_pre'
    prep_static.loc[prep_static.distrib == 'uni', 'distrib'] = 'uni_pre'
    fit = get_fit(emp_dist, sim_static, prep_static, 'distrib') 
    fit.run_fit('', models, show_fit = show)
    
    # Transfer effects
    fit = get_fit(emp, sim, prep, 'transf') 
    coef_fmtp, R_fmtp = fit.run_fit('', 'fMTP')
    coef_hz, R_hz = fit.run_fit('', 'fMTPhz_inv')
    
    # Get corrected R2 for fMTP models, a en b constrained across simulations,
    # but get R2 for this specific dataset (instead of "overall" R2)
    sim_dist = sim.groupby(['distrib', 'FP']).mean().reset_index()
    sim_dist = sim_dist[sim_dist.distrib != 'uni_post']
    fit_dist = get_fit(emp_dist, sim_dist, prep_dist, 'distrib')
    fit_transf = get_fit(emp, sim, prep, 'transf')
    print("")
    print("-----------------------------------------")
    print("R2 fMTP models separately for distribution- and transfer effects: ")
    print("-----------------------------------------")
    print("")
    for imod in ['fMTP', 'fMTPhz_inv'] :
        Rdist = fit_dist.get_R(coef_fmtp, imod)
        Rtransf = fit_transf.get_R(coef_hz, imod)
        print(imod, 'distribution: corrected R: ', "%.2f" %Rdist)
        print(imod, 'transfer: corrected R: ', "%.2f" %Rtransf)
        print("")
        
    # Transfer effects: display fMTP and fMTPhz in same figure
    distribs = np.unique(prep.distrib)
    plt_emp = dict(s = 25., marker = 'o', facecolor = 'w')    
    clr_sim = ['#1b9e77','#d95f02']
    clr_emp = ['#1b9e77cc','#d95f02cc']
    for i, dist in enumerate(distribs):
        if (dist != 'exp'):
            f, ax = show.init_plot(title = dist)
        emp_dist = emp.loc[emp.distrib == dist]
        prep_dist = prep.loc[prep.distrib == dist]
        FP = np.unique(sim.FP) * 1000
        for ig, gr in enumerate(np.unique(prep_dist.group)):
            if gr == 'exp_group':
                clr_i = 1
            else:
                clr_i = 0
            # Empirical data
            ax.scatter(FP, emp_dist.loc[emp_dist.group == gr].RT,
                            color = clr_emp[clr_i], **plt_emp)
            # Variable-FP: fMTP
            show.show_var(FP, prep_dist[prep_dist.group == gr], 'fMTP', 
                          coef_fmtp, '-', clr_sim[clr_i])
            # Variable-FP: Memory hazard
            show.show_var(FP, prep_dist[prep_dist.group == gr], 'fMTPhz_inv', 
                          coef_hz, (0, (2., 2.)), clr_sim[clr_i])
        f.tight_layout()
        f.show()         

    return
    
    
def gauss_trillberg2000 (sorter):
    """
    Here we simulate Trillenberg et al.'s (2000) study. We focus on the
    Gaussian FP distribution. It becomes clear from this simulation that 
    all preparation models do not match the data from this study except for the
    classical hazard function.
    doi: 10.1016/S1388-2457(00)00274-1
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/trillenberg2000.csv')
    emp = emp.loc[emp.distrib == 'GAUSS']
    emp['distrib'] = 'gauss'
    FP = np.unique(emp.FP) # discrete FPs
                                 
    # Simulate experiment
    FP_ = np.array([1.3, 1.95, 2.6, None])
    sim, prep = sorter.run_dist([None], FP_, ['gauss'])
    
    # Plot settings
    FP = FP * 1000 
    xlim = [FP[0] - 225, FP[-1] + 225]
    xticks = FP
    ylim = [235, 345]
    yticks = np.arange(275, 345, 25)
    
    # Fit models and display preparation curves together with empirical data
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv'), ('sub_hz', 'sub_hz_inv'), 
              ('c_hz', 'c_hz_inv')]
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    fit = get_fit(emp, sim, prep, 'distrib') 
    fit.run_fit('Trillenberg (2000)', models, show_fit = show, clr_it = 3)
    
    return


def gauss_replicate (sorter):
    """
    Here we simulate our replication of Trillenberg et al.'s (2000) study. 
    We focus on our two conditions: clock-invisible and standard FP. We 
    display the preparation models predictions together with the empirical 
    data.
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/replication_trillenberg.csv')
    emp_standard = emp.loc[emp.experiment == 'Standard_FP']
    emp_standard['distrib'] = 'gauss'
    # Re-sort data
    emp = emp.loc[emp.experiment == 'Clock']
    emp['distrib'] = 'gauss'
    FP = np.unique(emp.FP) # discrete FPs
                                 
    # Simulate experiment
    FP_ = np.array([1.3, 1.95, 2.6, None])
    sim, prep = sorter.run_dist([None], FP_, ['gauss'])
    
    # Plot settings
    xlim = [FP[0] - 225, FP[-1] + 225]
    xticks = FP
    ylim = [315, 410]
    yticks = np.arange(325, 401, 25)
    
    # Fit models and display preparation curves together with empirical data    
    models = [['fMTP'], ('fMTPhz', 'fMTPhz_inv'), ('sub_hz', 'sub_hz_inv'), 
              ('c_hz', 'c_hz_inv')]
    
    # Fit data only on no-clock condition
    emp_noclock = emp[emp.clock_visible == 'Clock invisible']
    for mod in models:
        # Initilaise plot
        show = show_fit(xlim, ylim, xticks, yticks)
        f, ax = show.init_plot(mod)
        for sub_mod in mod:
            # Trillenberg's replication (Experiment 1 and 2)
            fit = get_fit(emp_noclock, sim, prep, 'gauss_rep') 
            coef, R = fit.run_fit('Replication: clock vs no-clock', sub_mod)
            print('Clock vs no-clock: ')
            show.show_gauss_rep(sub_mod, coef, emp_noclock, sim, prep, R, 
                                clr_it = 4)
            
            # Standard FP
            fit = get_fit(emp_standard, sim, prep, 'gauss_rep') 
            coef, R = fit.run_fit('Standard FP paradigm', sub_mod)
            print('Standard FP: ')
            show.show_gauss_rep(sub_mod, coef, emp_standard, sim, prep, R, 
                                clr_it = 5)
            print("")
            
        f.tight_layout()
        f.show()      
    
    return


def gauss_emp (): 


    # Empirical data
    emp = pd.read_csv('empirical_data/replication_trillenberg.csv')
    # Re-sort data
    emp.loc[emp.experiment == 'Standard_FP', 'clock_visible'] = 'standard FP'
    FP = np.unique(emp.FP) # discrete FP
    
    # Plot settings
    xlim = [FP[0] - 225, FP[-1] + 225]
    xticks = FP
    ylim = [300, 425]
    yticks = np.arange(325, 401, 25)

    # color settings
    clr = ['#4ecdc4', '#7570b3', '#ef476f'] 
    plt_emp_gauss = dict(s = 17.5, marker = 'o', facecolor = 'w') 
    # Display plot
    show = show_fit(xlim, ylim, xticks, yticks)
    f, ax = show.init_plot(title = 'Gauss Replication')
    plt_ = zip(np.unique(emp.clock_visible), [-50, 50, 0])
    for i, i_exp in enumerate(plt_):
        emp_ = emp[emp.clock_visible == i_exp[0]]
        ax.plot(emp_.FP - i_exp[1], emp_.RT, color = clr[i], 
                zorder = 0)
        se = emp_.RT - emp_.lower
        ax.errorbar(emp_.FP - i_exp[1], emp_.RT, yerr = se, ls = 'none', 
                    elinewidth = 1, ecolor = clr[i], zorder = 1)
        ax.scatter(emp_.FP - i_exp[1], emp_.RT, color = clr[i], 
                   **plt_emp_gauss)
    f.tight_layout()
    f.show()
    
    return


def sequential_steinborn2012 (sorter):
    """
    Here we simulate Steinborn and Langner's (2012) study. We focus on the 
    influence of the direct preceding and second preceding trials in 
    Experiment 2 of their study.
    doi: 10.1016/j.actpsy.2011.10.010
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/steinborn2012.csv')
    FP = np.unique(emp.FP)
    
    # Sequential effects
    exp = FPexp(FPs = FP, distribution = 'uni')  
    sim, prep = sorter.sim_seq2(FP, exp)
    
    # Plot settings
    FP = FP * 1000 
    xlim = [FP[0] - 500, FP[-1] + 500]
    xticks = FP
    ylim = [365, 435]
    yticks = np.arange(375, 426, 25)    
    
    # Fit models and display preparation curves together with empirical data
    models = [('fMTP', 'fMTPhz_inv')] 
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # 1st order sequential effects
    fit = get_fit(emp, sim, prep, 'seq') 
    fit.run_fit('Steinborn (2012)', models, show_fit = show)
    # 2nd order sequential effects
    emp.rename(columns = {'FPn_2' : 'factor2'}, inplace = True)
    # Plot fMTP and memory hazard in the same plot    
    models = ['fMTP', 'fMTPhz_inv'] 
    f1, ax1 = show.init_plot(title = 'FP1')
    f2, ax2 = show.init_plot(title = 'FP2')
    f3, ax3 = show.init_plot(title = 'FP3')
    figs = [f1, f2, f3]
    axes = [ax1, ax2, ax3]
    for mod in models: 
        # Fit models
        fit = get_fit(emp, sim, prep, 'seq2') 
        coef, R = fit.run_fit('', mod)
        # Show plots 
        show.show_seq2(mod, coef, emp, sim, prep, R, figs, axes)
    return


def niemi1979_extension (sorter):
    """
    Here we simulate Niemi's (1979) study. We focus on the distinction between 
    the bright and dimmed stimulus condition used in this study. 
    doi: 10.1016/0001-6918(79)90038-6
    """
    
    # Empirical data
    emp_dat = pd.read_csv('empirical_data/niemi1979.csv')
    FP = np.arange(0.5, 3.5, 0.5) # discrete FPs
    # Re-sort data
    emp_bright = emp_dat.loc[emp_dat.condition == 'bright']       
    emp_dimmed = emp_dat.loc[emp_dat.condition == 'dimmed'] 
    emp_bright['FP'] = FP
    emp_bright['distrib'] = 'uni'
    emp_dimmed['FP'] = FP
    emp_dimmed['distrib'] = 'uni'
    
    # Simulate experiment
    sim, prep = sorter.run_dist([None], FP, ['uni'])

    # Plot settings
    FP = FP * 1000 
    xlim = [0, FP[-1] + 500]
    xticks = np.arange(500, 3001, 1250)
    ylim = [230, 345]
    yticks = np.arange(250, 326, 25)
    
    # Fit models and display preparation curves together with empirical data
    models = ['fMTP', 'fMTPhz_inv']
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # Start fitting procedure looping through all the models
    show.print_results(title = 'Niemi (1979) extension', model = '', coef = '',
                       R = '')
    show.init_plot(title = 'Niemi stim condition')
    for mod in models: 
        # Fit models
        fit = get_fit(emp_bright, sim, prep, 'distrib') 
        coef_br, R_br = fit.run_fit('Niemi: bright (1979)', mod)
        fit = get_fit(emp_dimmed, sim, prep, 'distrib') 
        coef_dim, R_dim = fit.run_fit('Niemi (1979): dimmed', mod)
        # Show plots 
        print('Bright stimulus: ')
        show.show_dist(mod, coef_br, emp_bright, sim, prep, 
                       'distrib', R_br, clr_it = 0)
        print('Dimmed stimulus: ')
        show.show_dist(mod, coef_dim, emp_dimmed, sim, prep, 
                       'distrib', R_dim, clr_it = 1)
        print("")
    
    return


def los2001_extension (sorter):
    """
    Here we simulate Los and Heuvel's (2001) study. In 'sim2' and we collapsed 
    over both reward conditions. Here, we focus on the distinction of the 
    reward and non-reward condition. 
    doi: 10.1037/0096-1523.27.2.370
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/losheuvel_seq.csv')
    FP = np.unique(emp.FP) # discrete FPs
    
    # Simulate experiment
    sim, prep = sorter.sim_seq(FP)
    
    # Plot settings
    FP = FP * 1000 
    xlim = [0, FP[-1] + 500]
    xticks = np.arange(FP[0], xlim[-1], 500)
    ylim = [215, 310]
    yticks = np.arange(225, 301, 25)    
    
    # Fit models and display preparation curves together with empirical data
    models = [('fMTP', 'fMTPhz_inv')] 
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # No-reward: sequential effects
    fit = get_fit(emp, sim, prep, 'seq') 
    fit.run_fit('Los and van den Heuvel (2001): No reward', 
                models, show_fit = show)
    # Reward: sequential effects
    emp.RT = emp.RTreward.values
    fit = get_fit(emp, sim, prep, 'seq') 
    fit.run_fit('Los and van den Heuvel (2001): Reward', 
                models, show_fit = show)   
    
    return


def gonogo_los2013 (sorter):
    """
    Here we simulate Los's (2013) study that studied the role of inhibition
    in the sequential effect. We focus on the data of Experiment 2. 
    doi: 10.1016/j.cognition.2013.07.013
    """
    
    # Empirical data
    emp = pd.read_csv('empirical_data/los2013.csv')
    FP = np.unique(emp.FP) # discrete FPs
    emp = emp.groupby(['gngn_1', 'FPn_1', 'FP']).mean().reset_index()
    
    # Simulate experiment
    # We simulate more N of trials as there are more conditions for FPn-1 to 
    # select for (besides FP length also condition: go / no-go / relax)
    exp = FPgonogo(distribution = tuple([1,1]), FPs = FP, relax = 1, 
                   tr_per_block = 3000)
    sim, prep = sorter.sim_seq2(FP, exp)
 
    # Plot settings
    FP = FP * 1000 
    xlim = [0, FP[-1] + 300]
    xticks = FP
    ylim = [280, 350]
    yticks = np.arange(290, 350, 25)

    # Fit models and display preparation curves together with empirical data
    models = [('fMTP', 'fMTPhz_inv')] 
    # Initialise plot
    show = show_fit(xlim, ylim, xticks, yticks)
    # Memory models: distribution: effects
    emp.rename(columns = {'gngn_1' : 'factor2'}, inplace = True)
    # Plot fMTP and memory hazard in the same plot    
    models = ['fMTP', 'fMTPhz_inv'] 
    f1, ax1 = show.init_plot(title = 'FP1')
    f2, ax2 = show.init_plot(title = 'FP2')
    f3, ax3 = show.init_plot(title = 'FP3')
    figs = [f1, f2, f3]
    axes = [ax1, ax2, ax3]
    for mod in models: 
        # Fit models
        fit = get_fit(emp, sim, prep, 'seq2') 
        coef, R = fit.run_fit('', mod)
        # Show plots 
        show.show_seq2(mod, coef, emp, sim, prep, R, figs, axes)

    return 


if __name__ == '__main__':
    
    plt.close("all")
    
    # Parameterization fMTP
    k = 4 # temporal precision
    r = -2.81 # rate of forgettting
    c = 0.0001 # memory persistence
    fmtp = fMTP(r, c, k)
    
    # Define fMTPhz
    fMTPhz_lin = fMTPhz(-3.0, 3e-4,  k)
    fMTPhz_inv = fMTPhz(-2.8, 1e-4, k)
    

    # Initialize the fitting procedure
    sorter = sort_fit(fmtp, fMTPhz_lin, fMTPhz_inv)
    
    # Run simulations
    
    # Constant and variable FP paradigm
    convarFP_niemi1979(sorter) # Niemi 1979
    convarFP_seq_los2001(sorter) # Los and v/d Heuvel 2001
    
    # Distribution and transfer effects 
    distrib_transfer_los2017(sorter) # distribution- and transfer effects
    
    # Gaussian distribution
    gauss_trillberg2000(sorter) # original data
    gauss_replicate(sorter) # replication: clock- vs no-clock
    gauss_emp() # replication: only empirical data
    
    # Sequential effects
    sequential_steinborn2012(sorter) # sequential effects
    
    # Extensions section
    niemi1979_extension(sorter) # bright/dimmed stimulus intensity
    los2001_extension(sorter) # high/low reward
    gonogo_los2013(sorter) # go/nogo/relax trials
    
   