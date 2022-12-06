"""
---------------------------------------------------------------------
fMTP: A unifying framework of temporal preparation across time scales
---------------------------------------------------------------------

In this file we determine the parameters of the power-law forgetting curve  
using least-squares error minimization via L-BFGS-G. We do this separtely for:
fMTP, linear fMTPhz, and inverse fMTPhz. 

We base this fitting procedure on data fitting the longest- and shortest time 
scale of the FP-effects under study: the sequential effect 
(Los & Van Den Heuvel, 2001; Steinborn & Langner, 2012) and the the transfer 
effect (Los et al. 2017). 

Be aware that running this optimization procedure 
may take quite some time to converge. It took ~5 minutes per fitting procedure
for each model on a laptop with an intel core i7 (1.9GHz, QC) 

Josh Manu Salet
Email: saletjm@gmail.com
"""


## Read in libaries & classes

# Import packages needed
import numpy as np 
import pandas as pd

# Import for using optimizer
from scipy import optimize as op

# Import classes
from fmtp import fMTP, FPexp, FPtransfer
from hazard import fMTPhz


## Helper functions

def temp2RT(value, a, b):
    """
    Temporal preparation (AI ratio) is simply linearly mapped to RTs:
    RT = y * (1 / AI) + b
    """
    
    return (a * value) + b


def sum_squares(parm, prep_mod, dat_emp):    
    """
    Finding the linear constants (a and b) is done by finding the sum of
    squared error (SSE). This serves as the 'obejective' function input for the
    L-BFGS-G algorithm for bound constrained minimization.
        parm: are the linear constants (a and b)
        prep_mod: the AI ratio calculated
        dat_emp: the emperical data RT to map to
    """
    
    RTmod = temp2RT(prep_mod, *parm)
    SSE = sum(pow(dat_emp - RTmod, 2))
    return SSE


def fit_forget_curve(p0, fit_modus, mapping_RT):
    """
    This function fits the parameters of the forgetting curve (r and c)
    """          
    
    # First define the time represenation (these parameters remain fixed)
    k = 4 
            
    # The constant of the memory curve fit (change for every iteration)
    r,c = p0                                                    
    
    # Define fMTP
    if fit_modus == 'fmtp':
        model = fMTP(r, c, k)
    elif fit_modus == 'hazard':
        model = fMTPhz(r, c, k)

    # Return sum of squared errors
    sse = 0
    # Sequential n-1 (Los & Van Den Heuvel, 2001)
    sse += seq1(model, mapping_RT)
    # Sequential n-2 (Steinborn & langner, 2012)
    sse += seq2(model, mapping_RT)
    # Transfer effect (Los et al. 2017)
    sse += transfer(model, mapping_RT)          
    
    print ("SSE:",  "%.2f" % sse, "r & c:", "%.4f" % p0[0],  "%.4f" % p0[1])
    
    return(sse)                                                    


## Sequential effects n-1

def seq1(model, mappingRT):
    """
    Here we simulate Los and Heuvel's (2001) study. We focus on those blocks 
    in which participants were not informed about the nature of the upcoming
    FP (neutral-cue condition) and average over different payout regimes 
    (reward vs. non-reward). Here we focus on the sequential effect.
    doi: 10.1037/0096-1523.27.2.370
    """
    
    # Read csv file containing grand averages empirical RT data
    emp_dat = pd.read_csv('empirical_data/losheuvel_seq.csv')
    # Discrete FPs used in constant and variable FP-paradigm
    FP = np.arange(0.5,2.,0.5)
    # Call Experiment object to set up experiment
    exp = FPexp(FPs = FP, distribution = "uni", tr_per_block = 500) 
    sim, prep = exp.run_exp(model) #run experiment     
    if mappingRT == 'inverse':
        sim.prep = 1. / sim.prep #hazard at discrete points

    #Sorting
    # Add n-1 column for effect of previous trial (n-1) on current trial (n)
    sim['FPn_1'] = np.r_[np.nan, sim.FP.iloc[:-1].values]   
                
    # Get rid of first row, no n-1
    sim = sim.iloc[1:,:]                                                        

    #Sequential effects
    # Sort by FPn-1 and FP of fMTP's predictions
    Dsim = sim.groupby(['FPn_1', 'FP']).mean()                               
    # Sort empirical data
    Demp = emp_dat.groupby(['FPn_1', 'FP']).mean()                           
    # Collapse over reward condition
    Demp = (Demp.RT + Demp.RTreward) / 2.                                       

    # Initial parameters to start optimizing with
    p0 = [4., 375.]
    # Objective function to be minimized (= Sum of Squared Error)
    fit = op.minimize(sum_squares, p0,                                          
                         # fMTP's preparation predictions and empirical RT data
                         args = (Dsim.prep.values, Demp.values),                
                         # Optimization method
                         method = 'L-BFGS-B')                                   
    #Return sum of squared errors
    return(fit.fun)                                                             


## Sequential effects n-2

def seq2(model, mappingRT):
    """
    Here we simulate Steinborn and Langner's (2001) study. We focus on the 
    influence of the direct preceding and second preceding trials in 
    Experiment 2 of their study.
    doi: 10.1016/j.actpsy.2011.10.010
    """
    
    # Read csv file containing grand averages empirical RT data
    emp_dat = pd.read_csv('empirical_data/steinborn2012.csv')
    # Discrete FPs used in constant and variable FP-paradigm
    FP = np.array([0.8,1.6,2.4])
    exp = FPexp(FPs = FP, distribution = "uni", tr_per_block = 500)         
    sim, prep = exp.run_exp(model) # Run experiment
    if mappingRT == 'inverse':
        sim.prep = 1. / sim.prep #hazard at discrete points      

    # Add n-1 column for effect of previous trial (n-1) on current trial (n)
    sim['FPn_1'] = np.r_[ np.nan, sim.FP.iloc[:-1].values]                       
    # Add n-2 column for effect of previous trial (n-2) on current trial (n)
    sim['FPn_2'] = np.r_[ np.repeat(np.nan, 2), sim.FP.iloc[:-2].values]         

    # Get rid of first row, no n-2
    sim = sim.iloc[2:,:]                                                        
    # now aggregate; group_by FP, and get the mean
    D_sim = sim.groupby(['FPn_2', 'FPn_1', 'FP']).mean()
    D_emp = emp_dat.groupby(['FPn_2', 'FPn_1', 'FP']).mean()

    # Initial parameters to start optimizing with
    p0 = [4., 375.]
    # Objective function to be minimized (= Sum of Squared Error)
    fit = op.minimize(sum_squares, p0,                                          
                      # fMTP's preparation predictions and empirical RT data
                      args = (D_sim.prep, D_emp.RT),                            
                      # Optimization method
                      method = 'L-BFGS-B')                                      
    # Return sum of squared errors
    return(fit.fun)                                                             


## Transfer effects

def transfer(model, mappingRT):
    """
    Here we focus on the transfer effects instead of the 
    distribution effects. Again, we focus on the data of Experiment 1. 
    doi: 10.1037/xhp0000279
    """
    
    # Read csv file containing grand averages empirical RT data
    emp_dat= pd.read_csv('empirical_data/transfer.csv')
    # Discrete FPs used in constant and variable FP-paradigm
    FP = np.arange(0.4, 1.8, 0.4)
    
    transfer = FPtransfer(FP)                                                   
    sim = transfer.run_exp(model) #Run experiment
    if mappingRT == 'inverse':
        sim.prep = 1. / sim.prep #hazard at discrete points
   
    # sort output simulation in line with empirical data
    sim.loc[sim['block_index'] == 3, 'block_index'] = 2
    sim.loc[sim['block_index'] == 4, 'block_index'] = 3
    sim.loc[sim['block_index'] == 5, 'block_index'] = 3   
  
    sim = sim.groupby(['block_index', 'group','FP']).mean()           

    # Empirical data
    emp_dat.loc[emp_dat['block_index'] == 3, 'block_index'] = 2
    emp_dat.loc[emp_dat['block_index'] == 4, 'block_index'] = 3
    emp_dat.loc[emp_dat['block_index'] == 5, 'block_index'] = 3
    
    # aggregate group by block, group and FP 
    emp = emp_dat.groupby(
        ["block_index", "group_name", "FP"]).mean().reset_index()                                                       
    
    # Initial parameters to start optimizing with
    p0 = [4., 375.]
    # Objective function to be minimized (= Sum of Squared Error)
    fit = op.minimize(sum_squares, p0,                                          
                         # fMTP's preparation predictions and empirical RT data
                         args = (sim.prep.values, emp.RT.values),               
                         # Optimization method
                         method = 'L-BFGS-B')                                   
    # Return sum of squared errors
    return(fit.fun)                                                           
   


if __name__ == '__main__':

    
    ## Fit fMTP forgetting parameters
    
    
    fit_model = 'fmtp' 
    mapping_RT = 'linear'
    p0 = [-3, 1e-4]
    bounds_opm = ((-5., 0.), (0., 1e-3))

    # Minimization procedure
    parms_mem = op.minimize(fit_forget_curve, p0, 
                            args = (fit_model, mapping_RT),
                            method = 'L-BFGS-B', 
                            bounds = bounds_opm, 
                            options = {'disp' : True,
                                       'ftol': 1e-4})
    print ("")
    print ("------------------------")
    print ("Fitted Memory Parameters:")
    print ("r:", parms_mem.x[0], " and c:", parms_mem.x[1])
    print ("------------------------")
    print ("" )
    
    
    ## Fit linear fMTPhz forgetting parameters
    
    
    fit_model = 'hazard' 
    mapping_RT = 'linear'
    p0 =  [-3, 1e-4]
    bounds_opm = ((-5., 0.), (0., 1e-3))
    
    # Minimization procedure
    parms_mem = op.minimize(fit_forget_curve, p0, 
                            args = (fit_model, mapping_RT),
                            method = 'L-BFGS-B', 
                            bounds = (bounds_opm), 
                            options = {'disp' : True})
    print ("")
    print ("------------------------")
    print ("Fitted Memory Parameters:")
    print ("r:", parms_mem.x[0], " and c:", parms_mem.x[1])
    print ("------------------------")
    print ("" )
    
    
    
    ## Fit inverse fMTPhz forgetting parameters
    
    
    fit_model = 'hazard' 
    mapping_RT = 'inverse'
    p0 = [-3., 1e-5]
    # Minimization procedure
    parms_mem = op.minimize(fit_forget_curve, p0, 
                            args = (fit_model, mapping_RT),
                            method = 'L-BFGS-B', 
                            bounds = (bounds_opm), 
                            options = {'disp' : True})
    print ("")
    print ("------------------------")
    print ("Fitted Memory Parameters:")
    print ("r:", parms_mem.x[0], " and c:", parms_mem.x[1])
    print ("------------------------")
    print ("" )