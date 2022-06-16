# -*- coding: utf-8 -*-
"""
Pulse Class

Created on Thu Nov 11 15:06:43 2021

@author: tgule
"""
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns

from pyts.approximation import SymbolicFourierApproximation
from pyts.datasets import load_gunpoint
from pyts.bag_of_words import BagOfWords
from pyts.bag_of_words import WordExtractor
from pyts.classification import BOSSVS
from pyts.transformation import BOSS

def get_metadata(s3_object):
    '''
    takes an s3 file object for a csv file with the standard pulse file naming convention
    outputs - dictionary with interval (20 or 400 ns), result (run/fault), timestamp
    test update
    '''
    # split filepath with slashes to get filename
    num_folders = len(s3_object.key.split('/')) - 1
    file_name = s3_object.key.split('/')[num_folders]
    
    # use string parsing to get interval, result, time
    interval = file_name.split('_')[1]
    result = file_name.split('_')[2]
    str_time = file_name.split('_')[3].replace('.csv', '')
    #time = datetime.datetime.strptime(str_time, '%m.%d.%Y-%I.%M.%S%p')
    return {'interval' : interval, 'result' : result, 'str_time' : str_time}


def get_median_pulse(pulses, normal=False):
    '''
    Takes the median of all pulses per component of all runs. 
    These are to be used as a baseline to determine if a run is normal or abnormal.
    INPUT: dictionary of pulse objects
    OUTPUT: median of all first pulses per component  of all runs.
    '''
    components = pulses[list(pulses)[0]].data.columns # get column list
    
    median_pulse = pd.DataFrame()
    
    for comp in components: # iterate through components
        
        temp = pd.DataFrame(columns = []) # set up temp df to store all values of component
        
        for k, v in pulses.items(): # iterate through pulses
            
            if normal:
                temp[k + str(1)] = v.pulse1norm[comp].values # pulse 1
                temp[k + str(2)] = v.pulse2norm[comp].values # pulse 2
            
            else:
                temp[k + str(1)] = v.pulse1[comp].values # pulse 1
                temp[k + str(2)] = v.pulse2[comp].values # pulse 2

        median_pulse[comp] = temp.median(axis=1) # medians
        
    return median_pulse

def distance_from_median(pulses, median_pulse):
    '''
    Measures the cumulative Euclidean distance between each pulse and the normal baseline. 
    We can use this to determine a threshold for faults, and filter out adnormal runs.
    INPUT: dictionary of pulse objects
    OUTPUT: A dictionary of dataframes of cumsum distance with components as columns
    '''
    components = pulses[list(pulses)[0]].data.columns # list of columns
    
    pulse_cumsum = {} # set up empty dict to store cumulative L2 distance dfs
    
    for k, v in pulses.items(): # iterate through pulses
        
        data = v.pulse1 # look at pulse 1 (predictor)
        temp = (data - median_pulse)**2 # calculate L2 distance
        
        pulse_cumsum[k] = temp.cumsum() # take cumulative sum and add to dictionary
        
    return pulse_cumsum

def cumulative_distance_from_median(pulses, median_pulse):
    '''
    Measures the cumulative Euclidean distance between each pulse and the normal baseline at PULSE END - ONLY ONE VALUE PER COLUMN/PULSE. 
    We can use this to determine a threshold for faults, and filter out adnormal runs.
    INPUT: dictionary of pulse objects, a representation of the median pulse (np array)
    OUTPUT: A dictionary of dataframes of cumsum distance with components as columns
    '''
    
    all_cumsums = pd.DataFrame() # set up df to store cumulative L2 values
    
    for k, v in pulses.items(): # iterate through pulses
        if k != '6.22.2020-4.34.23PM': # exclude pulse with null values [WILL BE MADE MORE DYNAMIC LATER]
            
            data = v.pulse1 # look at pulse 1 (predictor)
            temp = (data - median_pulse)**2 # calculate L2 distance

            pulse_cumsum = temp.cumsum().iloc[-1] # get final cumulative L2 value
            pulse_cumsum['result'] = v.result # add result to df (response variable)
            pulse_cumsum['time'] = k # add time to df for identification

            all_cumsums = all_cumsums.append(pulse_cumsum)
        
    return all_cumsums.reset_index(drop=True)

def get_normals(all_pulses, runs=True):
    '''
    This function takes a dictionary of pulses and returns the max value of each component, to be used for normalization
    INPUT - 'all_pulses' dictionary of pulse object, 'runs' boolean, whether to get max of only runs or of all pulses
    OUPUT - pandas series with max value for each component
    '''
    components = all_pulses[list(all_pulses)[0]].data.columns # get column list

    component_maxs = pd.DataFrame() # set up df to store max's for each component in each pulse
    
    if runs==True: # reduce to only runs
        all_pulses = {k : v for k, v in all_pulses.items() if v.result == 'Run'}
    
    # create df with max from each individual pulse
    for k, v in all_pulses.items():
        pulse1_maxs = pd.DataFrame(v.pulse1.abs().max()).transpose()

        component_maxs = pd.concat([component_maxs, pulse1_maxs]).reset_index(drop=True)
    
    # calculate max over all pulses
    return component_maxs.max()

def get_L2(pulse : pd.DataFrame, comparison : pd.DataFrame, cols = None):
    '''
    INPUTS - pulse - singular pulse (e.g. pulse1 from a certain runtime), comparison - prototype pulse, cols - components to compare (default is all)
    OUTPUTS - either a 
    '''
    # use all columns if none specified
    if cols:
        pass
    else:
        cols = pulse.columns
    
    difference = np.matrix(pulse[cols] - comparison[cols]) # subtract actual from prototype
    
    norm = np.linalg.norm(difference, ord='fro') # calculate Frobenius distance
    
    return norm

def get_all_L2s(pulses : dict, comparison : pd.DataFrame, cols = None, normalized=True, combined=False):
    '''
    GOAL - This function is intended to calculate L2 norms for the difference between actual and expected for some or all components across a set of pulses.
    This can be done as separate values for all components or as a combined norm
    INPUTS - pulses : dictionary of pulse objects, comparison : dataframe containing comparison pulse, cols : list of components to use, normal : whether to compare
    to the normalized pulse, combined : whether to output an overall L2 norm or report separately for each component
    OUTPUTS - X : feature tensor (2D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X'''
    
    # set up df
    L2_df = pd.DataFrame()
    y_fault = []
    y_type = []
    
    # loop through pulses
    for k, v in pulses.items():
    
        # use normalized pulse if specified, otherwise raw pulse
        if normalized:
            obs_pulses = [v.pulse1norm, v.pulse2norm]

        else:
            obs_pulses = [v.pulse1, v.pulse2]
            
        if v.result == 'Fault':
            obs_pulses = [obs_pulses[0]]
            y_fault.append(1)
            y_type.append(v.fault_type)
        
        else:
            _ = [y_fault.append(0) for x in obs_pulses]
            _ = [y_type.append('Run') for x in obs_pulses]
        
        # run get_L2 helper function to calculate L2 norms - either combined or separate
        if combined:
            this_L2 = [pd.DataFrame({'combined_L2' : get_L2(obs_pulses[x], comparison, cols=cols)}, index = [k + '-pulse' + str(x+1)]) for x in range(len(obs_pulses))]
        else:
            this_L2 = [pd.DataFrame({col : get_L2(obs_pulses[x], comparison, cols=col) for col in cols}, index = [k + '-pulse' + str(x+1)]) for x in range(len(obs_pulses))]
        
        # concatenate
        for x in range(len(this_L2)):
            L2_df = pd.concat([L2_df, this_L2[x]])
        
    X = np.array(L2_df)
    y_fault = np.array(y_fault)
    y_type = np.array(y_type)
    indices = (L2_df.index, L2_df.columns)
        
        
    return X, y_fault, y_type, indices


def get_component(pulses, component, reshape = None, normalized=True):
    '''
    returns a df of all pulse 1 data (and pulse 2 for runs) for all observations of a particular component, as well as the result
    INPUTS - dictionary of pulses, name of component to be used, whether to downsample, and whether to use normalized values
    OUTPUTS - df of observation values for one component, columns are timestamps and observations are rows, array of fault booleans as well as fault type for each pulse
    '''
    
    component_dict = {}

    y_fault = []
    y_type = []
    
    for k, v in pulses.items():

        if normalized:
            obs_pulses = [v.pulse1norm, v.pulse2norm]

        else:
            obs_pulses = [v.pulse1, v.pulse2]

        if v.result == 'Fault':
            obs_pulses = [obs_pulses[0]]
            y_fault.append(1)
            y_type.append(v.fault_type)

        else:
            _ = [y_fault.append(0) for x in obs_pulses]
            _ = [y_type.append('Run') for x in obs_pulses]

        for idx in range(len(obs_pulses)):
            component_dict[k + '-pulse' + str(idx+1)] = obs_pulses[idx][component]

    df = pd.DataFrame(component_dict)

    if reshape:
        df = df.loc[::reshape]

    df = df.T

    return df, np.array(y_fault), np.array(y_type)

def get_bow_results(pulses, components, reshape=None, normalized=True, window_size=10, word_size=3, n_bins=2):
    '''
    GOAL - get sentence version of waveform for all observations of a set of components
    IMPUTS -  dictionary of pulses, name of component to be used, downsampling rate, window size, word size, number of available letters for vocab
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X
    '''
    
    # get df with values of each pulse for one component
    all_comps = np.array([np.array(get_component(pulses, component, reshape=reshape, normalized=normalized)[0]) for component in components])
    
    # configure bag-of-words model
    bow = BagOfWords(window_size=window_size, word_size=word_size,
                     window_step=window_size, numerosity_reduction=False, n_bins=n_bins)
    
    # transform to BOW and reshape
    bow_results = np.array([bow.transform(all_comps[idx]) for idx in range(all_comps.shape[0])]).T
    nobs, ncols = bow_results.shape
    X = np.array([np.array([bow_results[x][y].split() for y in range(ncols)]) for x in range(nobs)])
    

    df, y_fault, y_type = get_component(pulses, components[0], reshape = 10, normalized=True)
    
    indices = (df.index, components, range(0, X.shape[2]))
    
    #return component_df
    return X, y_fault, y_type, indices

def get_boss_results(pulses, components, reshape=None, normalized=True, window_size=10, word_size=3, n_bins=2):
    '''
    GOAL - get histogram values for sentence version of waveform for all observations of a set of components
    IMPUTS -  dictionary of pulses, name of component to be used, downsampling rate, window size, word size, number of available letters for vocab
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X
    '''
    
    # get df with values of each pulse for one component
    all_comps = np.array([np.array(get_component(pulses, component, reshape=reshape, normalized=normalized)[0]) for component in components])
    
    # configure BOSS model
    boss = BOSS(window_size=window_size, word_size=word_size,
                     numerosity_reduction=False, n_bins=n_bins, sparse=False)
    
    # transform to BOSS and reshape
    boss_results = np.array([boss.fit_transform(all_comps[idx]) for idx in range(all_comps.shape[0])])

    dim2, dim1, dim3 = boss_results.shape

    X = np.empty((0, dim2, dim3))
    
    for idx in range(dim1):
        X = np.concatenate((X, boss_results[:, idx, :].reshape(1, dim2, dim3)), axis=0)
    
    # get index values
    df, y_fault, y_type = get_component(pulses, components[0], reshape = 10, normalized=True)
    
    indices = (df.index, components, range(0, X.shape[2]))
    
    #return component_df
    return X, y_fault, y_type, indices

def sentence_difference_hist(pulses, component, reshape=None, normalized=True, window_size=10, word_size=3, n_bins=2):
    
    '''
    GOAL - get histogram of sentence differences from mode between runs and faults in a particular component
    IMPUTS -  pulses : dictionary of pulses, component : name of component to be used, reshape : downsampling rate, 
    window size, word size, n_bins : number of available letters for vocab
    OUTPUTS - histogram of sentence differences from mode between runs and faults in a particular component
    '''
    # get bow results
    X, y_fault, y_type, indices = get_bow_results(pulses, component, reshape=reshape, normalized=normalized, window_size=window_size, word_size=word_size, n_bins=n_bins)
    
    # find pattern
    pattern = stats.mode(X)[0].reshape(X.shape[1])
    
    # get differences
    differences = X.shape[1] - (X == pattern).sum(axis=1)
    
    # graph
    color = y_fault
    plt.hist(differences[color], alpha=0.5, label='Fault')
    plt.hist(differences[~color], alpha=0.5, label='Run')
    plt.legend()
    plt.title(f'Number of Sentence Differences by Pattern - {component} with sentence size {X.shape[1]}')
    plt.show()
    
def get_fourier_values(pulse, cols, sampling_rate, reshape=None):
    '''
    GOAL - perform a Fourier transformation for a single pulse (e.g. pulse1norm attribute of one observation) (any number of components)
    INPUTS - pulse : a Pulse.Pulse type file, cols : list of cols for which to perform Fourier, sampling_rate : int ??, 
    normalized : boolean on whether to use normalized value of pulse, reshape : downsampling rate
    OUTPUT - dataframe of frequencies and resultant values by component (to be used in further function get_all_fourier_values)
    '''

    pulse_val = pulse[cols].loc[::reshape]
    
    # set size
    size = len(pulse_val)
    
    # get array frequencies to evaluate
    f_values = np.linspace(0.0, 1.0/(2.0*sampling_rate), size//2)
    
    # calculate and transform fourier values
    fft_values = {col : fft(pulse_val[col].values) for col in cols}
    fft_values = {col : 2.0/size * np.abs(vals[0:size//2]) for col, vals in fft_values.items()}
    
    output = pd.DataFrame({'frequency': f_values} | fft_values)
    
    return output 

def get_all_fourier_values(pulses : dict, cols = None, sampling_rate = 0.25, normalized=True, reshape=None, len_pulse=6100):
    '''
    GOAL - This function is intended to calculate Fourier transformations for some or all components across a set of pulses.
    INPUTS - pulses : dictionary of pulse objects, cols : list of components to use, normal : whether to use the normalized pulse, 
    sampling_rate : int?, reshape : int, downsampling rate, len_pulses : int, length of pulse
    OUTPUTS - X : feature tensor (2D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X'''
    
    # set up outputs
    y_fault = []
    y_type = []
    
    dim_2 = len(cols)
    
    if reshape:
        dim_3 = int(len_pulse/(2*reshape)) 
    else:
        dim_3 = int(len_pulse/2)
    
    X = np.empty((0, dim_2, dim_3))
    
    pulse_index = []
    
    # get frequencies
    if normalized:
        frequencies = get_fourier_values(pulses[list(pulses.keys())[0]].pulse1norm, cols, sampling_rate, reshape=reshape).frequency
        
    else:
        frequencies = get_fourier_values(pulses[list(pulses.keys())[0]].pulse1norm, cols, sampling_rate, reshape=reshape).frequency
    
    # loop through pulses
    for k, v in pulses.items():
    
        # use normalized pulse if specified, otherwise raw pulse
        if normalized:
            obs_pulses = [v.pulse1norm, v.pulse2norm]

        else:
            obs_pulses = [v.pulse1, v.pulse2]
            
        if v.result == 'Fault':
            obs_pulses = [obs_pulses[0]]
            y_fault.append(1)
            y_type.append(v.fault_type)
        
        else:
            _ = [y_fault.append(0) for x in obs_pulses]
            _ = [y_type.append('Run') for x in obs_pulses]
        
        # run get_L2 helper function to calculate L2 norms - either combined or separate
        
        this_X = np.array([np.array(get_fourier_values(obs_pulses[x], cols=cols, sampling_rate=sampling_rate, reshape=reshape).drop(columns=['frequency'])).T for x in range(len(obs_pulses))])
        X = np.concatenate((X, this_X), axis=0)
        
        _ = [pulse_index.append(k + '-pulse' + str(x+1)) for x in range(len(obs_pulses))]
        

    y_fault = np.array(y_fault)
    y_type = np.array(y_type)
    indices = (pulse_index, pd.Index(cols), list(pulses.values())[0].pulse1.loc[::reshape].index)
        
        
    return X, y_fault, y_type, indices, frequencies

def get_all_pulses(pulses : dict, cols = None, normalized=True, reshape=None):
    '''
    GOAL - This function is intended to pull an array of the raw pulse data for a set of pulses 
    (pulses 1 and 2 for runs, pulse 1 for faults) for a subset of components
    INPUTS - pulses : dictionary of pulse objects, cols : list of components to use, normal : whether to compare
    to the normalized pulse
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X'''
    
    # set up result arrays
    y_fault = []
    y_type = []
    pulse_index = []
    
    ex_pulse = pulses[list(pulses.keys())[0]].pulse1
    
    if not cols:
        cols = ex_pulse.columns
        
    X = np.empty((0, len(cols), ex_pulse.loc[::reshape].shape[0]))
    
    # loop through pulses
    for k, v in pulses.items():
    
        # use normalized pulse if specified, otherwise raw pulse
        if normalized:
            obs_pulses = [v.pulse1norm, v.pulse2norm]

        else:
            obs_pulses = [v.pulse1, v.pulse2]
       
        # only use first pulse if fault
        if v.result == 'Fault':
            obs_pulses = [obs_pulses[0]]
            y_fault.append(1)
            y_type.append(v.fault_type)
        
        else:
            _ = [y_fault.append(0) for x in obs_pulses]
            _ = [y_type.append('Run') for x in obs_pulses]
        
        # get timestamp
        _ = [pulse_index.append(k + '-pulse' + str(x+1)) for x in range(len(obs_pulses))]
        
        # concatenate all pulses to overall array
        this_X = np.array([pulse[cols].loc[::reshape].T for pulse in obs_pulses])
        
        
        X = np.concatenate((X, this_X), axis=0)
        
    
    y_fault = np.array(y_fault)
    y_type = np.array(y_type)
    indices = (pulse_index, pd.Index(cols), list(pulses.values())[0].pulse1.loc[::reshape].index)
        
        
    return X, y_fault, y_type, indices