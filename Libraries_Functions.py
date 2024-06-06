## Libraries and reading gds files
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.ticker import PercentFormatter, MultipleLocator, FuncFormatter
import math
from skimage.measure import ransac
import uncertainties
from uncertainties import unumpy as unp
from uncertainties import ufloat
from skimage.measure import EllipseModel
import ipywidgets as widgets
import seaborn as sns
import mplcursors as mpc
import mpld3
#from sklearn.preprocessing import StandardScaler
from shapely.geometry import Polygon
from uncertainties import ufloat_fromstr
from matplotlib.transforms import BlendedGenericTransform
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

""" Define functions that need using in the code
"""

# Function to calculate the centroid and area of the polygon for data of cycles
def calculate_centroid(x_values, y_values):
    A = abs( 0.5 * np.abs(np.dot(x_values, np.roll(y_values, 1)) - np.dot(y_values, np.roll(x_values, 1))))
    C_x = np.sum((x_values + np.roll(x_values, 1)) * (x_values * np.roll(y_values, 1) - np.roll(x_values, 1) * y_values)) / (6 * A)
    C_y = np.sum((y_values + np.roll(y_values, 1)) * (x_values * np.roll(y_values, 1) - np.roll(x_values, 1) * y_values)) / (6 * A)
    return C_x, C_y, A


## define the function for area and volume of the samples. 
##
def A(D):
    return (np.pi * (D/1000) ** 2) / 4
def V(A, H):
    return (A*(H/1000))

## define the function for Relative Uncertainty from the values with uncertainty (ufloat). 
##
def ur(data):
    return (unp.std_devs(data)/unp.nominal_values(data)) * 100


## Calculates the uncertainty mean and uncertainty for a list of ufloats.
##
def umean(data):
    nominal_values = unp.nominal_values(data)
    std_deviations = unp.std_devs(data)
    mean_value = np.mean(nominal_values)
    uncertainty = (np.sqrt(np.sum(np.square(std_deviations)))/len(data))
    return (ufloat(mean_value, uncertainty))

## Calculates the weighted mean and uncertainty for a list of ufloats.
##
def wmean(data):
    weights = [1 / d.s**2 for d in data]  # Weights based on inverse variance
    total_weight = sum(weights)
    weighted_mean = sum(w * d.n for w, d in zip(weights, data)) / total_weight
    uncertainty = 1 / np.sqrt(total_weight)
    return (ufloat(weighted_mean, uncertainty))


#### Function for caculation the parameteres in each stage of loading

def calculate_stage_info_base(df, cycles_data, name, P0q0_stage_nr, qm_stage_nr, qf, Su):
    # Perform the initial aggregation
    stage_info = df.groupby('Stage Number').agg({
        'Time since start of test (s)': ('min', 'max'),
        'index': ('first', 'last', 'count'),
        'Cycle nr': 'nunique',
        'Cycle time': 'mean'
    }).reset_index()

    # Rename columns
    stage_info.columns = ['Stage Number', 'Start time', 'End time', 'Start index', 'End index', 'Rows nr', 'Cycles nr', 'Cycle period (s)']
    
    # Adding more columns:
    stage_info['Frequency(Hz)'] = 1 / stage_info['Cycle period (s)']
    stage_info['Sample type'] = name
    stage_info['Test duration(d)'] = (stage_info['End time'] - stage_info['Start time']) / (3600 * 24)
        
    stage_info['Cyclic loading duration(d)'] = stage_info.apply(lambda row: (row['End time'] - row['Start time']) / (3600 * 24) if row['Cycles nr'] != 0 else 0, axis=1)
   
    ## Calculate 'p_0', 'q_0', 'qm' at 'start_index' row of different stage nr due to there is presharing 
    stage_info['q_0'] = stage_info.apply(lambda row: df.loc[row['Start index'], 'Deviator Stress & Uncertainty (kPa)'] if row['Stage Number'] == P0q0_stage_nr else 0, axis=1)
    stage_info['p_0'] = stage_info.apply(lambda row: df.loc[row['Start index'], 'Eff. Stress & Uncertainty (kPa)'] if row['Stage Number'] == P0q0_stage_nr else 0, axis=1)
    stage_info['q_m'] = stage_info.apply(lambda row: df.loc[row['Start index'], 'Deviator Stress & Uncertainty (kPa)'] if row['Stage Number'] == qm_stage_nr else 0, axis=1)

    mask = (stage_info['Cycles nr'] != 0)
    mask2 = np.where((stage_info['q_0'] != 0) | (stage_info['Cycles nr'] != 0) , True, False)
     
    stage_info.loc[mask2, 'q_m'] = stage_info.loc[stage_info['Stage Number'] == qm_stage_nr, 'q_m'].values[0]
    stage_info.loc[mask, 'p_0'] = stage_info.loc[stage_info['Stage Number'] == P0q0_stage_nr, 'p_0'].values[0]
    stage_info.loc[mask, 'q_0'] = stage_info.loc[stage_info['Stage Number'] == P0q0_stage_nr, 'q_0'].values[0]

    stage_info['q_p'] = stage_info['q_m'] - stage_info['q_0']
    

    stage_info['Cycles_accumulated'] = (stage_info['Cycles nr'] - 1).cumsum()
    
    q_max_values = []
    q_min_values = []
    q_avg_values = []
    # Iterate over the DataFrame rows using iterrows()
    for index, row in stage_info.iterrows():
        if row['Cycles nr'] == 0:
            q_max = 0
            q_min = 0
            q_avg = 0
        else:
            if index == 0:
                range_start = 0
            else:
                range_start = stage_info.loc[index - 1, 'Cycles_accumulated']
                range_end = row['Cycles_accumulated']

            q_max = [np.max(cycles_data[i]['Deviator Stress & Uncertainty (kPa)']) for i in range(range_start, range_end)]
            q_min = [np.min(cycles_data[i]['Deviator Stress & Uncertainty (kPa)']) for i in range(range_start, range_end)]
            q_avg = [umean(cycles_data[i]['Deviator Stress & Uncertainty (kPa)']) for i in range(range_start, range_end)]
            q_max = umean(q_max)
            q_min = umean(q_min)
            q_avg = umean(q_avg)
        q_max_values.append(q_max)
        q_min_values.append(q_min)
        q_avg_values.append(q_avg)

    stage_info['q_max'] = q_max_values
    stage_info['q_min'] = q_min_values
    stage_info['q_avg'] = q_avg_values
    stage_info['qcyc'] = (stage_info['q_max'] - stage_info['q_min']) / 2
    

    stage_info['qcyc/p0'] = stage_info.apply(lambda row: row['qcyc'] / row['p_0'] if row['p_0'] != 0 else 0, axis=1)
    stage_info['CSR'] = stage_info.apply(lambda row: row['qcyc'] / (2 * qf) if row['qcyc'] != 0 else 0, axis=1)    
    stage_info['qc/2Su'] = stage_info.apply(lambda row: row['qcyc'] / (2 * Su) if row['qcyc'] != 0 else 0, axis=1)
    stage_info['qm/2Su'] = stage_info.apply(lambda row: row['q_m'] / (2 * Su) if row['q_m'] != 0 else 0, axis=1)
    
    stage_info = stage_info[['Sample type', 'Stage Number', 'Start index', 'End index', 'Rows nr', 'Start time', 'End time', 'Test duration(d)', 'Cyclic loading duration(d)', 'Cycles nr', 'Cycle period (s)', 'Frequency(Hz)', 'p_0', 'q_0', 'q_p', 'q_m', 'q_min', 'q_max', 'q_avg', 'qcyc', 'qcyc/p0', 'CSR','qc/2Su','qm/2Su']]
    #total_df = pd.DataFrame([total_info])  # Create a DataFrame for total info
    stage_info = pd.concat([stage_info], ignore_index=True)  # Concatenate without resetting index
    return stage_info

def calculate_stage_info_all(df, cycles_data, name, P0q0_stage_nr, qm_stage_nr, qf, Su):
    stage_info_all = calculate_stage_info_base(df, cycles_data, name, P0q0_stage_nr, qm_stage_nr, qf, Su)
    # Concatenate without resetting index
    stage_info_all = pd.concat([stage_info_all], ignore_index=True)
    
    return stage_info_all




def calculate_stage_info(df, cycles_data, name, P0q0_stage_nr, qm_stage_nr, qf, Su, line_ranges):
    stage_info = calculate_stage_info_base(df, cycles_data, name, P0q0_stage_nr, qm_stage_nr, qf, Su)
    total_info = calculate_total_info(stage_info, name, line_ranges)
    total_info_df = pd.DataFrame(total_info)
    return total_info_df


def calculate_total_info(stage_info, name, line_ranges):
    total_infos = []  # List to store results for each line number
    for line_number, (start_stage, end_stage) in enumerate(line_ranges, start=1):
        mask_after_stage = (stage_info['Stage Number'] >= start_stage) & (stage_info['Stage Number'] <= end_stage)
        filtered_data_Cycleperiod = stage_info.loc[mask_after_stage, 'Cycle period (s)'].dropna()
        filtered_data_Frequency = stage_info.loc[mask_after_stage, 'Frequency(Hz)'].dropna()
        filtered_data_p_0 = stage_info.loc[mask_after_stage, 'p_0'].dropna()
        filtered_data_q_0 = stage_info.loc[mask_after_stage, 'q_0'].dropna()
        filtered_data_q_m = stage_info.loc[mask_after_stage, 'q_m'].dropna()
        filtered_data_q_p = stage_info.loc[mask_after_stage, 'q_p'].dropna()
        filtered_data_q_min = stage_info.loc[mask_after_stage, 'q_min'].dropna()
        filtered_data_q_max = stage_info.loc[mask_after_stage, 'q_max'].dropna()
        filtered_data_q_avg = stage_info.loc[mask_after_stage, 'q_avg'].dropna()
        filtered_data_qcyc = stage_info.loc[mask_after_stage, 'qcyc'].dropna()
        filtered_data_qcyc_p0 = stage_info.loc[mask_after_stage, 'qcyc/p0'].dropna()
        filtered_data_CSR = stage_info.loc[mask_after_stage, 'CSR'].dropna()
        filtered_data_qc_2Su = stage_info.loc[mask_after_stage, 'qc/2Su'].dropna()
        filtered_data_qm_2Su = stage_info.loc[mask_after_stage, 'qm/2Su'].dropna()
        
        
        total_info = {
            'Sample type': name,
            'Stage Number': line_number,
            'Start index': stage_info.loc[mask_after_stage, 'Start index'].min(),
            'End index': stage_info.loc[mask_after_stage, 'End index'].max(),
            'Rows nr': stage_info.loc[mask_after_stage, 'Rows nr'].sum(),
            'Star time': stage_info.loc[mask_after_stage, 'Start time'].min(),
            'End time': stage_info.loc[mask_after_stage, 'End time'].max(),
            'Test duration(d)': stage_info['Test duration(d)'].sum(),
            'Cyclic loading duration(d)': stage_info.loc[mask_after_stage, 'Cyclic loading duration(d)'].sum(),
            'Cycles nr': stage_info.loc[mask_after_stage, 'Cycles nr'].sum(),
            'Cycle period (s)': stage_info.loc[mask_after_stage, 'Cycle period (s)'].sum() / filtered_data_Cycleperiod[filtered_data_Cycleperiod != 0].shape[0] if filtered_data_Cycleperiod[filtered_data_Cycleperiod != 0].shape[0] != 0 else 0,
            'Frequency(Hz)': stage_info.loc[mask_after_stage, 'Frequency(Hz)'].sum() / filtered_data_Frequency[filtered_data_Frequency != 0].shape[0] if filtered_data_Frequency[filtered_data_Frequency != 0].shape[0] != 0 else 0,
            'p_0': stage_info.loc[mask_after_stage, 'p_0'].sum() / filtered_data_p_0[filtered_data_p_0 != 0].shape[0] if filtered_data_p_0[filtered_data_p_0 != 0].shape[0] != 0 else 0,
            'q_0': stage_info.loc[mask_after_stage, 'q_0'].sum() / filtered_data_q_0[filtered_data_q_0 != 0].shape[0] if filtered_data_q_0[filtered_data_q_0 != 0].shape[0] != 0 else 0,
            'q_m': stage_info.loc[mask_after_stage, 'q_m'].sum() / filtered_data_q_m[filtered_data_q_m != 0].shape[0] if filtered_data_q_m[filtered_data_q_m != 0].shape[0] != 0 else 0,
            'q_p': stage_info.loc[mask_after_stage, 'q_p'].sum() / filtered_data_q_p[filtered_data_q_p != 0].shape[0] if filtered_data_q_p[filtered_data_q_p != 0].shape[0] != 0 else 0,
            'q_min': stage_info.loc[mask_after_stage, 'q_min'].sum() / filtered_data_q_min[filtered_data_q_min != 0].shape[0] if filtered_data_q_min[filtered_data_q_min != 0].shape[0] != 0 else 0,
            'q_max': stage_info.loc[mask_after_stage, 'q_max'].sum() / filtered_data_q_max[filtered_data_q_max != 0].shape[0] if filtered_data_q_max[filtered_data_q_max != 0].shape[0] != 0 else 0,
            'q_avg': stage_info.loc[mask_after_stage, 'q_avg'].sum() / filtered_data_q_avg[filtered_data_q_avg != 0].shape[0] if filtered_data_q_avg[filtered_data_q_avg != 0].shape[0] != 0 else 0,
            'qcyc': stage_info.loc[mask_after_stage, 'qcyc'].sum() / filtered_data_qcyc[filtered_data_qcyc != 0].shape[0] if filtered_data_qcyc[filtered_data_qcyc != 0].shape[0] != 0 else 0,
            'qcyc/p0': stage_info.loc[mask_after_stage, 'qcyc/p0'].sum() / filtered_data_qcyc_p0[filtered_data_qcyc_p0 != 0].shape[0] if filtered_data_qcyc_p0[filtered_data_qcyc_p0 != 0].shape[0] != 0 else 0,
            'CSR': stage_info.loc[mask_after_stage, 'CSR'].sum() / filtered_data_CSR[filtered_data_CSR != 0].shape[0] if filtered_data_CSR[filtered_data_CSR != 0].shape[0] != 0 else 0,
            'qc/2Su': stage_info.loc[mask_after_stage, 'qc/2Su'].sum() / filtered_data_qc_2Su[filtered_data_qc_2Su != 0].shape[0] if filtered_data_qc_2Su[filtered_data_qc_2Su != 0].shape[0] != 0 else 0,
            'qm/2Su': stage_info.loc[mask_after_stage, 'qm/2Su'].sum() / filtered_data_qm_2Su[filtered_data_qm_2Su != 0].shape[0] if filtered_data_qm_2Su[filtered_data_qm_2Su != 0].shape[0] != 0 else 0
        }
        total_infos.append(total_info)
    
   
    # Create a DataFrame from the list of total information
    total_info_df = pd.DataFrame(total_infos)
    return total_info_df



#def calculate_total_info(stage_info, name, line_ranges):
#    total_infos = []  # List to store results for each line number
#    for line_number, (start_stage, end_stage) in enumerate(line_ranges, start=1):
#        mask_after_stage = (stage_info['Stage Number'] >= start_stage) & (stage_info['Stage Number'] <= end_stage)
#        non_zero_p_0     = stage_info.loc[mask_after_stage, 'p_0'   ][stage_info.loc[mask_after_stage, 'p_0'   ] != 0]
#        non_zero_q_0     = stage_info.loc[mask_after_stage, 'q_0'   ][stage_info.loc[mask_after_stage, 'q_0'   ] != 0]
#        non_zero_q_m     = stage_info.loc[mask_after_stage, 'q_m'   ][stage_info.loc[mask_after_stage, 'q_m'   ] != 0]
#        non_zero_q_p     = stage_info.loc[mask_after_stage, 'q_p'   ][stage_info.loc[mask_after_stage, 'q_p'   ] != 0]
#        non_zero_q_max   = stage_info.loc[mask_after_stage, 'q_max' ][stage_info.loc[mask_after_stage, 'q_max' ] != 0]
#        non_zero_q_min   = stage_info.loc[mask_after_stage, 'q_min' ][stage_info.loc[mask_after_stage, 'q_min' ] != 0]
#        non_zero_q_avg   = stage_info.loc[mask_after_stage, 'q_avg' ][stage_info.loc[mask_after_stage, 'q_avg' ] != 0]
#        non_zero_qcyc    = stage_info.loc[mask_after_stage, 'qcyc'  ][stage_info.loc[mask_after_stage, 'qcyc'  ] != 0]
#        non_zero_qcyc_p0 = stage_info.loc[mask_after_stage, 'qcyc/p0'][stage_info.loc[mask_after_stage,'qcyc/p0'] != 0]
#        non_zero_CSR     = stage_info.loc[mask_after_stage, 'CSR'   ][stage_info.loc[mask_after_stage, 'CSR'   ] != 0]
#        non_zero_qc_2Su  = stage_info.loc[mask_after_stage, 'qc/2Su'][stage_info.loc[mask_after_stage, 'qc/2Su'] != 0]       
#        non_zero_qm_2Su  = stage_info.loc[mask_after_stage, 'qm/2Su'][stage_info.loc[mask_after_stage, 'qm/2Su'] != 0]
#        
#        total_info = {
#            'Sample type': name,
#            'Stage Number': line_number,
#            'Start index': stage_info.loc[mask_after_stage, 'Start index'].min(),
#            'End index': stage_info.loc[mask_after_stage, 'End index'].max(),
#            'Rows nr': stage_info.loc[mask_after_stage, 'Rows nr'].sum(),
#            'Star time': stage_info.loc[mask_after_stage, 'Start time'].min(),
#            'End time': stage_info.loc[mask_after_stage, 'End time'].max(),
#            'Test duration(d)': stage_info['Test duration(d)'].sum(),
#            'Cyclic loading duration(d)': stage_info.loc[mask_after_stage, 'Cyclic loading duration(d)'].sum(),
#            'Cycles nr': stage_info.loc[mask_after_stage, 'Cycles nr'].sum(),
#            'Cycle period (s)': stage_info.loc[mask_after_stage, 'Cycle period (s)'].sum() / stage_info.loc[mask_after_stage, 'Cycle period (s)'].count(),
#            'Frequency(Hz)': stage_info.loc[mask_after_stage, 'Frequency(Hz)'].sum() / stage_info.loc[mask_after_stage, 'Frequency(Hz)'].count(),
#            'p_0':     non_zero_p_0.apply(lambda x: x.nominal_value).mean() if not non_zero_p_0 .empty else 0,
#            'q_0':     non_zero_q_0.apply(lambda x: x.nominal_value).mean() if not non_zero_q_0 .empty else 0,
#            'q_m':     non_zero_q_m.apply(lambda x: x.nominal_value).mean() if not non_zero_q_m .empty else 0,
#            'q_p':     non_zero_q_p.apply(lambda x: x.nominal_value).mean() if not non_zero_q_p.empty else 0,
#            'q_min':   non_zero_q_min.apply(lambda x: x.nominal_value).mean() if not non_zero_q_min.empty else 0,
#            'q_max':   non_zero_q_max.apply(lambda x: x.nominal_value).mean() if not non_zero_q_max.empty else 0,
#            'q_avg':   non_zero_q_avg.apply(lambda x: x.nominal_value).mean() if not non_zero_q_avg  .empty else 0,
#            'qcyc':    non_zero_qcyc.apply(lambda x: x.nominal_value).mean() if not non_zero_qcyc   .empty else 0,
#            'qcyc/p0': non_zero_qcyc_p0.apply(lambda x: x.nominal_value).mean() if not non_zero_qcyc_p0.empty else 0,
#            'CSR':     non_zero_CSR.apply(lambda x: x.nominal_value).mean() if not non_zero_CSR    .empty else 0,
#            'qc/2Su':  non_zero_qc_2Su.apply(lambda x: x.nominal_value).mean() if not non_zero_qc_2Su .empty else 0,
#            'qm/2Su':  non_zero_qm_2Su.apply(lambda x: x.nominal_value).mean() if not non_zero_qm_2Su .empty else 0,
#        }
#    #    total_infos.append(total_info)
#    #
#   
#    # Create a DataFrame from the list of total information
#    total_info_df = pd.DataFrame(total_infos)
#    return total_info_df


## Converter the values into its uncertainty when reading hte csv 

# Define a converter function
def convert_uncertainties(val):
    try:
        return ufloat_fromstr(val)
    except:
        return val

# Define a function to detect and convert uncertainty columns
def convert_uncertainty_columns(df):
    for col in df.columns:
        if df[col].dtype == object:  # if the column is of object (likely string) type
            # Try converting the first non-null element in the column
            first_non_null = df[col].dropna().iloc[0]
            try:
                ufloat_fromstr(first_non_null)
                # If the conversion succeeds, apply the conversion to the entire column
                df[col] = df[col].apply(convert_uncertainties)
            except:
                pass  # If the conversion fails, ignore the column
    return df



## Filter the outliers in the cycles
####################################
def Filter_outliers(data_x, data_y, f_b):
        
    ## remove outliers of the cycles when project it on horizontal axis.
    data = list(zip(data_x, data_y))
    q1 = np.percentile(data_x, 25)
    q3 = np.percentile(data_x, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    #lower_bound = q1 - (f_b1 * abs(iqr))
    upper_bound = q3 + (f_b * abs(iqr))

    outlier_data = [(x, y) for x, y in data if x > upper_bound]
    cleaned_data = [(x, y) for x, y in data if x <= upper_bound]

    #outlier_x = [x for x, y in outlier_data]
    #outlier_y = [y for x, y in outlier_data]
    cleaned_x = [x for x, y in cleaned_data]
    cleaned_y = [y for x, y in cleaned_data]

    ###
    ## get angle of max and min values of y.
    
    # Identify the maximum and minimum points of x_values
    X = np.array(cleaned_x)
    y = np.array(cleaned_y)
    
    # Define the model (linear regression)
    X = X.reshape(-1, 1)
    model = LinearRegression()
    
    # Initialize a large residual_threshold
    residual_threshold = 1000
    max_trials = 1000
    
    # Iteratively decrease residual_threshold until boundaries are captured
    while True:
        ransac = RANSACRegressor(model, min_samples=2, residual_threshold=residual_threshold, max_trials=max_trials)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        inlier_x = X[inlier_mask][:, 0]
        
        # Check if the minimum and maximum x-values are captured
        min_x = np.min(inlier_x)
        max_x = np.max(inlier_x)
        if min_x == np.min(X[:, 0]) and max_x == np.max(X[:, 0]):
           break
    
        # Decrease residual_threshold if boundaries are not captured
        residual_threshold /= 2

    # Get the fitted line parameters (slope and intercept)
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    
    # Calculate min_y and max_y using the captured boundaries
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    
    # Calculate the area of the triangle
    base = abs(max_x - min_x )
    height = abs(max_y - min_y)
    Cx_l  = min_x + base/ 2 
    Cy_l  = min_y + height / 2 
    angle_in_radians_ = np.arctan(slope)
    prefixed_major_axis_length = np.sqrt(height)**2 + (base)**2
    angle_in_degrees_ = np.degrees(angle_in_radians_)
    
 
    return cleaned_x, cleaned_y, min_x, max_x, min_y, max_y, Cx_l, Cy_l, angle_in_radians_, angle_in_degrees_, prefixed_major_axis_length 



""" The ingiven Data of the sample and Propogation error in Stress, Strain, pwp and deviator stress
"""



""" The peak deviatoric stress (qf ) which is taken from monotonic tests, in our study the value is 
no available, so the value is taken from other research which search the same soil and site ((Wong et al., 2023, p. 7)"""

Su_NC = 33   ##  kPa The undrained shear strength was determined from the monotonic tests 
qf_NC = 66 



""" The peak deviatoric stress (qf ) which is taken from monotonic tests, in our study the value is 
no available, so the value is taken from other research which search the same soil and site ((Wong et al., 2023, p. 7)"""

qf_LCC = np.mean([2.37, 2.53, 1.03, 1.34, 1.92, 2.56]) *1000   ## Unconfined compressive strength 
Su_LCC = qf_LCC / 2


""" The ingiven Data of the sample and Propogation error in Stress, Strain, pwp and deviator stress
"""
## Poisson's ratio
v = 0.2  
#the Uncertainties in the measurements
Ru_d_G = 0.0002      #    Relative Uncertainty in Global displacement sensor
Ru_d_L = 0.00105     #    Relative Uncertainty in Local displacement sensor 
Ru_l = 0.00          #    Relative Uncertainty in axial loading sensor 
v_error = 0.018      # mm Relative Uncertainty in vernier caliper 
Ru_c3 = 0.0086       #    Relative Uncertainty in cell pressure  
Ru_w = 0.0035        #    Relative Uncertainty in pore pressure  


#LCC03
######
H0_LCC03 = 103.36   # mm  initial height   
Lg_LCC03 = 47.37  # mm gauge length 
D0_LCC03 = 47.32 # mm  initial diameter 
A0_LCC03 = A(D0_LCC03)     # m^2
V0_LCC03 = V(A0_LCC03, Lg_LCC03)   # m^3


d_error_LCC = (50-45) / 2 # mm
Ru_A_LCC03 = (2 * d_error_LCC)/(D0_LCC03)                                           #  Relative Uncertainty in Area
Ru_H_G_LCC03 = (v_error/H0_LCC03)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain. 
Ru_H_L_LCC03 = (v_error/Lg_LCC03)                                                #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_LCC03 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_LCC03)**2))                 #  Relative Uncertainty in Global strain
Ru_strain_L_LCC03 = np.sqrt(((Ru_d_L)**2) + ((Ru_H_L_LCC03)**2))                 #  Relative Uncertainty in Local strain
Ru_stress_LCC03 = np.sqrt(((Ru_l)**2) + ((Ru_A_LCC03)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_LCC03 = np.sqrt(((Ru_stress_LCC03)**2) + ((Ru_strain_G_LCC03)**2))                       #  Relative Uncertainty in global E
Ru_E_L_LCC03 = np.sqrt(((Ru_stress_LCC03)**2) + ((Ru_strain_L_LCC03)**2))                       #  Relative Uncertainty in Local E
#Ru_Deviatoric_stress_LCC03 = np.sqrt(((Ru_stress_LCC03)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress


## LCC17
#######
H0_LCC17 = 100.98   # mm  initial height
Lg_LCC17 = 48.57  # mm gauge length 
D0_LCC17 = 47.82 # mm  initial diameter 
A0_LCC17 = A(D0_LCC17)     # m^2
V0_LCC17 = V(A0_LCC17, Lg_LCC17)   # m^3



d_error_LCC = (50-45) / 2 # mm
Ru_A_LCC17 = (2 * d_error_LCC)/(D0_LCC17)                                           #  Relative Uncertainty in Area
Ru_H_G_LCC17 = (v_error/H0_LCC17)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain. 
Ru_H_L_LCC17 = (v_error/Lg_LCC17)                                                #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_LCC17 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_LCC17)**2))                 #  Relative Uncertainty in Global strain
Ru_strain_L_LCC17 = np.sqrt(((Ru_d_L)**2) + ((Ru_H_L_LCC17)**2))                 #  Relative Uncertainty in Local strain
Ru_stress_LCC17 = np.sqrt(((Ru_l)**2) + ((Ru_A_LCC17)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_LCC17 = np.sqrt(((Ru_stress_LCC17)**2) + ((Ru_strain_G_LCC17)**2))                       #  Relative Uncertainty in global E
Ru_E_L_LCC17 = np.sqrt(((Ru_stress_LCC17)**2) + ((Ru_strain_L_LCC17)**2))                       #  Relative Uncertainty in Local E
#Ru_Deviatoric_stress_LCC17 = np.sqrt(((Ru_stress_LCC17)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress


## NC04
###
H0_NC04 = 100                 # mm  initial height
D0_NC04 = 50                  # mm  initial diameter 
A0_NC04 = A(D0_NC04)          # m^2
V0_NC04 = V(A0_NC04, H0_NC04/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC04 = (2 * d_error_LCC)/(D0_NC04)                                           #  Relative Uncertainty in Area
Ru_H_G_NC04 = (v_error/H0_NC04)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC04 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC04)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC04 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC04)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC04 = np.sqrt(((Ru_stress_NC04)**2) + ((Ru_strain_G_NC04)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC04 = np.sqrt(((Ru_stress_NC04)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress


## NC05
###
H0_NC05 = 100                 # mm  initial height
D0_NC05 = 50                  # mm  initial diameter 
A0_NC05 = A(D0_NC05)          # m^2
V0_NC05 = V(A0_NC05, H0_NC05/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC05 = (2 * d_error_LCC)/(D0_NC05)                                           #  Relative Uncertainty in Area
Ru_H_G_NC05 = (v_error/H0_NC05)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC05 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC05)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC05 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC05)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC05 = np.sqrt(((Ru_stress_NC05)**2) + ((Ru_strain_G_NC05)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC05 = np.sqrt(((Ru_stress_NC05)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress



## NC06
###
H0_NC06 = 100                 # mm  initial height
D0_NC06 = 50                  # mm  initial diameter 
A0_NC06 = A(D0_NC06)          # m^2
V0_NC06 = V(A0_NC06, H0_NC06/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC06 = (2 * d_error_LCC)/(D0_NC06)                                           #  Relative Uncertainty in Area
Ru_H_G_NC06 = (v_error/H0_NC06)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC06 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC06)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC06 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC06)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC06 = np.sqrt(((Ru_stress_NC06)**2) + ((Ru_strain_G_NC06)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC06 = np.sqrt(((Ru_stress_NC06)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress


## NC07
###
H0_NC07 = 100                 # mm  initial height
D0_NC07 = 50                  # mm  initial diameter 
A0_NC07 = A(D0_NC07)          # m^2
V0_NC07 = V(A0_NC07, H0_NC07/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC07 = (2 * d_error_LCC)/(D0_NC07)                                           #  Relative Uncertainty in Area
Ru_H_G_NC07 = (v_error/H0_NC07)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC07 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC07)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC07 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC07)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC07 = np.sqrt(((Ru_stress_NC07)**2) + ((Ru_strain_G_NC07)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC07 = np.sqrt(((Ru_stress_NC07)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress



## NC08
###
H0_NC08 = 100                 # mm  initial height
D0_NC08 = 50                  # mm  initial diameter 
A0_NC08 = A(D0_NC08)          # m^2
V0_NC08 = V(A0_NC08, H0_NC08/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC08 = (2 * d_error_LCC)/(D0_NC08)                                           #  Relative Uncertainty in Area
Ru_H_G_NC08 = (v_error/H0_NC08)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC08 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC08)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC08 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC08)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC08 = np.sqrt(((Ru_stress_NC08)**2) + ((Ru_strain_G_NC08)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC08 = np.sqrt(((Ru_stress_NC08)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress

## NC09
###
H0_NC09 = 100                 # mm  initial height
D0_NC09 = 50                  # mm  initial diameter 
A0_NC09 = A(D0_NC09)          # m^2
V0_NC09 = V(A0_NC09, H0_NC09/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC09 = (2 * d_error_LCC)/(D0_NC09)                                           #  Relative Uncertainty in Area
Ru_H_G_NC09 = (v_error/H0_NC09)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC09 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC09)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC09 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC09)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC09 = np.sqrt(((Ru_stress_NC09)**2) + ((Ru_strain_G_NC09)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC09 = np.sqrt(((Ru_stress_NC09)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress

## NC10
###
H0_NC10 = 100                 # mm  initial height
D0_NC10 = 50                  # mm  initial diameter 
A0_NC10 = A(D0_NC10)          # m^2
V0_NC10 = V(A0_NC10, H0_NC10/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC10 = (2 * d_error_LCC)/(D0_NC10)                                           #  Relative Uncertainty in Area
Ru_H_G_NC10 = (v_error/H0_NC10)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC10 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC10)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC10 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC10)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC10 = np.sqrt(((Ru_stress_NC10)**2) + ((Ru_strain_G_NC10)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC10 = np.sqrt(((Ru_stress_NC10)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress

## NC
###
H0_NC11 = 100                 # mm  initial height
D0_NC11 = 50                  # mm  initial diameter 
A0_NC11 = A(D0_NC11)          # m^2
V0_NC11 = V(A0_NC11, H0_NC11/1000) # m^3


d_error_LCC = 0 # mm
Ru_A_NC11 = (2 * d_error_LCC)/(D0_NC11)                                           #  Relative Uncertainty in Area
Ru_H_G_NC11 = (v_error/H0_NC11)                                                #  Relative Uncertainty in change of the hieght of specimne for global strain.                                           #  Relative Uncertainty in change of the hieght of specimne for local strain.   
Ru_strain_G_NC11 = np.sqrt(((Ru_d_G)**2) + ((Ru_H_G_NC11)**2))                 #  Relative Uncertainty in Global strain
Ru_stress_NC11 = np.sqrt(((Ru_l)**2) + ((Ru_A_NC11)**2))                       #  Relative Uncertainty in Local stress
Ru_E_G_NC11 = np.sqrt(((Ru_stress_NC11)**2) + ((Ru_strain_G_NC11)**2))                       #  Relative Uncertainty in global E
#Ru_Deviatoric_stress_NC11 = np.sqrt(((Ru_stress_NC11)**2) + ((Ru_c3)**2))      # Relative Uncertainty in deviatoric stress
