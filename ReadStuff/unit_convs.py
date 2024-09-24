import numpy as np

"""The file contains the functions that do the appropriate unit conversions for the states and inputs from the experimental data"""
""" Standard Units:
    Temperature: deference from 250 deg-C
    Mass flow rate: g/s
    Concentration: g/ml or g/cm^3
    Area: cm^2

    Concentration   & $mol/cm^{3} = mol/ml$ 
    Time            & $s$ 
    Mass            & $g$ 
    Length          & $cm$ 
    Temperature     & $(T-250)\lx{^o}{C}$ 
"""

def uConv(x, conv_type):
    """Unit conversion for the states"""
    match conv_type:
        case "T250C":
            return np.array([T250C(xi) for xi in x])
        case "kg/min to g/s":
            return np.array([xi * kgmin2gsec_gain for xi in x])
#===

kgmin2gsec_gain = 16.6667              # Conversion factor from kg/min to g/sec
gsec2kgmin_gain = 1 / kgmin2gsec_gain  # Conversion factor from g/sec to kg/min
#===

def T250C(T):
    """Make the temperature around 250 deg-C"""
    return T - 250
#===

