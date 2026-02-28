import numpy as np

def academic_explanation(values, horizon=""):

    values = np.array(values)

    mean_val = np.mean(values)
    variance = np.var(values)

    explanation = f"""
Academic Interpretation ({horizon}):

The projected load demonstrates statistically observable 
temporal dynamics with mean demand of {mean_val:,.2f} MW 
and variance of {variance:,.2f}.

The forecasting model captures nonlinear dependencies 
through attention-weighted temporal encoding and 
multi-horizon supervision, allowing consistent 
short-term and medium-term generalization.
"""

    return explanation