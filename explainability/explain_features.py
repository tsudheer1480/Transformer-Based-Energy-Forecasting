def feature_explanation(feature_cols):

    explanation = f"""
Feature Contribution Analysis:

The model integrates {len(feature_cols)} input features 
including lag variables, weather signals, and 
cyclical encodings (hour/day seasonality).

Lag-based features capture short-term autocorrelation, 
while cyclical transformations enable learning of 
periodic demand structures.

This hybrid feature representation enhances 
temporal generalization capability.
"""

    return explanation