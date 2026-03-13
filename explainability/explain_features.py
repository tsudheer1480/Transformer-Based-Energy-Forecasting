def feature_explanation(feature_cols, feature_importance=None):

    top_features = []

    if feature_importance:
        top_features = [f[0] for f in feature_importance[:3]]

    top_text = ", ".join(top_features) if top_features else "recent demand signals"

    explanation = f"""
Feature Influence Overview:

The forecasting model analyzes {len(feature_cols)} input variables
representing temporal behavior, historical load patterns,
and environmental conditions.

Among these signals, {top_text} emerged as the most influential
drivers for the current forecast.

Lag-based features capture short-term demand momentum,
while temporal indicators (hour of day and day of week)
help the model learn recurring consumption cycles.

Weather signals provide environmental context that
can influence electricity usage patterns.

Together these inputs allow the model to detect both
immediate demand changes and repeating seasonal patterns.
"""

    return explanation