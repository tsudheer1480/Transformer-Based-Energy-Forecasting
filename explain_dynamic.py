import os
import numpy as np

def generate_dynamic_explanation(feature_impacts, p10_real, p50_real, p90_real):

    # Detect peak hour
    peak_hour = int(np.argmax(p50_real)) + 1
    peak_value = float(np.max(p50_real))

    # Calculate average uncertainty
    avg_uncertainty = float((p90_real - p10_real).mean())

    # Generate 2-line simple explanation
    summary = (
        "\n\n================= SIMPLE MODEL SUMMARY =================\n\n"
        "The forecast is mainly based on recent electricity usage patterns "
        "and the regular daily demand cycle. Since electricity consumption "
        f"typically rises in the evening, the model predicts a peak around "
        f"hour {peak_hour} (approximately {peak_value:,.0f} MW), while also "
        f"providing a reasonable uncertainty range of about {avg_uncertainty:,.0f} MW "
        "to account for normal variations.\n"
    )

    print(summary)

    # Save to file
    os.makedirs("results/reports", exist_ok=True)
    with open("results/reports/simple_model_summary.txt", "w") as f:
        f.write(summary)

    print("Simple summary saved to results/reports/simple_model_summary.txt")
