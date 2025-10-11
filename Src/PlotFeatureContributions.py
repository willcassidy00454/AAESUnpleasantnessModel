import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


if __name__ == "__main__":
    features = ["Colour.", "Flutter", "Asym.", "Curv.", "Damp."]

    standardised_betas = [[0.359, 0.227, 0.236, 0.317, 0.223],
                          [0.681, -0.147, -0.149, 0.242, -0.200]]

    # This is the "part" column in SPSS when showing "part and partial correlations"
    # Each column here is for each programme item
    semi_partial_correlations = [[0.227, 0.171, 0.164, 0.243, 0.191],
                                 [0.430, -0.111, -0.103, 0.186, -0.172]]
    r_squares = [0.726, 0.753]

    unique_proportions_of_data_variance = np.square(semi_partial_correlations)
    unique_proportions_of_model_variance = [unique_proportions_of_data_variance[i, :] / r_squares[i] for i in range(2)]

    significances = [["***", "***", "**", "***", "***"],
                     ["***", "*", "*", "***", "***"]]

    fig, axes = plt.subplots(2)
    fig.set_size_inches(4, 6)
    fig.set_layout_engine("tight")

    rc("font",**{"size": 11, "sans-serif": "CMU Serif"})

    for plot_index, prog_item_betas in enumerate(standardised_betas):
        p = axes[plot_index].bar(features, prog_item_betas, zorder=2)

        axes[plot_index].bar_label(p, labels=significances[plot_index], label_type='edge')
        axes[plot_index].set_title("Saxophone" if plot_index else "Clapping")
        axes[plot_index].grid(zorder=0, axis="y")
        axes[plot_index].set_ylim([-0.25, 0.8])
        axes[plot_index].set_yticks([-0.25, 0.0, 0.25, 0.5, 0.75])
        axes[plot_index].set_yticklabels(["-0.25", "0.00", "0.25", "0.50", "0.75"])
        axes[plot_index].set_ylabel("Standardised Beta Coefficients")

    plt.show()

    fig, axes = plt.subplots(2)
    fig.set_size_inches(4, 6)
    fig.set_layout_engine("tight")

    rc("font",**{"size": 11, "sans-serif": "CMU Serif"})

    for plot_index, prog_item_unique_contributions in enumerate(unique_proportions_of_model_variance):
        p = axes[plot_index].bar(features, prog_item_unique_contributions, zorder=2)

        axes[plot_index].set_title("Saxophone" if plot_index else "Clapping")
        axes[plot_index].grid(zorder=0, axis="y")
        axes[plot_index].set_ylim([0.0, 0.27])
        axes[plot_index].set_yticks([0.0, 0.10, 0.20])
        axes[plot_index].set_yticklabels(["0 %", "10 %", "20 %"])
        axes[plot_index].set_ylabel("Unique Proportion of Model Variance")

    plt.show()