import matplotlib.pyplot as plt


if __name__ == "__main__":
    features = ["Colour.", "Flut.", "Asym.", "Curv.", "Damp."]
    contributions = [[0.261, 0.253, 0.139, -0.472, 0.286],
                     [0.461, -0.334, 0.026, -0.293, -0.079]]

    fig, axes = plt.subplots(2)
    fig.set_size_inches(4, 6)
    fig.set_layout_engine("tight")

    for plot_index, prog_item_contributions in enumerate(contributions):
        axes[plot_index].bar(features, prog_item_contributions, zorder=2)
        axes[plot_index].set_title("Saxophone" if plot_index else "Clapping")
        axes[plot_index].grid(zorder=0, axis="y")
        axes[plot_index].set_ylim([-0.45, 0.45])
        axes[plot_index].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        axes[plot_index].set_yticklabels(["-50 %", "-25 %", "0 %", "25 %", "50 %"])

    plt.show()