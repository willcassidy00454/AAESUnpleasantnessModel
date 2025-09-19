import matplotlib.pyplot as plt


if __name__ == "__main__":
    features = ["Colour.", "Flut.", "Asym.", "Curv.", "Damp."]
    contributions = [[0.385, 0.122, 0.161, 0.278, 0.314],
                     [0.448, 0.379, 0.027, 0.216, 0.105]]

    fig, axes = plt.subplots(2)
    fig.set_size_inches(4, 6)
    fig.set_layout_engine("tight")

    for plot_index, prog_item_contributions in enumerate(contributions):
        axes[plot_index].bar(features, prog_item_contributions, zorder=2)
        axes[plot_index].set_title("Saxophone" if plot_index else "Clapping")
        axes[plot_index].grid(zorder=0, axis="y")
        axes[plot_index].set_ylim([0, 0.45])
        axes[plot_index].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        axes[plot_index].set_yticklabels(["0 %", "10 %", "20 %", "30 %", "40 %", "50 %"])

    plt.show()