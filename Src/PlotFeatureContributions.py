import matplotlib.pyplot as plt


if __name__ == "__main__":
    features = ["Colour.", "Flut.", "Asym.", "Curv.", "Damp."]
    contributions = [[0.327, 0.267, 0.245, -0.386, 0.196],
                     [0.554, -0.263, 0.005, -0.239, -0.154]]

    significances = [["***", "***", "***", "***", "**"],
                     ["***", "***", "", "***", "*"]]

    fig, axes = plt.subplots(2)
    fig.set_size_inches(4, 6)
    fig.set_layout_engine("tight")

    for plot_index, prog_item_contributions in enumerate(contributions):
        p = axes[plot_index].bar(features, prog_item_contributions, zorder=2)

        axes[plot_index].bar_label(p, labels=significances[plot_index], label_type='edge')
        axes[plot_index].set_title("Saxophone" if plot_index else "Clapping")
        axes[plot_index].grid(zorder=0, axis="y")
        axes[plot_index].set_ylim([-0.6, 0.6])
        axes[plot_index].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        axes[plot_index].set_yticklabels(["-50 %", "-25 %", "0 %", "25 %", "50 %"])

    plt.show()