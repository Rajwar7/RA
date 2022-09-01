import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()

    fileResultsDetailsData = pd.read_csv(results_directory + "/experiment_details.csv")
    for j in range(0, len(objectivefunc)):

        # Box Plot
        data = []

        for i in range(len(optimizer)):
            objective_name = objectivefunc[j]
            optimizer_name = optimizer[i]

            detailedData = fileResultsDetailsData[
                (fileResultsDetailsData["Optimizer"] == optimizer_name)
                & (fileResultsDetailsData["objfname"] == objective_name)
            ]
            detailedData = detailedData["Iter" + str(Iterations)]
            detailedData = np.array(detailedData).T.tolist()
            data.append(detailedData)


        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.00

        # , notch=True
        box = plt.boxplot(data, patch_artist=True, labels=optimizer)

        colors = [
            "#404636",
        ]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        plt.ylabel('Function Value', fontname="Times New Roman", size=16, fontweight="bold")
        plt.xticks(color='black', fontweight='bold', fontsize='10')
        plt.yticks(color='black', fontweight='bold', fontsize='10')

        fig_name = results_directory + "/boxplot-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight", dpi=500)
        plt.clf()
        plt.show()
