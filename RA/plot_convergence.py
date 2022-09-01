import matplotlib.pyplot as plt
import pandas as pd


def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment.csv")

    for j in range(0, len(objectivefunc)):
        objective_name = objectivefunc[j]

        startIteration = 0

        allGenerations = [x + 1 for x in range(startIteration, Iterations)]

        colors = ['fuchsia']

        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.30


        plt.gca().set_prop_cycle(color=colors)
        for i in range(len(optimizer)):
            optimizer_name = optimizer[i]

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            row = row.iloc[:, 3 + startIteration :]
            plt.plot(allGenerations, row.values.tolist()[0], label=optimizer_name, linewidth=2.75)
        plt.xlabel('Iteration', fontname="Times New Roman", size=23,fontweight="bold")
        plt.ylabel('Fitness', fontname="Times New Roman", size=23,fontweight="bold")


        plt.xticks(color='black', fontweight='bold', fontsize='15')
        plt.yticks(color='black', fontweight='bold', fontsize='15')


        plt.ylim()
        plt.xlim()


        plt.legend(loc="upper right")
        #plt.legend()
        #plt.grid()
        fig_name = results_directory + "/convergence-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight", dpi=500)
        plt.clf()

