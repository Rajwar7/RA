'''
This is the ultimate control file.
Fixed the parameter as desire and run this file.
'''

from optimizer import run




optimizer = ["RA"]                                  #do not chnage this


objectivefunc = ["F1"]                              #choose any benchmark function from the set { "F1",  "F2",  "F3", ... "F49","F50"}


NumOfRuns =30                                       #Select number of repetitions for each experiment.
                                                    #To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.


params = {"PopulationSize": 50, "Iterations":500}   #Select general parameters for all optimizers (population size, number of iterations) ....


export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,                    #Choose whether to Export the results in different formats
    "Export_boxplot": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)





