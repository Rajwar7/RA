#RA:
# AUTHOR- KANCHAN RAJWAR, 2022#


import random
import math
import time
import numpy


from solution import solution

def RA(objf, lb, ub, dim, N, Max_iteration):

    # region define
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    Positions = numpy.zeros((N, dim))


    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
        )



    Fitness = numpy.full(N, float("inf"))
    for i in range(0, N):
        Fitness[i] = objf(Positions[i, :])
    Fitness = numpy.sort(Fitness)


    Convergence_curve = numpy.zeros(Max_iteration)
    s = solution()

    # Loop counter
    print('RA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")


    Iteration = 0

    while Iteration < Max_iteration:

        for i in range(0, N):
            # evaluate each pop
            Fitness[i] = objf(Positions[i, :])

        I = numpy.argsort(Fitness)

        ################################################

        Fitness = numpy.sort(Fitness)
        Convergence_curve[Iteration] = Fitness[0]
        if Iteration % 1 == 0:
            print(["At iteration "
                   + str(Iteration)
                   + " the best fitness is "
                   + str(Fitness[0])
                   ]
                  )

        ################################################
        sorted_Positions = numpy.copy(Positions[I, :])



        #PARAMETER____________________________________________________________________________________________________


        amp1 = random.randint(1, 51)                                                                # amp1 = P'
        r1=random.random()
        amp = math.floor(((64 - amp1) / 8) * r1 * (1 - (Iteration+1) / Max_iteration)) + 1          # amp = P, Eq. (3)



        #_____________________________________________________________________________________________________________

        amp_positions = sorted_Positions[0:amp]

        xm = numpy.mean(amp_positions, axis=0)    #p_mean
        xb=amp_positions[0]                       #p_best

        c1 =  math.exp(-((3 * Iteration / Max_iteration) ** 3))* math.sin(2*Iteration)   #c1 = alpha,  Eq. (6)



        # W generate:###################################################################################################

        W = sorted_Positions[amp:N].copy()
        for i in range(0, N - amp):
            if amp - i - 1 < 1:
                x = amp_positions[0]
            else:
                x = amp_positions[amp - i - 1]


            for j in range(0, dim):
                c2 = random.random()
                c3 = random.random()

                W[i][j] = x[j] +   c3 * c1 * ((ub[j] - lb[j]) * c2 + lb[j])             #   Eq. (5)


        # B generate:##################################################################################################
        B = W.copy()

        for i in range(0, N-amp):

            if amp - i - 1 < 1:
                x = amp_positions[0]
            else:
                x = amp_positions[amp - i - 1]

            r1 = random.random()
            if r1 <= 0.75:
               x_m=  numpy.multiply(2,xb)-B[i]
               x_i=  numpy.multiply(2,xb)-x

               S = [B[i], x, x_m, x_i]
               r = random.random()

               a= (r/(Iteration+1))*math.sin(1/r) + 0.5                                  # a = beta, Eq. (12)
               s1, s2, s3, s4 = random.sample(S, 4)
               v1 = numpy.multiply(a,s1) + numpy.multiply(1-a,s2)
               v2 = numpy.multiply(1-a,s3) + numpy.multiply(a,s4)
               B[i]=(numpy.array(v1)+numpy.array(v2))/2                                  # Eq. (9)

            else:
                 for j in range(0, dim):
                     r2=random.random()
                     B[i][j] = ((ub[j] - lb[j]) * r2 + lb[j])


        # R generate:##################################################################################################
        R = numpy.copy(B)
        for i in range(0, N - amp):
            rand2 = random.random()
            rand3 = random.random()
            beta = 1.5
            sigma = (math.gamma(1 * beta * math.sin(math.pi * beta / 2))) / (
                    (math.gamma((1 + beta) / 2)) * beta * 2 ** ((beta - 1) / 2))        # d1 = delta, Eq. (17)
            LD = ((0.01 * rand2 * sigma) / ((rand3) ** (1 / beta))) * (((-1) ** Iteration) / (Iteration + 1))

            e1= 1-LD
            e2=(((-1)**Iteration)/(1+Iteration))

            d1=e1*e2                                                                    # d1 = delta, Eq. (15)
            d2=LD                                                                       # d2 = LD,    Eq. (14)
            r=random.random()


            if objf(B[i]) <= objf(xb):
                R[i] = numpy.multiply(d1 ,B[i]) + numpy.multiply(d2, xb)+ (r*r)*(1/(Iteration+1))*xm
            else:                                                                                          #   Eq. (13)
                R[i] = numpy.multiply(d1, xb) + numpy.multiply(d2, B[i]) + (r*r)*(1/(Iteration+1))*xm
        # Patterning ############################################################################################
        #arg min function code in python

        R1=[]
        for i in range(0, N - amp):
            if objf(R[i])<=objf(B[i]) and objf(R[i])<=objf(W[i]) :
                R1.append(R[i])
            elif objf(B[i])<=objf(R[i]) and objf(B[i])<=objf(W[i]) :                                       #   Eq. (18)
                R1.append(B[i])
            else:
                R1.append(W[i])
            for j in range(0, dim):
                 R1[i][j] = numpy.clip(R1[i][j], lb[j], ub[j])
        ################################################################################################################

        Positions = numpy.vstack((amp_positions, R1))

        for i in range(0, N):
            # evaluate each pop
            Fitness[i] = objf(Positions[i, :])

        Iteration = Iteration + 1
    print("Final Solution (X) = ", Positions[0])
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "RA"
    s.objfname = objf.__name__

    return s

############################################################################################

