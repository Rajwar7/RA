#This file contains 23 classicle functions (SET 1 : F1-F23) and 27 special funcitons (SET 2: F24-F50)

import numpy
import numpy as np
import math


##############################################

# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def F1(x):
    s = numpy.sum((x) ** 2)
    return s


def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (numpy.sum(x[0:i])) ** 2
    return o


def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = numpy.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = numpy.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = numpy.sum(w * (x ** 4)) + numpy.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (numpy.sin(numpy.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = numpy.sum(x ** 2 - 10 * numpy.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 4000 - prod(numpy.cos(x / numpy.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((numpy.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + numpy.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((numpy.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + numpy.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (numpy.sin(3 * numpy.pi * x[:,0])) ** 2
        + numpy.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (numpy.sin(3 * numpy.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (numpy.sin(2 * numpy.pi * x[:,-1])) ** 2)
    ) + numpy.sum(Ufun(x, 5, 100, 4))
    return o


def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = numpy.asarray(aS)
    bS = numpy.zeros(25)
    v = numpy.matrix(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = numpy.sum((numpy.power(H, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    o = ((1.0 / 500) + numpy.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = numpy.asarray(aK)
    bK = numpy.asarray(bK)
    bK = 1 / bK
    fit = numpy.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (numpy.pi ** 2)) + 5 / numpy.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


# map the inputs to the function blocks
def F19(L):
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = numpy.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = numpy.asarray(cH)
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    pH = numpy.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * numpy.exp(-(numpy.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F20(L):
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    aH = numpy.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = numpy.asarray(cH)
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    pH = numpy.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * numpy.exp(-(numpy.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F21(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(5):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F22(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(7):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F23(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(10):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


#Special funtion______________________________________________________________________________________________________

def F24(x):         #Chung Reynolds
    s =   (numpy.sum((x) ** 2))**2
    return s


def F25(x):         #Schwefel 2.20
    s =   (numpy.sum(numpy.abs(x)))
    return s

def F26(x):         #Schwefel 2.23
    s =   (numpy.sum(x**10))
    return s


###step function def F27(x):
def F27(x):         #Box function
    s =   numpy.sum(np.floor(np.abs(x)))
    return s


def F28(x):         #Alpine
    s =   numpy.sum(numpy.abs(x * numpy.sin(x) + 0.1 * x))
    return s


def F29(x):         #Exponential
    s =   -numpy.exp(-0.5*numpy.sum(x*x))
    return s


def F30(x):        #Xin-She Yang 2
    s = (numpy.sum(numpy.abs(x))) * numpy.exp(-numpy.sum((numpy.sin(x))**2))
    return s


def F31(x):        #Xin-She Yang 3
    s = numpy.exp(-numpy.sum((x/15)**10)) - 2*numpy.exp(-numpy.sum((x)**2))*prod((numpy.cos(x))**2)
    return s


def F32(L):        #Bartels Conn
    o = (
       numpy.abs(L[0]*L[0] + L[1]*L[1] + L[0]*L[1])
       + numpy.abs(numpy.sin(L[0]))
       + numpy.abs(numpy.cos(L[1]))
    )
    return o


def F33(L):        #Bohachevsky 1
    o = (
       L[0]*L[0] + 2*L[1]*L[1] - 0.3*numpy.cos(3*numpy.pi*L[0])
       - 0.4 * numpy.cos(4 * numpy.pi * L[1])
    )
    return o


def F34(L):        #Bohachevsky 2
    o = (
       L[0]*L[0] + 2*L[1]*L[1] - 0.3*numpy.cos(3*numpy.pi*L[0])
       * 0.4 * numpy.cos(4 * numpy.pi * L[1])
    )
    return o


def F35(L):        #Bohachevsky 3
    o = (
       L[0]*L[0] + 2*L[1]*L[1] - 0.3*numpy.cos(3*numpy.pi*L[0] + 4 * numpy.pi * L[1])
    )
    return o


def F36(L):        #Camel-Three Hump
    o = (
       2*(L[0]*L[0]) + 1.05*(L[0]**4) + (L[0]**6)/6 + (L[0]*L[1]) + (L[1]*L[1])
    )
    return o


def F37(L):        #Chichinadze
    o = (
        L[0]*L[0] + 12*L[0] + 11 +
        10*numpy.cos(numpy.pi* L[0]/2) +
        8*numpy.sin(5*numpy.pi* L[0]/2)-
        ((1/5)**0.5)*numpy.exp(-0.5*( L[1]-0.5)**2)
    )
    return o


def F38(L):        #Cross-in-Trat
    o = (
       -0.0001*(numpy.abs(numpy.sin(L[0])*
        numpy.sin(L[1])*numpy.exp(numpy.abs(100-
        numpy.sqrt(L[0]*L[0]+L[1]*L[1])/numpy.pi)))+1)**0.1
    )
    return o


def F39(L):        #egg crate
    o = (
      L[0]*L[0] + L[1]*L[1] + 25 *
      (numpy.sin(L[0])*numpy.sin(L[0]) + numpy.sin(L[1])*numpy.sin(L[1]))
    )
    return o


def F40(L):        #Matyas
    o = (
       0.26*(L[0]*L[0]+L[1]*L[1])
       - 0.04*L[0]*L[1]
    )
    return o


def F41(L):        #periodic
    o = (
       1 +
      (numpy.sin(L[0])*numpy.sin(L[0])) + (numpy.sin(L[1])*numpy.sin(L[1]))
       - 0.1*numpy.exp(-L[0]*L[0]-L[1]*L[1])
    )
    return o

#
# def F42(L):        #RUMP
#     o = (
#         (333.75-L[0]**2)*(L[1]**6)
#         + (L[0]*L[0])*((11*(L[0]*L[0]) * (L[1]**4)) -2)
#         + 5.5*(L[1]**8)
#         + L[0]/(2*L[1])
#     )
#     return o


def F42(L):            #Rotated Ellipse
    o = (
       (L[0] ** 2)
        -  (L[0] * L[1])
        + (L[1] ** 2)
    )
    return o


def F43(L):            #Stenger
    o = (
       (L[0] ** 2)
        -  (L[0] * L[1])
        + (L[1] ** 2)
    )
    return o


# def F44(L):
#     o = (               #Trecanni
#        (4*L[0] ** 2 - 4 * L[1])**2
#         +  (L[1] **2 - 2 * L[0] + 4 * L[1] )**2
#     )
#     return o
def F44(L):
    o = (               #Trecanni
       L[0]**4 + 4*(L[0]**3) + 4*L[0]**2 + L[1]**2
    )
    return o

def F45(L):           #venter
    o = (
      L[0]**2 - 100 * (numpy.cos(L[0]))**2 - 100*numpy.cos((L[0]*L[0])/30)
       + L[1]*L[1] - 100 * (numpy.cos(L[1]))**2 - 100 * (numpy.cos((L[1]*L[1])/30))
    )
    return o

def F46(L):           #Wayburn seader 1
    o = (
        (L[0]**6 + L[1]**4 -17)**2 + (2*L[0] + L[1] -4)**2
    )
    return o


def F47(L):           #Wayburn seader 2
    o = (
            (1.631 - 4*(L[0]-0.3125)**2-4*(L[1]-1.625)**2)**2 + (L[1]-1)**2
    )
    return o



def F48(L):           #Wayburn seader 3
    o = (
            -numpy.cos(L[0])*numpy.cos(L[0])*numpy.exp( -(L[0]-numpy.pi)**2-(L[1]-numpy.pi)**2)
    )
    return o


def F49(L):           #Branin RCOS
    o = (
        (L[1] - (5.1*L[1]**2) / (4*numpy.pi*numpy.pi)  +  (5*L[1]*L[1])/(numpy.pi) - 6) **2 + 10*(1-1/(8*numpy.pi))*numpy.cos(L[0]) + 10
    )
    return o




def F50(L):           #Deckkers-Aarts
    o = (
        (10**5) * (L[0]**2) + L[1]**2 - (L[0]**2 + L[1]**2)**2 + (10**-5)*(L[0]**2 + L[1]**2)**4
    )
    return o





def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -70, 70, 30],
        "F2": ["F2", -2, 2, 30],
        "F3": ["F3", -25, 25,30],
        "F4": ["F4", -25, 25, 30],
        "F5": ["F5", -10, 10, 30],
        "F6": ["F6", -50, 50, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -25, 25, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -20, 20, 30],
        "F11": ["F11", -100, 100, 30],
        "F12": ["F12", -25, 25, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -2, 2, 2],
        "F17": ["F17", -5, 5, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 3, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
         #special funtion________________
        "F24": ["F24",  -25, 25, 30],
        "F25": ["F25", -100, 100, 30],
        "F26": ["F26", -10, 10, 30],
        "F27": ["F27", -100, 100, 30],
        "F28": ["F28", -10, 10, 30],
        "F29": ["F29", -1, 1, 30],
        "F30": ["F30", -2 * numpy.pi,  2 * numpy.pi, 30],
        "F31": ["F31", -20, 20, 30],
        "F32": ["F32", -500, 500, 2],
        "F33": ["F33", -100, 100, 2],
        "F34": ["F34", -20, 20, 2],
        "F35": ["F35", -100, 100, 2],
        "F36": ["F36", -12, 12, 2],
        "F37": ["F37", -30, 30, 2],
        "F38": ["F38", -10, 10, 2],
        "F39": ["F39", -2, 2, 2],
        "F40": ["F40", -10, 10, 2],
        "F41": ["F41", -10, 10, 2],
        #"F42": ["F42", -500, 500, 2],
        "F42": ["F42",-500, 500, 2],
        "F43": ["F43", -1, 4, 2],
        "F44": ["F44", -1, 1, 2],
        "F45": ["F45", -50, 50, 2],
        "F46": ["F46", -100, 100, 2],
        "F47": ["F47", -500, 500, 2],
        "F48": ["F48", -100, 100, 2],
        "F49": ["F49", -5, 15, 2],
        "F50": ["F50", -20, 20, 2],



        "Gear_train" : ["Gear_train", 16, 60, 4],


    }
    return param.get(a, "nothing")











