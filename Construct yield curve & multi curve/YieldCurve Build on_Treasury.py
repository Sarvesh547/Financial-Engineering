import numpy as np
import enum 

from copy import deepcopy
from scipy.interpolate import splrep, splev, interp1d

# This class defines puts and calls
class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0
    
#Swap function
def IRSwap(CP,notional,K,t,Ti,Tm,n,P0T):
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t
    ti_grid = np.linspace(Ti,Tm,int(n))
    tau = ti_grid[1]- ti_grid[0]
    
    # overwrite Ti if t>Ti
    prevTi = ti_grid[np.where(ti_grid<t)]
    if np.size(prevTi) > 0: #prevTi != []:
        Ti = prevTi[-1]
    
    
    # Now we need to handle the case when some payments are already done
    ti_grid = ti_grid[np.where(ti_grid>t)]           

    temp= 0.0
        
    for (idx,ti) in enumerate(ti_grid):
        if ti>Ti:
            temp = temp + tau * P0T(ti)
            
    P_t_Ti = P0T(Ti)
    P_t_Tm = P0T(Tm)
    
    if CP==OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp #We just sum ZCB first & last payment date & divide by annuity
    elif CP==OptionTypeSwap.RECEIVER:
        swap = K * temp - (P_t_Ti - P_t_Tm)
    
    return swap * notional

def P0TModel(t,ti,ri,method):
    rInterp = method(ti,ri)
    r = rInterp(t)
    return np.exp(-r*t)

def YieldCurve(instruments, maturities, r0, method, tol):
    r0 = deepcopy(r0)
    ri = MultivariateNewtonRaphson(r0, maturities, instruments, method, tol=tol)
    return ri

def MultivariateNewtonRaphson(ri, ti, instruments, method, tol):
    err = 10e10
    idx = 0
    while err > tol:      #tol=10^-15
        idx = idx +1  #keep tracking the index or error
        #ti = apine points, ri= time of spine points, inst= vector of swap, method=linear interpolation
        values = EvaluateInstruments(ti,ri,instruments,method)
        #Jacobian is sensitivity of individual pv
        J = Jacobian(ti,ri, instruments, method)
        J_inv = np.linalg.inv(J)
        err = - np.dot(J_inv, values) 
        ri[0:] = ri[0:] + err #next set of spine point
        err = np.linalg.norm(err)
        print('index in the loop is',idx,' Error is ', err)
    return ri

def Jacobian(ti, ri, instruments, method):
    #Jacobian calculate perform numerically i.e shocking of curve i.e spine points
    eps = 1e-05
    swap_num = len(ti)
    J = np.zeros([swap_num, swap_num]) #empty matrix
    val = EvaluateInstruments(ti,ri,instruments,method)
    ri_up = deepcopy(ri) #copy spine points
    
    for j in range(0, len(ri)):
        ri_up[j] = ri[j] + eps  
        #for every spine point we evaluate our instruments & store the result
        val_up = EvaluateInstruments(ti,ri_up,instruments,method)
        ri_up[j] = ri[j]
        #eps = shock size, val_up= value after stock, val=berfore shock
        dv = (val_up - val) / eps
        J[:, j] = dv[:]
    return J

def EvaluateInstruments(ti,ri,instruments,method):
    #construct a yeild curve on given ti & ri
    P0Ttemp = lambda t: P0TModel(t,ti,ri,method)
    val = np.zeros(len(instruments))
    for i in range(0,len(instruments)):
        val[i] = instruments[i](P0Ttemp) #evaluate on each swap as func of ZCB
    return val

#we are interpulating log of ZCB (log/time)
def linear_interpolation(ti,ri):
    interpolator = lambda t: np.interp(t, ti, ri)
    return interpolator

def mainCode():
    
    # Convergence tolerance
    tol = 1.0e-15 #multidimensional newton raphson stoping curve
    # Initial guess for the spine points
    r0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #spine points 1% 
    # Interpolation method
    method = linear_interpolation
    
    # Construct swaps that are used for building of the yield curve
    #Swap rate taken from market
    K   = np.array([0.04/100.0,	0.16/100.0,	0.31/100.0,	0.81/100.0,	1.28/100.0,	1.62/100.0,	2.22/100.0,	2.30/100.0])
    mat = np.array([1.0,2.0,3.0,5.0,7.0,10.0,20.0,30.0]) #Vector for maturity
    
    #Dividing swap rate because we have to evaluate each of swap inst on whole vector of spine point
    swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[0],0.0,0.0,mat[0],4*mat[0],P0T)
    swap2 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[1],0.0,0.0,mat[1],4*mat[1],P0T)
    swap3 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[2],0.0,0.0,mat[2],4*mat[2],P0T)
    swap4 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[3],0.0,0.0,mat[3],4*mat[3],P0T)
    swap5 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[4],0.0,0.0,mat[4],4*mat[4],P0T)
    swap6 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[5],0.0,0.0,mat[5],4*mat[5],P0T)
    swap7 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[6],0.0,0.0,mat[6],4*mat[6],P0T)
    swap8 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[7],0.0,0.0,mat[7],4*mat[7],P0T)
    instruments = [swap1,swap2,swap3,swap4,swap5,swap6,swap7,swap8] #each swap is lambda exp which depends on ZCB
    #object here is find ZCB function, find ZCB=t such that all those instruments price back to par
    
    # determine optimal spine points
    #yield curve construcion & yield curve return our spine points
    ri = YieldCurve(instruments, mat, r0, method, tol)
    print('\n Spine points are',ri,'\n')
    
    # Build a ZCB-curve/yield curve from the spine points
    P0T_Initial = lambda t: P0TModel(t,mat,r0,method)
    P0T = lambda t: P0TModel(t,mat,ri,method) #yc based on maturity & ri is spine points obtain from calibration
    
    # price back the swaps
    swapsModel = np.zeros(len(instruments))
    swapsInitial = np.zeros(len(instruments))
    for i in range(0,len(instruments)):
        #once we have P0T_initital & P0T we can iterate our all the swap we have in our portfolio & calculate what is current PV of swap
        swapsInitial[i] = instruments[i](P0T_Initial)
        swapsModel[i] = instruments[i](P0T)
    
    print('Prices for Pas Swaps (initial) = ',swapsInitial,'\n')
    print('Prices for Par Swaps = ',swapsModel,'\n')
    
    return 0.0

mainCode()
    
