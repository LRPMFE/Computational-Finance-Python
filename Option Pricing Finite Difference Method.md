# Pricing European/American option using Explicit Finite-Difference method, Implicit Finite-Difference method, and Crank-Nicolson Finite-Difference method.

### 1. Pricing European option using EFD, IFD and C-NFD method

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
```
Pricing a European put option using Explicit Finite-Difference method.

```python
def dX_1(sigma,dt,n):
    return sigma*(n*dt)**0.5

def EFD_EuroPut(S0,sigma,r,dt,dX,T,K,Smax,Smin):
    N=int((np.log(Smax)-np.log(Smin))/dX)+1
    M=int(T/dt)
    S=np.exp(np.arange(np.log(Smax),np.log(Smin),-dX))
    Pu=dt*(sigma**2/(2*dX**2)+(r-sigma**2/2)/(2*dX))
    Pm=1-dt*sigma**2/dX**2-r*dt
    Pd=dt*(sigma**2/(2*dX**2)-(r-sigma**2/2)/(2*dX))
    A=np.eye(N-2,N)*Pu+np.eye(N-2,N,1)*Pm+np.eye(N-2,N,2)*Pd
    A_first=np.hstack((np.array([[Pu,Pm,Pd]]),np.zeros((1,N-3))))
    A_last=np.hstack((np.zeros((1,N-3)),np.array([[Pu,Pm,Pd]])))
    A=np.vstack((A_first,A,A_last))
    B=np.zeros((N,1))
    B[N-1,0]=-(S[N-1]-S[N-2])
    F=np.zeros((N,M+1))
    F[:,M]=np.maximum(K-S,0)
    for i in range(M,0,-1):
        F[:,i-1]=np.dot(A,F[:,i])+B.flatten()
    return F[:,0]
```

Pricing a European put option using Implicit Finite-Difference method.
```python
def IFD_EuroPut(S0,sigma,r,dt,dX,T,K,Smax,Smin):
    N=int((np.log(Smax)-np.log(Smin))/dX)+1
    M=int(T/dt)
    S=np.exp(np.arange(np.log(Smax),np.log(Smin),-dX))
    Pu=-1/2*dt*(sigma**2/(dX**2)+(r-sigma**2/2)/dX)
    Pm=1+dt*sigma**2/dX**2+r*dt
    Pd=-1/2*dt*(sigma**2/(dX**2)-(r-sigma**2/2)/dX)
    A=np.eye(N-2,N)*Pu+np.eye(N-2,N,1)*Pm+np.eye(N-2,N,2)*Pd
    A_first=np.hstack((np.array([[1,-1]]),np.zeros((1,N-2))))
    A_last=np.hstack((np.zeros((1,N-2)),np.array([[1,-1]])))
    A=np.vstack((A_first,A,A_last))
    F=np.zeros((N,M+1))
    F[:,M]=np.maximum(K-S,0)
    F[0,M]=0
    F[N-1,M]=(S[N-1]-S[N-2]).flatten()
    for i in range(M,0,-1):
        F[:,i-1]=np.dot(np.linalg.inv(A),F[:,i])
        if i==1:
            break
        else:
            F[0,i-1]=0
            F[N-1,i-1]=(S[N-1]-S[N-2])
    return F[:,0]
```

Pricing a European put option using Crank-Nicolson Finite-Difference method.

```python
def CNFD_EuroPut(S0,sigma,r,dt,dX,T,K,Smax,Smin):
    N=int((np.log(Smax)-np.log(Smin))/dX)+1
    M=int(T/dt)
    S=np.exp(np.arange(np.log(Smax),np.log(Smin),-dX))
    Pu=-1/4*dt*(sigma**2/(dX**2)+(r-sigma**2/2)/dX)
    Pm=1+dt*sigma**2/(2*dX**2)+r*dt/2
    Pd=-1/4*dt*(sigma**2/(dX**2)-(r-sigma**2/2)/dX)
    A=np.eye(N-2,N)*Pu+np.eye(N-2,N,1)*Pm+np.eye(N-2,N,2)*Pd
    A_first=np.hstack((np.array([[1,-1]]),np.zeros((1,N-2))))
    A_last=np.hstack((np.zeros((1,N-2)),np.array([[1,-1]])))
    A=np.vstack((A_first,A,A_last))
    F=np.zeros((N,M+1))
    F[:,M]=np.maximum(K-S,0)
    zcoef=-np.eye(N-2,N)*Pu-np.eye(N-2,N,1)*(Pm-2)-np.eye(N-2,N,2)*Pd
    zcoef=np.vstack((np.zeros((1,N)),zcoef,np.zeros((1,N))))
    for i in range(M,0,-1):
        z=np.dot(zcoef,F[:,i])
        z[N-1]=(S[N-1]-S[N-2])
        F[:,i-1]=np.dot(np.linalg.inv(A),z)
    return F[:,0]
```

Pricing  a European put option using Black-Scholes-Merton result.

```python
def BS_put(S0,sigma,r,T,K):
    d1=(np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*T**0.5)
    d2=d1-sigma*T**0.5
    price=K*np.exp(-r*T)*scipy.stats.norm.cdf(-d2)-S0*scipy.stats.norm.cdf(-d1)
    return price
```
Assign the parameters and assume current stock prices range from 4 to 16
```python
S0,sigma,r,dt,T,K,Smax,Smin=10,0.2,0.04,0.002,0.5,10,16,4
```

Assign different delta_x value. Assume delta_x = sigma * sqrt(delta_t), sigma * sqrt(3 * delta_t), sigma * sqrt(4 * delta_t).  

We calculate all the put option prices with different delta_x and stock prices and compare the reuslts from Monte Carlo Simulation and 
those from Black-Scholes-Merton model.

```python
dX1=dX_1(sigma,dt,1)
EFD_EuroPut1=EFD_EuroPut(S0,sigma,r,dt,dX1,T,K,Smax,Smin)
IFD_EuroPut1=IFD_EuroPut(S0,sigma,r,dt,dX1,T,K,Smax,Smin)
CNFD_EuroPut1=CNFD_EuroPut(S0,sigma,r,dt,dX1,T,K,Smax,Smin)
S0_list=np.exp(np.arange(np.log(Smax),np.log(Smin),-dX1))
N1=int((np.log(Smax)-np.log(Smin))/dX1)+1
BS_put_1=np.zeros((N1,1))
for i in range(N1):
    BS_put_1[i]=BS_put(S0_list[i],sigma,r,T,K)

dX3=dX_1(sigma,dt,3)
EFD_EuroPut3=EFD_EuroPut(S0,sigma,r,dt,dX3,T,K,Smax,Smin)
IFD_EuroPut3=IFD_EuroPut(S0,sigma,r,dt,dX3,T,K,Smax,Smin)
CNFD_EuroPut3=CNFD_EuroPut(S0,sigma,r,dt,dX3,T,K,Smax,Smin)
S0_list=np.exp(np.arange(np.log(Smax),np.log(Smin),-dX3))
N3=int((np.log(Smax)-np.log(Smin))/dX3)+1
BS_put_3=np.zeros((N3,1))
for i in range(N3):
    BS_put_3[i]=BS_put(S0_list[i],sigma,r,T,K)

dX4=dX_1(sigma,dt,4)
EFD_EuroPut4=EFD_EuroPut(S0,sigma,r,dt,dX4,T,K,Smax,Smin)
IFD_EuroPut4=IFD_EuroPut(S0,sigma,r,dt,dX4,T,K,Smax,Smin)
CNFD_EuroPut4=CNFD_EuroPut(S0,sigma,r,dt,dX4,T,K,Smax,Smin)
S0_list=np.exp(np.arange(np.log(Smax),np.log(Smin),-dX4))
N4=int((np.log(Smax)-np.log(Smin))/dX4)+1
BS_put_4=np.zeros((N4,1))                                                                                                                                  
for i in range(N4):
    BS_put_4[i]=BS_put(S0_list[i],sigma,r,T,K)
```
From the results, we can see that the prices using these three methods under Monte Carlo and BSM are almost the same.
    
### 2. Pricing American option using EFD, IFD and C-NFD method

Use the Black-Scholes PDE for stock prices to price American Call/Put using EFD, IFD and C-NFD method

```python
def a1(sigma,r,alpha,j):
    return (sigma**2*j**2-r*j)*(1-alpha)/2
def a2(sigma,r,alpha,j):
    return -1/dt-(sigma**2*j**2+r)*(1-alpha)
def a3(sigma,r,alpha,j):
    return (sigma**2*j**2+r*j)*(1-alpha)/2
def b1(sigma,r,alpha,j):
    return (sigma**2*j**2-r*j)*alpha/2
def b2(sigma,r,alpha,j):
    return 1/dt-(sigma**2*j**2+r)*alpha
def b3(sigma,r,alpha,j):
    return (sigma**2*j**2+r*j)*alpha/2

def AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,method,CallorPut):
    if method=="EFD":
        alpha=1
    elif method=="IFD":
        alpha=0
    elif method=="CNFD":
        alpha=0.5
    N=int((Smax-Smin)/dS)
    M=int(T/dt)
    S=np.arange(Smax,Smin-dS,-dS)
    A=np.zeros((N-1,N+1))
    Bcoef=np.zeros((N-1,N+1))
    for i in range(N-1):
        Bcoef[i,i]=b3(sigma,r,alpha,N-i-1)
        Bcoef[i,i+1]=b2(sigma,r,alpha,N-i-1)
        Bcoef[i,i+2]=b1(sigma,r,alpha,N-i-1)
        A[i,i]=a3(sigma,r,alpha,N-i-1)
        A[i,i+1]=a2(sigma,r,alpha,N-i-1)
        A[i,i+2]=a1(sigma,r,alpha,N-i-1)   
    A_first=np.hstack((np.array([[1,-1]]),np.zeros((1,N-1))))   
    A_last=np.hstack((np.zeros((1,N-1)),np.array([[1,-1]])))
    A=np.vstack((A_first,A,A_last))
    if method=="EFD":
        Bcoef_first=np.hstack((np.array([[b3(sigma,r,alpha,N-1),b2(sigma,r,alpha,N-1),b1(sigma,r,alpha,N-1)]]),np.zeros((1,N-2))))
        Bcoef_last=np.hstack((np.zeros((1,N-2)),np.array([[b3(sigma,r,alpha,1),b2(sigma,r,alpha,1),b1(sigma,r,alpha,1)]])))
    else:
        Bcoef_first=np.zeros((1,N+1))
        Bcoef_last=np.zeros((1,N+1))
    Bcoef=np.vstack((Bcoef_first,Bcoef,Bcoef_last))
    B=np.zeros((N+1,1))
    F=np.zeros((N+1,M+1))
    if CallorPut=="call":
        F[:,M]=np.maximum(S-K,0)
        B[0,0]=S[0]-S[1]
    elif CallorPut=="put":
        F[:,M]=np.maximum(K-S,0)
        B[N,0]=-S[0]+S[1]
    for i in range(M,0,-1):
        if method=="EFD":
            d=np.reshape(np.dot(Bcoef,F[:,i]),(N+1,1))*dt+B
            F[:,i-1]=np.maximum(d.flatten(),F[:,M])
        else:
            d=np.reshape(np.dot(-Bcoef,F[:,i]),(N+1,1))+B
            F[:,i-1]=np.maximum(np.dot(np.linalg.inv(A),d).flatten(),F[:,M])
    if method=="EFD":
        price=F[:,0]
    else:
        price=F
    return price
```
Assign values to parameters and stock prices range from 4 to 16 with dS = 0.25 and 1. Calculate American call/put option prices.

```python
S0,sigma,r,dt,dS,T,K,Smin,Smax,method,CallorPut=10,0.2,0.04,0.002,1.25,0.5,10,4,16,"EFD","call"

dS=0.25
EFDcall1=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"EFD","call")
IFDcall1=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"IFD","call")
CNFDcall1=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"CNFD","call")

EFDput1=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"EFD","put")
IFDput1=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"IFD","put")
CNFDput1=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"CNFD","put")
S1=np.arange(Smax,Smin-dS,-dS)

dS=1
EFDcall2=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"EFD","call")
IFDcall2=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"IFD","call")
CNFDcall2=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"CNFD","call")

EFDput2=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"EFD","put")
IFDput2=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"IFD","put")
CNFDput2=AmeriOption_deltaS(S0,sigma,r,dt,dS,T,K,Smin,Smax,"CNFD","put")
S2=np.arange(Smax,Smin-dS,-dS)
```
Plot the American call/put option prices.

```python
plt.figure()
plt.figure(figsize=(12,8))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(221)
plt.plot(S1,EFDcall1)
plt.plot(S1,IFDcall1)
plt.plot(S1,CNFDcall1)
plt.legend(['EFDcall', 'IFDcall', 'CNFDcall'])
plt.xlabel("Stock Price")
plt.ylabel("Option Price")
plt.title("Call Option Prices with dS=0.25")
plt.subplot(222)
plt.plot(S1,EFDput1)
plt.plot(S1,IFDput1)
plt.plot(S1,CNFDput1)
plt.legend(['EFDput', 'IFDput', 'CNFDput'])
plt.xlabel("Stock Price")
plt.ylabel("Option Price")
plt.title("Put Option Prices with dS=0.25")
plt.subplot(223)
plt.plot(S2,EFDcall2)
plt.plot(S2,IFDcall2)
plt.plot(S2,CNFDcall2)
plt.legend(['EFDcall', 'IFDcall', 'CNFDcall'])
plt.xlabel("Stock Price")
plt.ylabel("Option Price")
plt.title("Call Option Prices with dS=1")
plt.subplot(224)
plt.plot(S2,EFDput2)
plt.plot(S2,IFDput2)
plt.plot(S2,CNFDput2)
plt.legend(['EFDput', 'IFDput', 'CNFDput'])
plt.xlabel("Stock Price")
plt.ylabel("Option Price")
plt.title("Put Option Prices with dS=1")
plt.show()
```

![](/FD method/FD Pricing.png)

We can see that the prices are almost the same under different methods.
