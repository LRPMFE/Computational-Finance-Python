## Pricing MBS

We price MBS product using Numerix Prepayment Model (www.numerix.com).  
Assume interest rates follow CIR model dr_t = k (r_bar - r_t) dt + sigma * sqrt(r_t) dW_t

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import optimize
```

Instead of using the random function in python, we build up the succession of “pseudorandom” numbers using
Lewis Goodman, Miller- IBM (LGM) method. We generate uniformly distributed random variables on [0,1].

```python
def uniform(n): 
    m=(2**31)-1
    a=7**5
    lgmnum=[9]*n
    for i in range(n-1):
        lgmnum[i+1]=(lgmnum[i]*a)%m
    uniformrand=np.array(lgmnum)/m
    return uniformrand
```

We generate n standard normally distributed random variables by Box-Muller method using the uniform variables.

```python
def normal(uniformrand,n):
    Z1=[0]*int(n)
    Z2=[0]*int(n)
    for i in range(int(n)):
        Z1[i]=np.sqrt(-2*np.log(uniformrand[2*i]))*np.cos(2*np.pi*uniformrand[2*i+1])
        Z2[i]=np.sqrt(-2*np.log(uniformrand[2*i]))*np.sin(2*np.pi*uniformrand[2*i+1])  
    return np.hstack((Z1,Z2))
```

Build the CIR process

```python
def Bond_CIR(r0,sigma,kappa,rbar,T):
    h1=(kappa**2+2*sigma**2)**0.5
    h2=(kappa+h1)/2
    h3=2*kappa*rbar/(sigma**2)
    AT=(h1*np.exp(h2*T)/(h2*(np.exp(h1*T)-1)+h1))**h3
    BT=(np.exp(h1*T)-1)/(h2*(np.exp(h1*T)-1)+h1)
    PT=AT*np.exp(-BT*r0)
    return PT
```

Use Monte Carlo simulation to buid the MBS pricing model

```python
def MBS_CIR(WAC,T,Loan,r0,kappa,rbar,sigma,sim):
    N=int(T*12)
    r=WAC/12
    uniformrand=uniform(N*sim)
    Z=normal(uniformrand,int(N*sim/2))
    Z=np.reshape(Z,(sim,N))
    rt,dt=np.zeros((sim,N+1)),np.zeros((sim,N+1))
    rt[:,0]=r0
    for i in range(N):
        rt[:,i+1]=rt[:,i]+kappa*(rbar-rt[:,i])/12+np.sign(rt[:,i])*sigma*(np.abs(rt[:,i])/12)**0.5*Z[:,i]
        dt[:,i]=np.exp(-np.sum(rt[:,0:i],axis=1)/12)
    PV,rt10,RI,BU,SG,CPR,c,tpp=np.zeros((sim,N+1)),np.zeros((sim,N+1)),np.zeros((sim,N+1)),np.zeros((sim,N+1)),np.zeros((sim,N+1)),np.zeros((sim,N+1)),np.zeros((sim,N+1)),np.zeros((sim,N+1))
    PV[:,0]=Loan
    SY=np.array([0.94,0.76,0.74,0.95,0.98,0.92,0.98,1.1,1.18,1.22,1.23,0.98]*T)
    for i in range(1,N+1):
        rt10[:,i-1]=-1/10*np.log(Bond_CIR(rt[:,i-1],sigma,kappa,rbar,10))
        RI[:,i]=0.28 + 0.14*np.arctan(-8.57 + 430*(WAC-rt10[:,i-1]))
        BU[:,i]=0.3 + 0.7*PV[:,i-1]/PV[:,0]
        SG[:,i]=np.minimum(1,i/T)
        CPR[:,i]=RI[:,i]*BU[:,i]*SG[:,i]*SY[i-1]
        tpp[:,i]=PV[:,i-1]*r*(1/(1-(1+r)**(-N+i-1))-1)+(PV[:,i-1]-PV[:,i-1]*r*(1/(1-(1+r)**(-N+i-1))-1))*(1-(1-CPR[:,i])**(1/12))
        c[:,i]=tpp[:,i]+PV[:,i-1]*r
        PV[:,i]=PV[:,i-1]-tpp[:,i]
    price=np.mean(np.sum(c[:,1:(N+1)]*dt[:,1:(N+1)],axis=1))
    return price
```

We can compute the price of the MBS, given parameters. We simulate 50000 times.

```python
WAC,T,Loan,r0,kappa,rbar,sigma,sim=0.08,30,100000,0.078,0.6,0.08,0.12,50000
MBS_a=MBS_CIR(WAC,T,Loan,r0,kappa,rbar,sigma,sim)
```

We can get the MBS price 100724.26.  

Compute the price of the MBS for different kappa and plot the price

```python
MBS_b=[0]*7
i=0
for kappa_b in np.arange(0.3,1.0,0.1):
    MBS_b[i]=MBS_CIR(WAC,T,Loan,r0,kappa_b,rbar,sigma,sim)
    i=i+1
    
plt.figure()
plt.figure(figsize=(8,6))
plt.plot(np.arange(0.3,1.0,0.1),MBS_b)
plt.xlabel("Kappa")
plt.ylabel("Price")
plt.title("1b_MBS Price under Numerix Prepayment Model")
```
![](/MBS/1b.png)

Compute the price of the MBS for different r_bar and plot the price

```python
MBS_c=[0]*7
i=0
for rbar_c in np.arange(0.03,0.091,0.01):
    MBS_c[i]=MBS_CIR(WAC,T,Loan,r0,kappa,rbar_c,sigma,sim)
    i=i+1
    
plt.figure()
plt.figure(figsize=(8,6))
plt.plot(np.arange(0.03,0.091,0.01),MBS_c)
plt.xlabel("rbar")
plt.ylabel("Price")
plt.title("1c_MBS Price under Numerix Prepayment Model")
```
![](/MBS/1c.png)

Compute the price of the MBS for different sgima and plot the price
```python
MBS_d=[0]*11
i=0
for sigma_d in np.arange(0.1,0.2001,0.01):
    MBS_d[i]=MBS_CIR(WAC,T,Loan,r0,kappa,rbar,sigma_d,sim)
    i=i+1
    
plt.figure()
plt.figure(figsize=(8,6))
plt.plot(np.arange(0.1,0.2001,0.01),MBS_d)
plt.xlabel("sigma")
plt.ylabel("Price")
plt.title("1d_BS Price under Numerix Prepayment Model")
```
![](/MBS/1d.png)
