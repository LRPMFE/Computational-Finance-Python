# Pricing models under Vasicek, CIR, and G2++

### 1. Bond/option pricing under Vasicek 

```python
import numpy as np
from scipy.stats import ncx2
```

Instead of using the random function in python, we build up the succession of “pseudorandom” numbers  
using Lewis Goodman, Miller- IBM (LGM) method. We generate uniformly distributed random variables on [0,1].

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
Use Monte Carlo simulation to price zero coupon bond under Vasicek.

```python
def Vasicek_zeroCBond(r0,sigma,kappa,rbar,T,FaceV,sim):
    N=int(T*365)
    dt=1/365
    randuniform=uniform(N*sim)
    randnormal=normal(randuniform,int(N*sim/2))
    randnormal=np.reshape(randnormal,(sim,N))
    rt=np.zeros((sim,N+1))
    rt[:,0]=r0
    for i in range(N):
        rt[:,i+1]=rt[:,i]+kappa*(rbar-rt[:,i])*dt+sigma*dt**0.5*randnormal[:,i]
    price=np.mean(FaceV*np.exp(-np.sum(rt[:,1:(N+1)]*dt,axis=1)))
    return price
```
We can compute the price of the MBS, given parameters, simulating 1000 times.

```python
r0,sigma,kappa,rbar,T,FaceV,sim=0.05,0.18,0.82,0.05,0.5,1000,1000
price_1a=Vasicek_zeroCBond(r0,sigma,kappa,rbar,T,FaceV,sim)
```
The zero coupon bond price is 975.55.

Use Monte Carlo simulation to price semiannual coupon bond under Vasicek.

```python
def Vasicek_CBond(r0,sigma,kappa,rbar,T,FaceV,C,sim):
    N=int(np.max(T)*365)
    D=np.int_(np.array(T)*365)
    dt=1/365
    randuniform=uniform(N*sim)
    randnormal=normal(randuniform,int(N*sim/2))
    randnormal=np.reshape(randnormal,(sim,N))
    rt=np.zeros((sim,N+1))
    rt[:,0]=r0
    for i in range(N):
        rt[:,i+1]=rt[:,i]+kappa*(rbar-rt[:,i])*dt+sigma*dt**0.5*randnormal[:,i]
    PV=np.zeros((sim,1))
    for j in range(len(T)):
        if j!=len(T)-1:
            PV=PV+C*np.exp(-np.sum(rt[:,1:(D[j]+1)]*dt,axis=1))
        else:
            PV=PV+(FaceV+C)*np.exp(-np.sum(rt[:,1:(D[j]+1)]*dt,axis=1))
    price=np.mean(PV)
    return price
```

```python
T,C=[0.5,1,1.5,2,2.5,3,3.5,4],30
price_1b=Vasicek_CBond(r0,sigma,kappa,rbar,T,FaceV,C,sim)
```
The semiannual coupon bond price is	1088.36.

Use Monte Carlo simulation to price a European Call option on zero coupon bond under Vasicek.

```python
def EuroCall_Vasicek_c(r0,sigma,kappa,rbar,T,S,FaceV,sim,K):
    B=(1-np.exp(-kappa*(S-T)))/kappa
    A=np.exp((rbar-sigma**2/(2*kappa**2))*(B-(S-T))-sigma**2/(4*kappa)*B**2)
    N=int(T*365)
    dt=1/365
    randuniform=uniform(N*sim)
    randnormal=normal(randuniform,int(N*sim/2))
    randnormal=np.reshape(randnormal,(sim,N))
    rt=np.zeros((sim,N+1))
    rt[:,0]=r0
    for i in range(N):
        rt[:,i+1]=rt[:,i]+kappa*(rbar-rt[:,i])*dt+sigma*dt**0.5*randnormal[:,i]
    P=FaceV*A*np.exp(-B*rt[:,N])
    price=np.mean(np.exp(-np.sum(rt[:,1:(N+1)]*dt,axis=1))*np.maximum(P-K,0))
    return price
```

```python
K,T,S=980,0.25,0.5
price_1c=EuroCall_Vasicek_c(r0,sigma,kappa,rbar,T,S,FaceV,sim,K)
```
The European Call option on zero coupon bond is 11.69.

In order to price European option on coupon bond, we need first to find r_star.

```python
def findrstar_Vasicek(sigma,kappa,rbar,T,S,FaceV,C,sim,K):
    C=[C]*len(S)
    C[len(S)-1]=C[0]+FaceV
    rmin=0
    rmax=0.3
    iteration=1000
    S=np.array(S)
    C=np.array(C)
    B=(1-np.exp(-kappa*(S-T)))/kappa
    A=np.exp((rbar-sigma**2/(2*kappa**2))*(B-(S-T))-sigma**2/(4*kappa)*B**2)
    for i in range(iteration):
        rstar=(rmin+rmax)/2
        if abs(np.sum(A*np.exp(-B*rstar)*C)-K)<0.0001:
            break
        elif np.sum(A*np.exp(-B*rstar)*C)-K>=0.0001:
            rmin=rstar
        elif np.sum(A*np.exp(-B*rstar)*C)-K<=-0.0001:
            rmax=rstar
    return rstar
```
Use Monte Carlo simulation to price a European Call option on coupon bond under Vasicek.

```python
def EuroCall_Vasicek_d(r0,sigma,kappa,rbar,T,S,FaceV,C,sim,K):
    rstar=findrstar_Vasicek(sigma,kappa,rbar,T,S,FaceV,C,sim,K)
    C=[C]*len(S)
    C[len(S)-1]=C[0]+FaceV
    S=np.array(S)
    C=np.array(C)
    B=(1-np.exp(-kappa*(S-T)))/kappa
    A=np.exp((rbar-sigma**2/(2*kappa**2))*(B-(S-T))-sigma**2/(4*kappa)*B**2)
    K=A*np.exp(-B*rstar)*C
    NS=int(np.max(S)*365)
    DS=np.int_(np.array(S)*365)
    NT=int(T*365)
    dt=1/365
    randuniform=uniform(NS*sim)
    randnormal=normal(randuniform,int(NS*sim/2))
    randnormal=np.reshape(randnormal,(sim,NS))
    rt=np.zeros((sim,NS+1))
    rt[:,0]=r0
    for i in range(NS):
        rt[:,i+1]=rt[:,i]+kappa*(rbar-rt[:,i])*dt+sigma*dt**0.5*randnormal[:,i]
    Payoff=np.zeros((sim,len(S)))
    for j in range(len(S)):
        Payoff[:,j]=np.maximum(C[j]*np.exp(-np.sum(rt[:,(NT+1):(DS[j]+1)]*dt,axis=1))-K[j],0)
    PV=np.sum(Payoff,axis=1)
    price=np.mean(np.exp(-np.sum(rt[:,1:(NT+1)]*dt,axis=1))*PV)
    return price
```

```python
T,S,C,K=0.25,[0.5,1,1.5,2,2.5,3,3.5,4],30,980
price_1d=EuroCall_Vasicek_d(r0,sigma,kappa,rbar,T,S,FaceV,C,sim,K)
```
The European Call option on semiannual coupon bond is 120.19.

### 2. Option pricing under CIR

Use Monte Carlo simulation to price a European Call option on zero bond under CIR.

```python
def EuroCall_CIR(r0,sigma,kappa,rbar,T,S,FaceV,sim,K):
    NS=int(S*365)
    NT=int(T*365)
    dt=1/365
    randuniform=uniform(NT*sim)
    randnormal=normal(randuniform,int(NT*sim/2))
    randnormal=np.reshape(randnormal,(sim,NT))
    rt1=np.zeros((sim,NT+1))
    rt1[:,0]=r0
    for i in range(NT):
        rt1[:,i+1]=rt1[:,i]+kappa*(rbar-rt1[:,i])*dt+sigma*(rt1[:,i]*dt)**0.5*randnormal[:,i]
    randuniform=uniform((NS-NT)*sim)
    randnormal=normal(randuniform,int((NS-NT)*sim/2))
    randnormal=np.reshape(randnormal,(sim,(NS-NT)))
    Payoff=np.zeros((sim,1))
    for k in range(sim):
        rt2=np.zeros((sim,(NS-NT)+1))
        rt2[:,0]=rt1[k,NT]
        for j in range(NS-NT):
            rt2[:,j+1]=rt2[:,j]+kappa*(rbar-rt2[:,j])*dt+sigma*(rt2[:,i]*dt)**0.5*randnormal[:,j]
        P=FaceV*np.exp(-np.sum(rt2[:,1:(NS-NT)],axis=1)*dt)
        Payoff[k]=np.maximum(np.mean(P)-K,0)
    price=np.mean(np.exp(-np.sum(rt1[:,1:NT]*dt,axis=1))*Payoff)
    return price

r0,sigma,kappa,rbar,T,S,FaceV,sim,K=0.05,0.18,0.92,0.055,0.5,1,1000,1000,980
price_2a=EuroCall_CIR(r0,sigma,kappa,rbar,T,S,FaceV,sim,K)
```
The European Call option on zero coupon bond is 1.1847.

Use explicit formula results to calculate the European Call option price on zero bond under CIR.

```python
def EuroCall_CIR_formula(r0,sigma,kappa,rbar,T,S,FaceV,K):
    h1=(kappa**2+2*sigma**2)**0.5
    h2=(kappa+h1)/2
    h3=2*kappa*rbar/(sigma**2)
    AT=(h1*np.exp(h2*T)/(h2*(np.exp(h1*T)-1)+h1))**h3
    BT=(np.exp(h1*T)-1)/(h2*(np.exp(h1*T)-1)+h1)
    AS=(h1*np.exp(h2*S)/(h2*(np.exp(h1*S)-1)+h1))**h3
    BS=(np.exp(h1*S)-1)/(h2*(np.exp(h1*S)-1)+h1)
    PS=FaceV*AS*np.exp(-BS*r0)
    PT=AT*np.exp(-BT*r0)
    theta=(kappa**2+2*sigma**2)**0.5
    phi=2*theta/(sigma**2*(np.exp(theta*T)-1))
    psi=(kappa+theta)/(sigma**2)
    ATS=(h1*np.exp(h2*(S-T))/(h2*(np.exp(h1*(S-T))-1)+h1))**h3
    BTS=(np.exp(h1*(S-T))-1)/(h2*(np.exp(h1*(S-T))-1)+h1)
    rstar=np.log(ATS*FaceV/K)/BTS
    price=PS*ncx2.cdf(2*rstar*(phi+psi+BTS),df=4*kappa*rbar/sigma**2,nc=2*phi**2*r0*np.exp(theta*T)/(phi+psi+BTS))\
                             -K*PT*ncx2.cdf(2*rstar*(phi+psi),df=4*kappa*rbar/sigma**2,nc=2*phi**2*r0*np.exp(theta*T)/(phi+psi))
    return price
   
price_2b=EuroCall_CIR_formula(r0,sigma,kappa,rbar,T,S,FaceV,K)
```
The European Call option on zero coupon bond is 1.1234. 
We can see that Monte Carlo Simulationg result and explicit formula result are almost the same.

### 3. Option pricing under G2++

Define bivariate normally distributed function.

```python
def binormal(miu1,miu2,sigma1,sigma2,rho,N1,N2,n,sim):    
    Z1=np.zeros((sim,n))
    Z2=np.zeros((sim,n))
    for i in range(n):
        Z1[:,i]=miu1+sigma1*N1[:,i]
        Z2[:,i]=miu2+sigma2*rho*N1[:,i]+sigma2*((1-rho**2)**0.5)*N2[:,i]
    return Z1,Z2
```

Use Monte Carlo simulation to price a European Put option on zero coupon bond under G2++.

```python
def G2PP(x0,y0,phi0,r0,rho,a,b,sigma,eta,phit,T,S,K,FaceV,sim):
    dt=1/365
    NS=int(S*365)
    NT=int(T*365)
    uniformrand=uniform(2*sim*NT)
    normalrand=normal(uniformrand,sim*NT)
    N1=normalrand[0:NT*sim]
    N2=normalrand[NT*sim:2*NT*sim]
    N1=np.reshape(N1,(sim,NT))
    N2=np.reshape(N2,(sim,NT))
    N1,N2=binormal(0,0,1,1,rho,N1,N2,NT,sim)
    xt1,yt1,rt1=np.zeros((sim,NT+1)),np.zeros((sim,NT+1)),np.zeros((sim,NT+1))
    xt1[:,0],yt1[:,0],rt1[:,0]=x0,y0,r0
    Payoff=np.zeros((sim,1))
    uniformrand=uniform(2*sim*(NS-NT))
    normalrand=normal(uniformrand,sim*(NS-NT))
    N3=normalrand[0:(NS-NT)*sim]
    N4=normalrand[(NS-NT)*sim:2*(NS-NT)*sim]
    N3=np.reshape(N3,(sim,NS-NT))
    N4=np.reshape(N4,(sim,NS-NT))
    N3,N4=binormal(0,0,1,1,rho,N3,N4,NS-NT,sim)
    for i in range(NT):
        xt1[:,i+1]=xt1[:,i]-a*xt1[:,i]*dt+sigma*dt**0.5*N1[:,i]
        yt1[:,i+1]=yt1[:,i]-b*yt1[:,i]*dt+eta*dt**0.5*N2[:,i]
        rt1[:,i+1]=xt1[:,i+1]+yt1[:,i+1]+phit
    for k in range(sim):
        xt2,yt2,rt2=np.zeros((sim,NS-NT+1)),np.zeros((sim,NS-NT+1)),np.zeros((sim,NS-NT+1))
        xt2[:,0]=xt1[k,NT]
        yt2[:,0]=yt1[k,NT]
        rt2[:,0]=rt1[k,NT]
        for j in range(NS-NT):
            xt2[:,j+1]=xt2[:,j]-a*xt2[:,j]*dt+sigma*dt**0.5*N3[:,j]
            yt2[:,j+1]=yt2[:,j]-b*yt2[:,j]*dt+eta*dt**0.5*N4[:,j]
            rt2[:,j+1]=xt2[:,j+1]+yt2[:,j+1]+phit
        P=FaceV*np.exp(-np.sum(rt2[:,1:(NS-NT)]*dt,axis=1))
        Payoff[k]=np.maximum(K-np.mean(P),0)
    price=np.mean(np.exp(-np.sum(rt1[:,1:NT]*dt,axis=1))*Payoff)
    return price
```

```python   
x0,y0,phi0,r0,rho,a,b,sigma,eta,phit,T,S,K,FaceV,sim=0,0,0.03,0.03,0.7,0.1,0.3,0.03,0.08,0.03,0.5,1,985,1000,1000
price_3=G2PP(x0,y0,phi0,r0,rho,a,b,sigma,eta,phit,T,S,K,FaceV,sim)
```

The price of a European Put option on zero coupon bond under G2++ is 12.1603.
