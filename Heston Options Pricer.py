import numpy as np
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as impvol


#Parameters for generating the correlated Brownian Motions
steps=252 #252 trading days in a year
rho=-0.65
mu=np.array([0, 0])
sig=np.array([[1, rho],[rho, 1]])

#Parameters for simulating the volatility process
nu0=0.2  #Initial volatility
theta=0.2  #Long-term mean of the volatility
kappa=3 #Mean reversion rate
ksi=0.5 #Volatility of the volatility process
T=1 #Time until expiry of the option. Here we take T=1 year, and each step corresponds to 1 trading day
dt=T/steps #Split into equal time steps

#Parameters for simulating the underlying stock process
S0=100.0 #Initial stock price
r=0.001  #Risk-free rate
M=1000 #Number of simulations

def heston_model(steps, r, rho, S0, nu0, theta, kappa, ksi, T, M):
    dt=T/steps #Split into equal time steps

    #Parameters for generating the correlated Brownian Motions
    mu=np.array([0, 0])
    sig=np.array([[1, rho],[rho, 1]])

    #M pairs of Brownian Motion increments drawn from the bivariate normal distribution
    bminc=np.random.multivariate_normal(mu, sig, (steps, M))
    Wnuinc=np.squeeze(bminc[:,:,0]) #Increments of the BM in the volatility process.
    WSinc=np.squeeze(bminc[:,:,1]) #Increments of the BM in the stock price process.


    #Plot a pair of correlated BMs
    plt.figure(figsize=(10, 6))
    plt.plot(Wnuinc[:,0].cumsum(axis=0), label=r"$W^\nu_t$", linestyle='-', color='blue')
    plt.plot(WSinc[:,0].cumsum(axis=0), label=r"$W^S_t$", linestyle='-', color='orange')
    plt.xlabel("Time Steps")
    plt.ylabel("Brownian Motion")
    plt.title(rf"Correlated Brownian Motion, $\rho$={rho}")
    plt.legend(prop={'size': 15})
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    
    plt.tight_layout()
    plt.show()
    
        
    #Simulating the volatility process & the stock price process
    St=np.full(shape=(steps,M), fill_value=S0)
    nut=np.full(shape=(steps,M), fill_value=nu0)
    
    for i in range(1,steps):
        nut[i]=np.abs(nut[i-1] + kappa*(theta-nut[i-1])*dt + ksi*np.sqrt(nut[i-1]*dt)*Wnuinc[i-1,:])
        St[i]=St[i-1]+r*St[i-1]*dt+np.sqrt(nut[i-1]*dt)*St[i-1]*WSinc[i-1,:]
        
    return St, nut



t=np.linspace(0, T, steps)
St, nut = heston_model(steps, r, rho, S0, nu0, theta, kappa, ksi, T, M)

#Plot the volatility process along with the long-term mean
plt.figure(figsize=(10, 6))
plt.plot(t, nut)
plt.axhline(y=theta, color='black', linestyle="--", label=r"Long-term mean, $\theta$")
plt.legend()
plt.xlabel("Time")
plt.ylabel(r"Volatility $\nu_t$")
plt.title("Heston Volatility Paths")
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.tight_layout()
plt.show()
       
#Plot the stock process
plt.figure(figsize=(10, 6))
plt.plot(t, St, label="Stock Price")
plt.xlabel("Time")
plt.ylabel(r"Price $S(t)$")
plt.title("Heston Underlying Paths")
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.tight_layout()
plt.show()
        
ST=St[-1,:] #An array of the underlying price at maturity for each path
K=np.arange(50,200,2) #An array of strikes

#Option price under the risk-neutral measure is the expectation value
#of the payoff function, discounted to today
callprices=np.array([np.mean(np.exp(-r*T)*np.maximum(ST-k,0)) for k in K])
putprices=np.array([np.mean(np.exp(-r*T)*np.maximum(k-ST,0)) for k in K])
 
#Calculate implied volatilities under Black-Scholes & plot them as a function of strike price ,
#to show that the Heston model captures volatility smiles/smirks/skews.      
callimpvols=impvol(callprices, S0, K, T, r, flag="c", q=0, return_as="numpy")
putimpvols=impvol(putprices, S0, K, T, r, flag="p", q=0, return_as="numpy")

plt.figure(figsize=(10, 6))
plt.plot(K, callimpvols, label="Calls")
plt.plot(K, putimpvols, label="Puts")
plt.title("Implied Volatility as a Function of Strike Price")
plt.legend()
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.tight_layout()
plt.show()

#Compute implied volatilities as a function of time to maturity
Ts=np.arange(1,3,1/12)
volsurfcall=np.array([impvol(callprices, S0, K, t, r, flag="c", q=0, return_as="numpy") for t in Ts])
volsurfput=np.array([impvol(putprices, S0, K, t, r, flag="p", q=0, return_as="numpy") for t in Ts])

#Volatility surface for calls
X, Y = np.meshgrid(Ts, K, indexing="ij")
Z=volsurfcall
fig = plt.figure(figsize=(8,8), dpi=150)
ax = fig.add_subplot(111, projection='3d')
volsurface = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True)

ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Strike Price')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Implied Volatility', rotation=90)

ax.xaxis._axinfo['grid'].update(color='grey', linestyle='dotted')
ax.yaxis._axinfo['grid'].update(color='grey', linestyle='dotted')
ax.zaxis._axinfo['grid'].update(color='grey', linestyle='dotted')

ax.view_init(elev=20, azim=65)
plt.title("Heston Volatility Surface, Calls", y=1)
plt.show()

#Volatility surface for puts
X, Y = np.meshgrid(Ts, K, indexing="ij")
Z=volsurfput
fig = plt.figure(figsize=(8,8), dpi=150)
ax = fig.add_subplot(111, projection='3d')
volsurface = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True)

ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Strike Price')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Implied Volatility', rotation=90)

ax.xaxis._axinfo['grid'].update(color='grey', linestyle='dotted')
ax.yaxis._axinfo['grid'].update(color='grey', linestyle='dotted')
ax.zaxis._axinfo['grid'].update(color='grey', linestyle='dotted')

ax.view_init(elev=20, azim=65)
plt.title("Heston Volatility Surface, Puts", y=1)
plt.show()

        
        