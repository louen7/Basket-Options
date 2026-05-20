import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm

#q3

#si nécessaire, je crée une fonction pour l'approximation de la fonction de répartition de la gaussienne centrée réduite
#avec précision 7.5*10^-8

def repartition_normale(x):
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1/(1+b0*x)
    return 1 - (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)*(b1*t+b2*t**2+b3*t**3+b4*t**4+b5*t**5)

def price_call_BS(alpha,beta,r,rho,S1_0,S2_0,sigma1,sigma2,T,K):
    SB_0=alpha*S1_0+beta*S2_0
    rB=r
    sigmaB=np.sqrt((1/T)*np.log((alpha**2*S1_0**2*np.exp(sigma1**2*T)+beta**2*S2_0**2*np.exp(sigma2**2*T)+2*alpha*beta*S1_0*S2_0*np.exp(rho*sigma1*sigma2*T))/(SB_0)**2))

    d1 = (np.log(SB_0/K)+(rB+sigmaB**2/2)*T)/(sigmaB*np.sqrt(T))
    d2 = (np.log(SB_0/K)+(rB-sigmaB**2/2)*T)/(sigmaB*np.sqrt(T))
    #price = SB_0*norm.cdf(d1) - K*np.exp(-rB*T)*norm.cdf(d2)
    price = SB_0*repartition_normale(d1) - K*np.exp(-rB*T)*repartition_normale(d2)
    return price

#q4

def call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size):
    Gamma = np.array([[1,rho],[rho,1]])
    
    A=np.linalg.cholesky(Gamma)
    W = np.sqrt(T)* A @ G

    S1=S1_0*np.exp((r-sigma1**2/2)*T+sigma1*W[0])
    S2=S2_0*np.exp((r-sigma2**2/2)*T+sigma2*W[1])
    
    payoff = np.exp(-r*T)*np.maximum(alpha*S1+beta*S2-K,0)
    MC_price = np.mean(payoff)
    
    STD = np.std(payoff)
    error = 1.645*STD/np.sqrt(Sample_size)

    return MC_price, STD, error

#q5
def call_MC_red(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size):
    Gamma = np.array([[1,rho],[rho,1]])
    
    A=np.linalg.cholesky(Gamma)
    W = np.sqrt(T)* A @ G

    S1=S1_0*np.exp((r-sigma1**2/2)*T+sigma1*W[0])
    S2=S2_0*np.exp((r-sigma2**2/2)*T+sigma2*W[1])

    K_prime=(K-beta*S2)/alpha
    mask_pos = K_prime>0  #booléen
    mask_neg = ~mask_pos #opposé du booléen précédent
    sigma_cond = sigma1*np.sqrt(1-rho**2)
    F = S1_0*np.exp((r-0.5*(rho*sigma1)**2)*T+sigma1*rho*W[1]) #espérance de S1(T) sachant W2(T)

    K_pos = K_prime[mask_pos]
    F_pos = F[mask_pos]

    d1 = (np.log(F_pos/K_pos)+(sigma_cond**2/2)*T)/(sigma_cond*np.sqrt(T))
    d2 = d1 - sigma_cond*np.sqrt(T)
    
    price_cond_brut=np.zeros(Sample_size)
    price_cond_brut[mask_pos] = F_pos*norm.cdf(d1) - K_pos*norm.cdf(d2)
    price_cond_brut[mask_neg] = F[mask_neg] - K_prime[mask_neg]

    price_cond = np.exp(-r*T)*alpha*price_cond_brut
    MC_price_red = np.mean(price_cond)

    STD = np.std(price_cond)
    error = 1.645*STD/np.sqrt(Sample_size)

    return MC_price_red, STD, error


alpha = 1
beta = 1
S1_0 = 1
S2_0 = 1
r = 0.01
sigma1 = 0.35
sigma2 = 0.4
rho = 0.3
T = 2 #ans
K = 2
Sample_size = 1000000

#box müller 
def loi_normale(Sample_size):
    U = npr.uniform(size=Sample_size)
    V = npr.uniform(size=Sample_size)
    X = np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V)
    Y = np.sqrt(-2*np.log(U))*np.sin(2*np.pi*V)
    return X,Y

X,Y = loi_normale(Sample_size)
G = np.array([X, Y])

price_approx = price_call_BS(alpha,beta,r,rho,S1_0,S2_0,sigma1,sigma2,T,K)
[MC_price, STD, error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)
# [MC_price_red, STD_red, error_red] = call_MC_red(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)

print(price_approx)
print(MC_price)
# print(MC_price_red)
# print(STD)
# print(STD_red)

#q6 var_empiriques en fonction du nb de trajectoires
# n=1000
# liste_sample_size = np.linspace(1000,100000,n,dtype=int)
# liste_STD = np.zeros(n)
# liste_STD_red = np.zeros(n)
# for i in range(n):
#     [MC_price,liste_STD[i],error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,liste_sample_size[i])
#     [MC_price_red, liste_STD_red[i], error_red] = call_MC_red(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,liste_sample_size[i])

# var_empiriques = [liste_STD[i]**2/liste_sample_size[i] for i in range(n)]
# var_empiriques_red = [liste_STD_red[i]**2/liste_sample_size[i] for i in range(n)]

# plt.plot(liste_sample_size,var_empiriques)
# plt.plot(liste_sample_size,var_empiriques_red)
# plt.show()

#q6 estimateurs de P en fonction du nb de trajectoires
# n=200
# liste_sample_size = np.linspace(1000,100000,n,dtype=int)
# liste_price = np.zeros(n)
# liste_price_red = np.zeros(n)

# liste_STD = np.zeros(n)
# liste_STD_red = np.zeros(n)

# IC_inf=np.zeros(n)
# IC_sup=np.zeros(n)
# IC_inf_red=np.zeros(n)
# IC_sup_red=np.zeros(n)

# for i in range(n):
#     [liste_price[i],liste_STD[i],error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,liste_sample_size[i])
#     [liste_price_red[i], liste_STD_red[i], error_red] = call_MC_red(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,liste_sample_size[i])
#     IC_inf[i]=liste_price[i]-1.645*liste_STD[i]/np.sqrt(liste_sample_size[i])
#     IC_sup[i]=liste_price[i]+1.645*liste_STD[i]/np.sqrt(liste_sample_size[i])
#     IC_inf_red[i]=liste_price_red[i]-1.645*liste_STD_red[i]/np.sqrt(liste_sample_size[i])
#     IC_sup_red[i]=liste_price_red[i]+1.645*liste_STD_red[i]/np.sqrt(liste_sample_size[i])



color1 = "#980FB4"
color2 = "#C597F4"
color3 = "#0D7219"
color4 = "#B5E77C"
color5 = "#E00505"

# #approximation log normale 
# price_approx = price_call_BS(alpha,beta,r,rho,S1_0,S2_0,sigma1,sigma2,T,K)
# liste_price_approx = [price_approx for _ in range(n)]
# plt.plot(liste_sample_size, liste_price_approx, color = color5, label = 'Approximation log-normale')
# plt.fill_between(liste_sample_size,IC_inf,IC_sup,color=color2, alpha=0.2,label='IC standard 90%')
# plt.plot(liste_sample_size,liste_price, color = color1, label = 'Estimateur MC standard')
# plt.fill_between(liste_sample_size,IC_inf_red,IC_sup_red,color=color4, alpha=0.4,label='IC réduit 90%')
# plt.plot(liste_sample_size,liste_price_red, color = color3, label = 'Estimateur MC réduit')
# plt.legend()
# plt.show()

#q7
# n=50
# liste_rho = np.linspace(-0.99,0.99,n)
# liste_price_approx = np.zeros(n)
# liste_price_red = np.zeros(n)

# for i in range(n):
#     liste_price_approx[i] = price_call_BS(alpha,beta,r,liste_rho[i],S1_0,S2_0,sigma1,sigma2,T,K)
#     [liste_price_red[i], STD_red, error_red] = call_MC_red(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,liste_rho[i],T,K,Sample_size)

# #plt.plot(liste_rho,liste_price_approx, color=color1, label = 'Approximation log-normale')
# #plt.plot(liste_rho,liste_price_red, color=color3, label = 'Prix avec variance réduite')
# plt.plot(liste_rho,liste_price_approx-liste_price_red,color=color5, label = 'Différence des estimateurs')
# plt.legend()
# plt.show()

#q8
# n=100
# liste_alpha = np.linspace(0.01,2,n)
# liste_price_approx = np.zeros(n)
# liste_price_red = np.zeros(n)

# for i in range(n):
#     liste_price_approx[i] = price_call_BS(liste_alpha[i],beta,r,rho,S1_0,S2_0,sigma1,sigma2,T,K)
#     [liste_price_red[i], STD_red, error_red] = call_MC_red(G,liste_alpha[i],beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)

# #les 2 plots sont superposés: seule la diff des 2 est intéressante
# #plt.plot(liste_alpha,liste_price_approx, color=color1, label = 'Approximation log-normale')
# #plt.plot(liste_alpha,liste_price_red, color=color3, label = 'Prix avec variance réduite')
# plt.plot(liste_alpha,liste_price_approx-liste_price_red,color=color5, label = 'Différence des estimateurs')
# plt.legend()
# plt.show()

#q9
# n=100
# liste_K = np.linspace(1,3,n)
# liste_price_approx = np.zeros(n)
# liste_price_red = np.zeros(n)
# liste_STD_red = np.zeros(n)
# IC_inf_red = np.zeros(n)
# IC_sup_red = np.zeros(n)

# for i in range(n):
#     liste_price_approx[i] = price_call_BS(alpha,beta,r,rho,S1_0,S2_0,sigma1,sigma2,T,liste_K[i])
#     [liste_price_red[i], liste_STD_red[i], error_red] = call_MC_red(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,liste_K[i],Sample_size)
#     IC_inf_red[i]=liste_price_red[i]-1.645*liste_STD_red[i]/np.sqrt(Sample_size)
#     IC_sup_red[i]=liste_price_red[i]+1.645*liste_STD_red[i]/np.sqrt(Sample_size)

# #plt.fill_between(liste_K,IC_inf_red,IC_sup_red,color=color2, alpha=0.4,label='IC 90%')
# #plt.plot(liste_K,liste_price_red,color = color1,label = 'Prix avec variance réduite')
# #plt.plot(liste_K,liste_price_approx,color=color3,label = 'Approximation log-normale')
# plt.plot(liste_K,liste_price_approx-liste_price_red,color=color5, label = 'Différence des estimateurs')
# plt.legend()
# plt.show()

#q10

def call_MC_red_control(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size):
    Gamma = np.array([[1,rho],[rho,1]])
    
    A=np.linalg.cholesky(Gamma)
    W = np.sqrt(T)* A @ G

    S1=S1_0*np.exp((r-sigma1**2/2)*T+sigma1*W[0])
    S2=S2_0*np.exp((r-sigma2**2/2)*T+sigma2*W[1])
    
    payoff_put = np.exp(-r*T)*np.maximum(K-(alpha*S1+beta*S2),0)  #on calcule le put car ATM variance plus faible (bcp de payoff = 0 donc plus stable)
    MC_price_put = np.mean(payoff_put)
    #on en déduit le prix du call grâce à la "parité call-put"
    MC_price = MC_price_put + alpha*S1_0+beta*S2_0 - K*np.exp(-r*T)
    
    STD = np.std(payoff_put) #la variance est inchangée car on ajoute juste une constante
    error = 1.645*STD/np.sqrt(Sample_size)

    return MC_price, STD, error

# [MC_price_red_control, STD_red_control, error_red_control] = call_MC_red_control(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)
#[MC_price, STD, error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)

# print(MC_price)
# print(MC_price_red_control)
# print(STD)
# print(STD_red_control)

#tracé des STD des 2 méthodes en fonction du strike K (le paramètre discriminant l'efficacité)
# n= 100
# liste_K = np.linspace(1,3,n)
# liste_STD = np.zeros(n)
# liste_STD_red_control = np.zeros(n)

# for i in range(n):
#     [MC_price, liste_STD[i], error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,liste_K[i],Sample_size)
#     [MC_price_red_control, liste_STD_red_control[i], error_red_control] = call_MC_red_control(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,liste_K[i],Sample_size)

# plt.plot(liste_K,liste_STD,color=color1,label='STD MC classique')
# plt.plot(liste_K,liste_STD_red_control, color=color3, label='STD par variable de contrôle')
# plt.legend()
# plt.show()

#q11
# alpha = 0.5
# beta = 0.5
# rho = 0.5

def delta1_calc_MC(G,alpha,beta,r,liste_S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size,n):
    #on note P pour MC_price
    P = np.zeros(n)
    delta1 = np.zeros(n)
    for i in range(n):
        [P[i], STD, error] = call_MC(G,alpha,beta,r,liste_S1_0[i],S2_0,sigma1,sigma2,rho,T,K,Sample_size)
    delta1 = [(P[i+1]-P[i-1])/(liste_S1_0[i+1]-liste_S1_0[i-1]) for i in range(1,n-1)]
    return delta1

def delta1_calc_approx(alpha,beta,r,rho,liste_S1_0,S2_0,sigma1,sigma2,T,K,n):
    rB=r
    P = np.zeros(n)
    delta1 = np.zeros(n)
    for i in range(n):
        S1_0 = liste_S1_0[i]
        SB_0=alpha*S1_0+beta*S2_0
        sigmaB=np.sqrt((1/T)*np.log((alpha**2*S1_0**2*np.exp(sigma1**2*T)+beta**2*S2_0**2*np.exp(sigma2**2*T)+2*alpha*beta*S1_0*S2_0*np.exp(rho*sigma1*sigma2*T))/(SB_0)**2))

        d1 = (np.log(SB_0/K)+(rB+sigmaB**2/2)*T)/(sigmaB*np.sqrt(T))
        d2 = (np.log(SB_0/K)+(rB-sigmaB**2/2)*T)/(sigmaB*np.sqrt(T))
        P[i] = SB_0*norm.cdf(d1) - K*np.exp(-rB*T)*norm.cdf(d2)
    delta1 = [(P[i+1]-P[i-1])/(liste_S1_0[i+1]-liste_S1_0[i-1]) for i in range(1,n-1)]  
    return delta1

# n = 200
# liste_S1_0 = np.linspace(0.1,5,n)
# delta1_approx = delta1_calc_approx(alpha,beta,r,rho,liste_S1_0,S2_0,sigma1,sigma2,T,K,n)
# delta1_MC_pos = delta1_calc_MC(G,alpha,beta,r,liste_S1_0,S2_0,sigma1,sigma2,0.5,T,K,Sample_size,n)
# delta1_MC_neg = delta1_calc_MC(G,alpha,beta,r,liste_S1_0,S2_0,sigma1,sigma2,-0.5,T,K,Sample_size,n)
# liste_S1_0_tronquee = [liste_S1_0[i] for i in range(1,n-1)]
# # plt.plot(liste_S1_0_tronquee,delta1_MC_neg,color=color2,label='Delta1 MC rho=-0.5')
# # plt.plot(liste_S1_0_tronquee,delta1_MC_pos,color=color4,label='Delta1 MC rho = 0.5')
# # plt.plot(liste_S1_0_tronquee,delta1_approx,color=color3,label='Delta1 log-normale')
# diff_delta1 = [delta1_approx[i]-delta1_MC_pos[i] for i in range(n-2)]
# plt.plot(liste_S1_0_tronquee,diff_delta1,color=color2,label='Différence des Delta1')
# plt.legend()
# plt.show()

#vouloir augmenter la précision n augmente le bruit des simulations de dérivées numériques
#à partir d'ici j'ai inclus G comme paramètre (et modifié dans les précédentes fonctions)
#comme ça entre deux itérations de dérivée numérique je garde la même simulation

#q12

def call_MC_red_control_geo(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size):
    mu = alpha*(np.log(S1_0)+(r-0.5*sigma1**2)*T)+beta*(np.log(S2_0)+(r-0.5*sigma2**2)*T)
    sigma = np.sqrt((alpha**2*sigma1**2+beta**2*sigma2**2+2*alpha*beta*sigma1*sigma2*rho)*T)

    Gamma = np.array([[1,rho],[rho,1]])
    
    A=np.linalg.cholesky(Gamma)
    W = np.sqrt(T)* A @ G

    S1=S1_0*np.exp((r-sigma1**2/2)*T+sigma1*W[0])
    S2=S2_0*np.exp((r-sigma2**2/2)*T+sigma2*W[1])

    S_arithm = alpha*S1+beta*S2
    S_geo = np.exp(alpha*np.log(S1)+beta*np.log(S2))

    payoff_arithm = np.exp(-r*T)*np.maximum(S_arithm-K,0)

    #payoff du call géométrique par BC
    d1 = (mu-np.log(K)+sigma**2)/sigma
    d2 = (mu-np.log(K))/sigma
    payoff_geo = np.exp(-r*T)*np.maximum(S_geo-K,0)

    #on fait la régression linéaire du call arithmétique sur le call géométrique
    covariance = np.cov(payoff_arithm, payoff_geo)[0,1] #temps de calcul négligeable
    var_geo = np.var(payoff_geo)
    c = covariance/var_geo

    payoff_red = payoff_arithm - c*(payoff_geo - np.exp(-r*T)*(np.exp(mu+0.5*sigma**2)*norm.cdf(d1) - K*norm.cdf(d2)))

    MC_price = np.mean(payoff_red)
    STD = np.std(payoff_red)
    error = error = 1.645*STD/np.sqrt(Sample_size)
    return MC_price, STD, error

# [MC_price_red_control, STD_red_control, error_red_control] = call_MC_red_control(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)
# [MC_price_red_control_geo, STD_red_control_geo, error_red_control_geo] = call_MC_red_control_geo(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)
# [MC_price, STD, error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,K,Sample_size)

# print(MC_price)
# print(MC_price_red_control)
# print(MC_price_red_control_geo)
# print(STD)
# print(STD_red_control)
# print(STD_red_control_geo)

#si strike dans la monnaie : un put comme variable de contrôle est plus efficace 
#si en dehors de la monnaie : le call géométrique est plus efficace

#tracé des STD des 2 méthodes en réduction par variables de contrôle, en fonction de K
# n= 100
# liste_K = np.linspace(1,3,n)
# liste_STD = np.zeros(n)
# liste_STD_red_control_geo = np.zeros(n)
# liste_STD_red_control = np.zeros(n)

# for i in range(n):
#     [MC_price, liste_STD[i], error] = call_MC(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,liste_K[i],Sample_size)
#     [MC_price_red_control_geo, liste_STD_red_control_geo[i], error_red_control_geo] = call_MC_red_control_geo(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,liste_K[i],Sample_size)
#     [MC_price_red_control, liste_STD_red_control[i], error_red_control] = call_MC_red_control(G,alpha,beta,r,S1_0,S2_0,sigma1,sigma2,rho,T,liste_K[i],Sample_size)


# plt.plot(liste_K,liste_STD, color=color4, label='STD classique')
# #plt.plot(liste_K,liste_STD_red_control, color=color3, label='STD avec put')
# plt.plot(liste_K,liste_STD_red_control_geo,color=color1,label='STD avec call géométrique')
# plt.legend()
# plt.show()


    

