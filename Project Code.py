import numpy as np
from scipy.integrate import odeint
import random, math
import matplotlib.pyplot as plt

# ============================
# 1. Piezoelectric model (ODE)
# ============================

def piezo_model(y, t, m, c, k, theta, Cp, R, a_func):
    x, xdot, v = y
    a = a_func(t)
    xddot = (-c*xdot - k*x - theta*v - m*a)/m
    vdot = (theta*xdot - v/R)/Cp
    return [xdot, xddot, vdot]

def piezo_power_series(tspan, a_func,
                       m=0.02, c=0.02, k=1200, theta=0.015, Cp=22e-9, R=1e6):
    y0 = [0,0,0]
    sol = odeint(piezo_model,y0,tspan,args=(m,c,k,theta,Cp,R,a_func))
    v = sol[:,2]
    i = v/R
    p = v*i
    return p

# ============================
# 2. Thermoelectric model
# ============================

def thermo_power_series(S, R_int, delta_T_series):
    return (S**2 * delta_T_series**2)/(4*R_int)

# ============================
# 3. Solar model (diode)
# ============================

def solar_power_series(Iph, I0, n, Vt, V_series, irradiance_series, angle_series):
    I_series = Iph*irradiance_series*np.cos(np.deg2rad(angle_series)) - I0*(np.exp(V_series/(n*Vt))-1)
    return V_series*I_series

# ============================
# 4. Load model
# ============================

def load_energy(P_total_series, dt, C_storage=0.01, R_load=100, eff_dc=0.9):
    Vc = 1e-3
    E_total = 0
    for P in P_total_series:
        I = (P*eff_dc)/max(Vc,1e-3)
        Vc += (I*dt)/C_storage
        I_load = Vc/R_load
        E_total += Vc*I_load*dt
    return E_total

# ============================
# 5. Cost function
# ============================

def cost_function(params, density=7800, price_area=100, price_thickness=50, price_kth=200):
    thickness, area, angle, k_th = params
    volume = area*thickness
    weight = density*volume
    return price_area*area + price_thickness*thickness + price_kth*k_th + 0.01*weight

# ============================
# 6. Stability metric (multi-level smoothing)
# ============================

def stability_metric(P_total_series, short_window=20, long_window=100):
    short = np.convolve(P_total_series, np.ones(short_window)/short_window, mode='valid')
    long = np.convolve(P_total_series, np.ones(long_window)/long_window, mode='valid')
    return (np.std(short) + np.std(long))/2

# ============================
# 7. Compute total energy
# ============================

def compute_total_energy(params,data):
    thickness, area, angle, k_th = params
    if thickness<0.3 or area<0.001 or angle<0 or angle>90 or k_th<0.01 or k_th>0.2:
        return -1e9, None, None

    tspan=data["tspan"]; dt=data["dt"]

    P_piezo = piezo_power_series(tspan, data["a_func"])
    P_thermo = thermo_power_series(data["S"],data["R_int"],data["delta_T"])
    angle_series = np.ones(len(tspan))*angle
    P_solar = solar_power_series(data["Iph"],data["I0"],data["n"],data["Vt"],
                                 data["V_series"],data["irradiance"],angle_series)

    P_total = P_piezo + P_thermo + P_solar
    E_total = load_energy(P_total,dt)

    # سقف فيزيائي واقعي
    E_total = np.clip(E_total,0,0.5)

    return E_total,P_total,cost_function(params)

def compute_metrics(params,data):
    E_total,P_total,cost = compute_total_energy(params,data)
    if E_total<0: return {"Params":params,"Invalid":True}
    stability = stability_metric(P_total)
    return {
        "Params": params,
        "Total Energy (J)": round(E_total,4),
        "Cost": round(cost,2),
        "Energy/Cost": round(E_total/cost,6),
        "Stability (σ)": round(stability,4)
    }

# ============================
# 8. Unified fitness function
# ============================

def fitness(params,data,alpha=0.7,beta=0.3):
    E_total,P_total,cost = compute_total_energy(params,data)
    if E_total<0: return -1e9
    stability = stability_metric(P_total)
    score = alpha*(E_total/cost) - beta*stability
    # Adaptive diversity control
    if cost < 20 and stability > 0.05:
        score -= 0.5
    return score

# ============================
# 9. Optimizers (PSO, DE, GA, Hybrid)
# ============================

class Optimizer:
    def __init__(self,pop_size=30,iters=50,mode="PSO"):
        self.pop_size=pop_size
        self.iters=iters
        self.mode=mode

    def init_pop(self):
        return [[random.uniform(0.3,1.5),
                 random.uniform(0.001,0.03),
                 random.uniform(0,90),
                 random.uniform(0.01,0.2)]
                for _ in range(self.pop_size)]

    def optimize(self,data):
        pop=self.init_pop()
        best=None; best_fit=-1e9
        inertia=0.9
        for it in range(self.iters):
            new_pop=[]
            for ind in pop:
                fit=fitness(ind,data)
                if fit>best_fit:
                    best_fit=fit; best=ind
                mutant=[ind[d]+random.uniform(-0.05,0.05) for d in range(4)]
                if self.mode=="Hybrid":
                    inertia *= 0.99
                    if random.random()<0.2:
                        mutant=[best[d]+inertia*(ind[d]-best[d]) for d in range(4)]
                    if random.random()<0.3:
                        partner=random.choice(pop)
                        d=random.randint(0,3)
                        mutant[d]=partner[d]
                    if random.random()<0.3:
                        a,b,c=random.sample(pop,3)
                        F=0.8
                        d=random.randint(0,3)
                        mutant[d]=a[d]+F*(b[d]-c[d])
                else:
                    if random.random()<0.3:
                        d=random.randint(0,3)
                        if d==0: mutant[d]=random.uniform(0.3,1.5)
                        elif d==1: mutant[d]=random.uniform(0.001,0.03)
                        elif d==2: mutant[d]=random.uniform(0,90)
                        else: mutant[d]=random.uniform(0.01,0.2)
                new_pop.append(mutant)
            pop=new_pop
        return best,best_fit

# ============================
# 10. Main comparison
# ============================

def run_all_cases(data):
    results={}
    baseline=[0.8,0.02,30,0.1]
    results["Baseline"]=compute_metrics(baseline,data)
    results["PSO"]=compute_metrics(Optimizer(mode="PSO").optimize(data)[0],data)
    results["DE"]=compute_metrics(Optimizer(mode="DE").optimize(data)[0],data)
    results["GA"]=compute_metrics(Optimizer(mode="GA").optimize(data)[0],data)
    results["Hybrid"]=compute_metrics(Optimizer(mode="Hybrid").optimize(data)[0],data)
    return results

# ============================
# 11. Simulation data
# ============================

tspan=np.linspace(0,10,1000)
data={
    "tspan":tspan,
    "dt":tspan[1]-tspan[0],
    "a_func":lambda t:0.1*np.sin(2*np.pi*5*t)+0.05*np.sin(2*np.pi*12*t),
    "S":210e-6,
    "R_int":1.2,
    "delta_T":np.clip(15+2*np.sin(0.1*tspan)+0.5*np.random.randn(len(tspan)),5,25),
    "Iph":0.03,
    "I0":1e-10,
    "n":1.3,
    "Vt":0.0258,
   "V_series":0.5*np.ones(len(tspan)),
    "irradiance":np.clip(600+200*np.sin(0.05*tspan)+50*np.random.randn(len(tspan)),300,1000)
}

# ============================
# 12. Run system
# ============================

results = run_all_cases(data)
print("نتائج المقارنة بين الحالات:\n")
for name,res in results.items():
    print(name,":",res)

# ============================
# 13. Plot comparison
# ============================

labels = list(results.keys())
energies = [res["Total Energy (J)"] for res in results.values() if "Total Energy (J)" in res]
ratios = [res["Energy/Cost"] for res in results.values() if "Energy/Cost" in res]
stabilities = [res["Stability (σ)"] for res in results.values() if "Stability (σ)" in res]

plt.figure(figsize=(15,6))

# الطاقة الكلية
plt.subplot(1,3,1)
plt.bar(labels, energies, color=['gray','blue','green','orange','purple'])
plt.ylabel("الطاقة الكلية (J)")
plt.title("مقارنة الطاقة الكلية")

# الطاقة/التكلفة
plt.subplot(1,3,2)
plt.bar(labels, ratios, color=['gray','blue','green','orange','purple'])
plt.ylabel("الطاقة/التكلفة")
plt.title("كفاءة الطاقة مقابل التكلفة")

# الاستقرار
plt.subplot(1,3,3)
plt.bar(labels, stabilities, color=['gray','blue','green','orange','purple'])
plt.ylabel("الاستقرار (σ)")
plt.title("مقارنة الاستقرار")

plt.tight_layout()
plt.show()