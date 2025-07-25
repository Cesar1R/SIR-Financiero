import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.interpolate import BSpline
from scipy.integrate import odeint
import pygad 

import math

np.random.seed(123)




def eval_bspline(
    t:np.ndarray,
    coeffs: np.ndarray,
    t_min: float, 
    t_max: float,
    degree: int =3
) -> np.ndarray:
    k = degree
    n_coeffs = len(coeffs)

    n_internal = n_coeffs - (k + 1)
    if n_internal < 0:
        raise ValueError("La cantidad de coeficientes es insuficiente")

    internal_knots = np.linspace(t_min, t_max, n_internal+2)[1:-1]

    knots = np.concatenate((
        np.repeat(t_min, k+1),
        internal_knots,
        np.repeat(t_max, k+1)
    ))

    spline = BSpline(knots, coeffs, k, extrapolate=True)

    return spline(t)

def plot_res(datos_I: np.ndarray, m: int = 10, generations: int = 50, pop_size: int = 30, N: int = 34) -> None:
    if len(datos_I) == 0:
        return;

    Nt = len(datos_I)
    T = Nt - 1
    dt = T / Nt
    t = np.linspace(0, T, Nt)

    I0 = datos_I[0]
    R0 = 0.0
    S0 = N - I0 - R0

    def simulate_sir_euler_maruyama(genes):
        coeffs_a = genes[0:m]
        coeffs_gammaI = genes[m:2*m]
        b, gamma_S, gamma_R, delta_, delta_L, delta_H = genes[2*m:]

        X = np.zeros((Nt, 3))
        X[0] = [S0, I0, R0]
        val = 10
        dBt = np.clip(np.random.normal(loc=0.0, scale=np.sqrt(dt), size=Nt), -val, val)


        for i in range(1, Nt):
            S, I, R = X[i - 1]

            a_t = eval_bspline(t[i], coeffs_a, t[0], t[-1], degree=3)
            gamma_I_t = eval_bspline(t[i], coeffs_gammaI, t[0], t[-1], degree=3)
            dB = dBt[i]

            dS = (b - gamma_S*S - a_t*S*I + delta_L*R) * dt + delta_L * dB
            dI = (a_t*S*I - (gamma_I_t + delta_)*I + delta_H*R) * dt + delta_H * dB
            dR = (delta_*I - (gamma_R + delta_H + delta_L)*R) * dt - (delta_L + delta_H) * dB

            X[i] = [
            max(0, min(N, S + dS)),
            max(0, min(N, I + dI)),
            max(0, min(N, R + dR))
            ]


        return X


    def fitness(ga_instance, solution, sol_idx):
        sim = simulate_sir_euler_maruyama(solution)
        I_pred = sim[:, 1]
        mse = np.mean((I_pred - datos_I) ** 2)
        return -mse

    num_genes = 2*m + 6
    ga = pygad.GA(
        num_generations=generations,
        sol_per_pop=pop_size,
        num_parents_mating=pop_size // 2,
        fitness_func=fitness,
        num_genes=num_genes,
        gene_space=[{'low': 0.0, 'high': 1.0}] * num_genes,
        mutation_percent_genes=10,
        suppress_warnings=True
    )


    ga.run()
    best_solution, best_fitness, _ = ga.best_solution()

    sim = simulate_sir_euler_maruyama(best_solution)
    S_fit, I_fit, R_fit = sim.T

    print("Parametros:")
    print("a_t:", [f'{x:0.4f}' for x in best_solution[0:m]])
    print("gamma_I_t:", [f'{x:0.4f}' for x in best_solution[m:2*m]])
    print("b, gamma_S, gamma_R, delta_, delta_L, delta_H:",
          [f'{x:0.4f}' for x in best_solution[2*m:]])


    plt.figure(figsize=(8, 4))
    index_ = np.linspace(0, Nt - 1, Nt)
    plt.plot(index_, datos_I, 'b', label="Datos reales (I)", alpha=0.6)
    plt.plot(index_, I_fit, '-', color='red', label="Ajuste evolutivo (I)")
    plt.title("Ajuste estocástico SIR por Euler–Maruyama")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Número de infectados")
    plt.ylim(0, N)
    plt.grid(True)
    plt.legend()
    plt.show()




def main():
    archivo = "resumen.csv"
    df = pd.read_csv(archivo)
    datos_I  = df["Infectados"].values
    
    plot_res(datos_I)


if __name__ == "__main__":
    main()

