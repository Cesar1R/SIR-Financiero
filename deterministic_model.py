import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.interpolate import BSpline
from scipy.integrate import odeint
import pygad

import time



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
def plot_res_evolutivo(datos_I:np.ndarray, m:int =10, generations:int =50, pop_size:int =30) -> None:
    if len(datos_I) ==0:
        return;

    # 1. Preparar datos y condiciones iniciales
    Nt = len(datos_I)
    t = np.linspace(0, Nt-1, Nt)
    I0 = datos_I[0]
    R0 = 0.0
    N = 34
    S0 = N - I0 - R0
    X0 = (S0, I0, R0)

    t_min =t[0]
    t_max =t[-1]
    # 2. Defino la derivada con parámetros dependientes del tiempo
    def deriv(X, t, genes, t_min, t_max):
        S, I, R = X
        
        n_basis = m 
        coeffs_a = genes[0:m]
        coeffs_gammaI = genes[m: 2 * m]

        # Interpolación de a(t) y gamma_I(t)
        a_t = eval_bspline(t, coeffs_a, t_min, t_max, degree=3)
        gamma_I_t = eval_bspline(t, coeffs_gammaI, t_min, t_max, degree=3)

        # Constantes b, gamma_S, gamma_R, delta_, delta_L, delta_H
        b, gamma_S, gamma_R, delta_, delta_L, delta_H = genes[2*m:]
        dS = b - gamma_S*S - a_t*S*I + delta_L*R
        dI = a_t*S*I - (gamma_I_t + delta_)*I + delta_H*R
        dR = delta_*I - (gamma_R + delta_H + delta_L)*R
        return [dS, dI, dR]

    # 3. Fitness function con los 3 parámetros requeridos por PyGAD
    # Error cuadratico medio
    def fitness(ga_instance, solution, sol_idx):
        sim = odeint(deriv, X0, t, args=(solution,t_min, t_max))
        I_pred = sim[:, 1]
        mse = np.mean((I_pred - datos_I) ** 2)
        return -mse

    # 4. Configuración del GA
    num_genes = 2*m + 6
    ga = pygad.GA(
        num_generations=generations,
        sol_per_pop=pop_size,
        num_parents_mating=pop_size//2,
        fitness_func=fitness,
        num_genes=num_genes,
        gene_space=[{'low': 0.0, 'high': 1.0}] * num_genes,
        mutation_percent_genes=10,
        suppress_warnings=True
    )

    # 5. Ejecutar GA y extraer la mejor solución
    ga.run()
    best_solution, best_fitness, _ = ga.best_solution()
    
    # 6. Simulación final con parámetros óptimos
    sim = odeint(deriv, X0, t, args=(best_solution,t_min, t_max))
    S_fit, I_fit, R_fit = sim.T

    print("Parametros:")
    print("a_t:", [f'{x:0.4f}' for x in best_solution[0:m]])
    print("gamma_I_t: ", [f'{x:0.4f}' for x in best_solution[m:2*m] ])
    print("b, gamma_S, gamma_R, delta_, delta_L, delta_H : ", [f'{x:0.4f}' for x in best_solution[2*m:]])

    index_ = np.linspace(0, len(datos_I)-1, len(datos_I))
    # 7. Gráfica de ajuste
    plt.figure(figsize=(8,4))
    plt.plot(index_, datos_I, 'b', label="Datos reales (I)", alpha=0.6)
    plt.plot(index_, I_fit, '-', color='red', label="Ajuste evolutivo (I)")
    plt.title("Ajuste SIR con parámetros dependientes del tiempo")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Número de infectados")
    plt.ylim(0, N)
    plt.grid(True)
    plt.legend()
    plt.show()




def main():
    # try:
    archivo = "resumen.csv"
    df = pd.read_csv(archivo)
    datos_I  = df["Infectados"].values
    
    plot_res_evolutivo(datos_I)



if __name__ == "__main__":
    main()

