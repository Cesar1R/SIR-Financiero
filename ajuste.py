import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.integrate import odeint
import pygad

import ruptures as rpt


import time




def gen_particion(datos, pen=10, flag_print=False, model="l2"):
    model="l2"
    algo = rpt.Pelt(model=model).fit(datos)
    result = algo.predict(pen=pen)

    if flag_print:
        print("Cambios detectados: ", len(result))
        plt.figure(figsize=(10, 6))
        plt.plot(datos, 'b', label="Infectados")

        for cp in result:
            plt.axvline(x=cp, color='r', linestyle='--', label="Puntos de cambio" if cp==result[0] else "")

        plt.title("Cambios en la serie de tiempo")
        plt.xlabel("Tiempo")
        plt.ylabel("Infectados")
        plt.legend()
        plt.show()

    puntos = [0] + result + [len(datos)]

    segmentos = []
    for i in range(len(puntos) - 1):
        inicio = puntos[i]
        fin = puntos[i+1]
        segmentos.append(datos[inicio:fin])

    return segmentos

    

def plot_res_evolutivo(datos_I, m=10, generations=50, pop_size=30):

    # 1. Preparar datos y condiciones iniciales
    Nt = len(datos_I)
    t = np.linspace(0, Nt-1, Nt)
    I0 = datos_I[0]
    R0 = 0.0
    N = 34
    S0 = N - I0 - R0
    X0 = (S0, I0, R0)

    # 2. Defino la derivada con parámetros dependientes del tiempo
    def deriv(X, t, genes):
        S, I, R = X
        # Interpolación de a(t) y gamma_I(t)
        pts_t = np.linspace(0, Nt-1, m)
        a_t = np.interp(t, pts_t, genes[0:m])
        gamma_I_t = np.interp(t, pts_t, genes[m:2*m])
        # Constantes b, gamma_S, gamma_R, delta_, delta_L, delta_H
        b, gamma_S, gamma_R, delta_, delta_L, delta_H = genes[2*m:]
        dS = b - gamma_S*S - a_t*S*I + delta_L*R
        dI = a_t*S*I - (gamma_I_t + delta_)*I + delta_H*R
        dR = delta_*I - (gamma_R + delta_H + delta_L)*R
        return [dS, dI, dR]

    # 3. Fitness function con los 3 parámetros requeridos por PyGAD
    # Error cuadratico medio
    def fitness(ga_instance, solution, sol_idx):
        sim = odeint(deriv, X0, t, args=(solution,))
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
    sim = odeint(deriv, X0, t, args=(best_solution,))
    S_fit, I_fit, R_fit = sim.T

    print("Parametros:")
    print("a_t:", [f'{x:0.4f}' for x in best_solution[0:m]])
    print("gamma_I_t: ", [f'{x:0.4f}' for x in best_solution[m:2*m] ])
    print("b, gamma_S, gamma_R, delta_, delta_L, delta_H : ", [f'{x:0.4f}' for x in best_solution[2*m:]])

    index_ = np.linspace(0, len(datos_I)-1, len(datos_I))
    # 7. Gráfica de ajuste
    plt.figure(figsize=(8,4))
    plt.plot(index_, datos_I, 'o', label="Datos reales (I)", alpha=0.6)
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
    
    particion = gen_particion(datos_I, pen=50, flag_print=True)

    for segmento, i in zip(particion, range(1, len(particion)+1)):
        inicio = time.time()
        plot_res_evolutivo(segmento)
        fin = time.time()

        print(f"Tiempo de ajuste {fin - inicio:0.2f}s para el elemeto {i}")

    plot_res_evolutivo(datos_I)
    # except Exception as e:
    #     print(f"Error: {e}")


if __name__ == "__main__":
    main()

