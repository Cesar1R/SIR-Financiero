# SIR-Financiero

Codigos de un modelo SIR estocastico y uno determinista. Ambos buscan captar las tendencias generales de una serie de tiempo en el contexto financiero. 

Los modelos cuentan con parametros fijos y parametros que dependen del tiempo, se ajustan por medio de un algoritmo genetico clasico. Los algoritmos que dependen del tiempo se ajustan con B-spline cubico. 

## Modelo Determinista
Se basa en un modelo SIR modificado, considera solo los parametros mencionados antes. 

Los resultados del ajuste de la curva de infectados resultan bastante satisfactorios al captar la tendencia general de los datos de interés.

## Modelo Estocastico
Se utilizo un movimiento browniano para considerar el factor estocastico. Este movimiento genera perturbaciones que no captan de manera fidedigna los movimientos de la serie de tiempo de interés. 

