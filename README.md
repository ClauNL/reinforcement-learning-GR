# RECONOCIMIENTO DE METAS MEDIANTE APRENDIZAJE POR REFUERZO

## Ejecución 


`<folder>` : directorio con


- `domain.pddl`: fichero con la definición del dominio en PDDL.
- `problems/` : directorio con un problema en formato PDDL para cada una de las metas. Por ejemplo: `problem0.pddl, problem1.pddl`.
- `obs<porcentaje>.pkl`: fichero con la secuencia de observaciones de la forma estado/acción. 
La etiqueta <porcentaje> en el nombre debe sustituirse por el porcentaje de observabilidad deseado, por ejemplo: `obs0.5.pkl, obs1.0.pkl`



### Módulo Q-Learning

`./train.py <folder>`

Ejemplo

`./train.py  blocks_gr`


### Módulo Reconocimiento de metas

`./recognize.py  <folder>  <metric>  <obs>`

- `<metric>`: MaxUtil, KL, DP
- `<obs>`: 0.1, 0.5, 1.0, etc



Ejemplo

`./recognize.py   blocks_gr   KL  1.0`



