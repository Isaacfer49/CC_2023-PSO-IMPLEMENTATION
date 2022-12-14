# Programa una función usando PSO en paralelo para encontrar el minimo en una función. 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def particle_swarm_optimization(f, x0, n, c1, c2, w):
    #f es la función a minimizar. 
    #x0 es la posición inicial.
    #n es el numero de particulas.
    #c1 es el parametro cognitivo.
    #c2 es el parametro social. 
    #w es el parametro de inercia. 
    #regresa una lsita de posiciones en cada paso. 
    x = x0
    v = np.random.normal(0,1,(n,len(x0)))
    p = x + np.random.normal(0,1,(n,len(x0)))
    xs = [x]
    for i in range(100):
        for j in range(n):
            if f(p[j]) < f(x):
                x = p[j]
            if f(p[j]) < f(x):
                p[j] = x + np.random.normal(0,1,len(x0))
        v = w*v + c1*np.random.uniform(0,1,(n,len(x0)))*(p-x) + c2*np.random.uniform(0,1,(n,len(x0)))*(np.mean(p,0)-x)
        x = x + v
        xs.append(x)
    return xs

def f(x):
    return np.sum(x**2)

def parallel_particle_swarm_optimization(f, x0, n, c1, c2, w, n_jobs):
    #n_jobs es el numero de threads 
    pool = mp.Pool(n_jobs)
    xs = pool.map(particle_swarm_optimization, [f, x0, n, c1, c2, w])
    return xs

x0 = np.array([0,0])
n = 10
c1 = 2
c2 = 2
w = 0.5
n_jobs = 4
xs = parallel_particle_swarm_optimization(f,x0,n,c1,c2,w,n_jobs)
plt.plot(np.array(xs)[:,0],np.array(xs)[:,1])
plt.show()
