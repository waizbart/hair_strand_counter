import scipy.optimize as opt
import numpy as np

def FO(x):
    x1, x2, x3, x4 = x # Para facilitar a escrita, e para percebermos que x é um vetor
    print("x1 =", x1)
    result = x1*x4*(x1+x2+x3) + x3 # Expressão da FO
    return result

fronteira = opt.Bounds(0, 5)

# Definindo a função da restrição de desigualdade
def F_rest_ineq(x):
    x1, x2, x3, x4 = x
    F1 = x1*x2*x3*x4
    return F1

# Definindo a função da restrição de igualdade
def F_rest_eq(x):
    x1, x2, x3, x4 = x
    F2 = x1**2 + x2**2 + x3**2 + x4**2
    return F2

# Definindo as restrições com os limites inferior e superior
Restri_ineq = opt.NonlinearConstraint(F_rest_ineq, 25, np.inf)
Restri_eq = opt.NonlinearConstraint(F_rest_eq, 40, 40)

Restri = [Restri_ineq, Restri_eq]

x0 = np.array([1,5,5,1])

Solucao = opt.minimize(FO, x0, bounds=fronteira)

print("\n")
print("A solução ótima é:")
print("x1 =", Solucao.x[0])
print("x2 =", Solucao.x[1])
print("x3 =", Solucao.x[2])
print("x4 =", Solucao.x[3])
