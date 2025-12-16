# BIO_A – Spojitý LTI systém chemických reakčných sietí
# Varianta (a)
# Reakcie:
# A -> B, B -> C, C -> A
# _ -> A (prítok A)
# A -> _ , B -> _ , C -> _ (odtoky)
# Všetky reakčné koeficienty = 1

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------------------------------
# 1. DEFINÍCIA MODELU (LTI ODE)
# -------------------------------------------------


def model(t,x):
    A, B, C = x
    dA = -A + C + 1 - A   # A->B, C->A, prítok, odtok
    dB = A - B - B       # A->B, B->C, odtok
    dC = B - C - C       # B->C, C->A, odtok
    return [dA, dB, dC]

# -------------------------------------------------
# 2. SIMULÁCIA V ČASE
# -------------------------------------------------

t_span = (0, 20)
t_eval = np.linspace(0, 20, 400)

# významné iniciálne podmienky
initial_conditions = {
    "nízke koncentrácie": [0.1, 0.1, 0.1],
    "stredné koncentrácie": [1, 1, 1],
    "vysoké A": [5, 0, 0]
}

plt.figure(figsize=(10,6))

for label, x0 in initial_conditions.items():
    sol = solve_ivp(model, t_span, x0, t_eval=t_eval)
    plt.plot(sol.t, sol.y[0], label=f"A ({label})")
    plt.plot(sol.t, sol.y[1], '--', label=f"B ({label})")
    plt.plot(sol.t, sol.y[2], ':', label=f"C ({label})")

plt.xlabel("čas")
plt.ylabel("koncentrácia")
plt.title("Časová simulácia reakčnej siete (varianta a)")
plt.legend(fontsize=8)
plt.grid()
plt.show()

# -------------------------------------------------
# 3. ANALYTICKÉ URČENIE EKVILIBRIA
# -------------------------------------------------

A, B, C = sp.symbols('A B C', real=True)


f1 = -A + C + 1 - A
f2 = A - B - B
f3 = B - C - C

solutions = sp.solve([f1, f2, f3], (A, B, C))
print("Ekvilibrium:", solutions)

# -------------------------------------------------
# 4. STABILITA – JACOBIHO MATICA
# -------------------------------------------------

J = sp.Matrix([f1, f2, f3]).jacobian([A, B, C])
J_eq = J.subs(solutions)

print("Jacobiho matica v ekvilibriu:")
sp.pprint(J_eq)

# vlastné čísla
print("Vlastné čísla:")
print(J_eq.eigenvals())

# -------------------------------------------------
# 5. VEKTOROVÉ POLE – 2D PROJEKCIE
# -------------------------------------------------

# def vector_field(x, y, fixed_value, proj="AB"):
#     if proj == "AB":
#         A, B, C = x, y, fixed_value
#     elif proj == "AC":
#         A, B, C = x, fixed_value, y
#     else:  # BC
#         A, B, C = fixed_value, x, y
    

#     dA = -A + C + 1 - A
#     dB = A - B - B
#     dC = B - C - C

#     if proj == "AB":
#         return dA, dB
#     elif proj == "AC":
#         return dA, dC
#     else:
#         return dB, dC

def vector_field(x, y, fixed_value, proj="AB"):
    if proj == "AB":
        A, B, C = x, y, fixed_value
        dA, dB, _ = model(0, [A, B, C])
        return dA, dB

    elif proj == "AC":
        A, B, C = x, fixed_value, y
        dA, _, dC = model(0, [A, B, C])
        return dA, dC

    else:  # BC
        A, B, C = fixed_value, x, y
        _, dB, dC = model(0, [A, B, C])
        return dB, dC


# def plot_vector_field(proj, fixed_value):
#     x = np.linspace(0, 3, 20)
#     y = np.linspace(0, 3, 20)
#     X, Y = np.meshgrid(x, y)
#     U = np.zeros_like(X)
#     V = np.zeros_like(Y)

#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             U[i,j], V[i,j] = vector_field(X[i,j], Y[i,j], None, fixed_value, proj)

#     plt.figure(figsize=(5,5))
#     plt.quiver(X, Y, U, V)
#     plt.xlabel(proj[0])
#     plt.ylabel(proj[1])
#     plt.title(f"Vektorové pole – projekcia {proj}, fix={fixed_value}")
#     plt.grid()
#     plt.show()

# plot_vector_field("AB", fixed_value=solutions[C])
# plot_vector_field("AC", fixed_value=solutions[B])
# plot_vector_field("BC", fixed_value=solutions[A])

def plot_vector_field(proj, fixed_value, equilibrium):
    x = np.linspace(0, 3, 20)
    y = np.linspace(0, 3, 20)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i, j], V[i, j] = vector_field(
                X[i, j], Y[i, j], fixed_value, proj
            )

    plt.figure(figsize=(5, 5))
    plt.quiver(X, Y, U, V)

    # --- ekvilibrium ---
    if proj == "AB":
        eq_x, eq_y = equilibrium[A], equilibrium[B]
    elif proj == "AC":
        eq_x, eq_y = equilibrium[A], equilibrium[C]
    else:  # BC
        eq_x, eq_y = equilibrium[B], equilibrium[C]

    plt.plot(eq_x, eq_y, 'ro', markersize=8, label="Ekvilibrium")

    plt.xlabel(proj[0])
    plt.ylabel(proj[1])
    plt.title(f"Vektorové pole – projekcia {proj}")
    plt.legend()
    plt.grid()
    plt.show()

plot_vector_field("AB", fixed_value=solutions[C], equilibrium=solutions)
plot_vector_field("AC", fixed_value=solutions[B], equilibrium=solutions)
plot_vector_field("BC", fixed_value=solutions[A], equilibrium=solutions)