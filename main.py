import numpy as np
import matplotlib.pyplot as plt

# --- 1. Defini��o da Fun��o e do Gradiente ---

# Constante 'c' (que combina infecciosidade do pat�geno, etc.)
# Voc� pode mudar esse valor para ver como afeta o gr�fico.
c = 1.0

# Define a fun��o C(I, E)
def C(I, E):
    return (c * E) / I

# Define os componentes do gradiente
# Componente dC/dI (dire��o I)
def dC_dI(I, E):
    return (-c * E) / (I**2)

# Componente dC/dE (dire��o E)
def dC_dE(I, E):
    return c / I

# --- 2. Prepara��o da Grade (Grid) ---

# Define os intervalos para Imunidade (I) e Exposi��o (E)
# NOTA: Come�amos I de um valor pequeno > 0 para evitar divis�o por zero.
i_range = np.linspace(0.5, 5.0, 20)  # Imunidade de 0.5 a 5.0
e_range = np.linspace(0.5, 5.0, 20)  # Exposi��o de 0.5 a 5.0

# Cria a grade de pontos 2D
I, E = np.meshgrid(i_range, e_range)

# --- 3. C�lculo da Fun��o e do Gradiente na Grade ---

# Calcula a chance de infec��o Z (eixo z) em cada ponto da grade
Z_chance = C(I, E)

# Calcula os componentes U (eixo x) e V (eixo y) do gradiente em cada ponto
U_grad = dC_dI(I, E)
V_grad = dC_dE(I, E)

# --- 4. Normaliza��o dos Vetores (Para melhor visualiza��o) ---
# O gradiente explode perto de I=0. Normalizamos os vetores (mudamos
# seu comprimento para 1) para focar apenas na DIRE��O.
magnitude = np.sqrt(U_grad**2 + V_grad**2)
U_norm = U_grad / magnitude
V_norm = V_grad / magnitude

# --- 5. Gera��o do Gr�fico ---

print("Gerando o gr�fico...")

plt.figure(figsize=(12, 9))

# 5a. Plot das Curvas de N�vel (Mapa de Calor)
# Isso mostra o valor da fun��o C(I, E)
# 'inferno' � um bom mapa de cores: amarelo = ALTO risco, preto = BAIXO risco
contour = plt.contourf(I, E, Z_chance, levels=25, cmap='inferno', alpha=0.8)
plt.colorbar(contour, label='C (Chance de Infec��o)')
plt.savefig('veremos.png')

# 5b. Plot do Campo Vetorial (Setas do Gradiente)
# Sobrep�e as