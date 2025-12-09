import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Carregar Dados
nome_arquivo = "data/resultado_complexo_siderurgico.csv"
df = pd.read_csv(nome_arquivo)

# SELEÇÃO DO PONTO REPRESENTATIVO
colunas_dados = [c for c in df.columns if c != 'Data_Hora']
row_series = df.iloc[0][colunas_dados] 
print(row_series)
z_medido = row_series.values.astype(float)

# -------------------------
def get_sigma(z_series):
    factors = []
    for name in z_series.index:
        if name == 'TOTAL_PLANTA_GLOBAL': factors.append(0.01)
        elif str(name).endswith('|TOTAL'): factors.append(0.02)
        else: factors.append(0.05)
    return np.maximum(np.abs(z_series.values) * np.array(factors), 100)

# 2. Topologia
topology_map = {}
for col in colunas_dados:
    parts = col.split('|')
    if len(parts) == 3:
        planta, etapa, equipamento = parts
        key = (planta, etapa)
        if key not in topology_map: topology_map[key] = {'components': [], 'total': None}
        if equipamento == 'TOTAL': topology_map[key]['total'] = col
        else: topology_map[key]['components'].append(col)

col_to_index = {name: i for i, name in enumerate(colunas_dados)}
map_indices_numeric = {}
for key, val in topology_map.items():
    if val['total']:
        map_indices_numeric[key] = {
            'components': [col_to_index[c] for c in val['components']],
            'total': col_to_index[val['total']]
        }

# 3. Funções de Custo
def calc_raw_costs(x, z_medido, sigmas):
    eps = 1e-15
    j_wls = np.sum(((x - z_medido) ** 2) / (sigmas ** 2 + eps))
    j_l1 = np.sum(np.abs(x - z_medido))
    return j_wls, j_l1

def objective_function_normalized_pre(x, z_medido, sigmas, l1, l2, 
                                      wls_min, wls_range, l1_min, l1_range):
    """
    Função Objetivo já normalizada.
    Cada termo varia estritamente entre 0 e 1 (teoricamente).
    """
    j_wls_raw, j_l1_raw = calc_raw_costs(x, z_medido, sigmas)
    
    # Normalização Min-Max: (Valor - Min) / (Max - Min)
    norm_wls = (j_wls_raw - wls_min) / (wls_range + 1e-9)
    norm_l1 = (j_l1_raw - l1_min) / (l1_range + 1e-9)
    
    return (l1 * norm_wls) + (l2 * norm_l1)

def constraint_balance(x, map_indices):
    balance_errors = []
    for key, indices in map_indices.items():
        sum_components = np.sum(x[indices['components']])
        val_total = x[indices['total']]
        balance_errors.append(sum_components - val_total)
    return np.array(balance_errors)

# Preparação
sigmas = get_sigma(row_series)
cons = {'type': 'eq', 'fun': constraint_balance, 'args': (map_indices_numeric,)}
bounds = [(0, None) for _ in range(len(z_medido))]

# ==============================================================================
# FASE 1: CÁLCULO DA MATRIZ DE PAYOFF (NORMALIZAÇÃO PRÉVIA)
# ==============================================================================
print("--- Fase 1: Calculando Limites de Normalização (Pontos de Utopia) ---")

# A) Achar o Melhor WLS possível (Lambda=1.0 para WLS)
#    Usamos uma função dummy simples ou a raw com lambda extremo
print("Minimizando WLS...")
res_wls_only = minimize(
    lambda x, z, s: calc_raw_costs(x, z, s)[0], # Retorna só WLS
    z_medido, args=(z_medido, sigmas),
    method='SLSQP', bounds=bounds, constraints=cons
)
utopia_wls_val, nadir_l1_val = calc_raw_costs(res_wls_only.x, z_medido, sigmas)

# B) Achar o Melhor L1 possível (Lambda=1.0 para L1)
print("Minimizando L1...")
res_l1_only = minimize(
    lambda x, z, s: calc_raw_costs(x, z, s)[1], # Retorna só L1
    z_medido, args=(z_medido, sigmas),
    method='SLSQP', bounds=bounds, constraints=cons
)
nadir_wls_val, utopia_l1_val = calc_raw_costs(res_l1_only.x, z_medido, sigmas)

# C) Definir Ranges
range_wls = nadir_wls_val - utopia_wls_val
range_l1 = nadir_l1_val - utopia_l1_val

print(f"\nLimites Calculados:")
print(f"  WLS (Estatística): Min={utopia_wls_val:.4f} | Max={nadir_wls_val:.4f} | Range={range_wls:.4f}")
print(f"  L1  (Massa kg/h):  Min={utopia_l1_val:.4f} | Max={nadir_l1_val:.4f} | Range={range_l1:.4f}")

# ==============================================================================
# FASE 2: GERAÇÃO DA FRONTEIRA DE PARETO (COM OBJETIVO NORMALIZADO)
# ==============================================================================
lambda_values = np.linspace(0, 1, 21) 
pareto_results = []

print("\n--- Fase 2: Gerando Fronteira de Pareto ---")

for l1 in lambda_values:
    l2 = 1.0 - l1
    
    res = minimize(
        objective_function_normalized_pre, # <--- USANDO FUNÇÃO NORMALIZADA
        z_medido,
        args=(z_medido, sigmas, l1, l2, 
              utopia_wls_val, range_wls, utopia_l1_val, range_l1), # Passando limites
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-4, 'disp': False}
    )
    
    if res.success:
        final_wls, final_l1 = calc_raw_costs(res.x, z_medido, sigmas)
        
        # Calcular os valores normalizados correspondentes para plotagem
        final_wls_norm = (final_wls - utopia_wls_val) / (range_wls + 1e-9)
        final_l1_norm = (final_l1 - utopia_l1_val) / (range_l1 + 1e-9)
        
        pareto_results.append({
            'lambda_1': l1,
            'J_WLS_Raw': final_wls,
            'J_L1_Raw': final_l1,
            'J_WLS_Norm': final_wls_norm,
            'J_L1_Norm': final_l1_norm
        })

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================
df_pareto = pd.DataFrame(pareto_results)

if not df_pareto.empty:
    plt.figure(figsize=(10, 8))
    
    # Plotar pontos normalizados (que agora correspondem à 'força' real do otimizador)
    scatter = plt.scatter(df_pareto['J_WLS_Norm'], df_pareto['J_L1_Norm'], 
                          c=df_pareto['lambda_1'], cmap='viridis', s=120, edgecolors='k')
    
    plt.plot(df_pareto['J_WLS_Norm'], df_pareto['J_L1_Norm'], 'k--', alpha=0.3)

    # Identificar melhor ponto (distância à origem)
    df_pareto['Dist'] = np.sqrt(df_pareto['J_WLS_Norm']**2 + df_pareto['J_L1_Norm']**2)
    best_idx = df_pareto['Dist'].idxmin()
    best_row = df_pareto.loc[best_idx]
    
    plt.scatter(best_row['J_WLS_Norm'], best_row['J_L1_Norm'], 
                color='red', s=200, marker='*', label=f'Best (λ={best_row["lambda_1"]:.2f})')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Lambda 1 (Peso WLS)', rotation=270, labelpad=15)
    
    plt.title('Fronteira de Pareto\n(Otimização feita com gradientes normalizados)')
    plt.xlabel('Custo WLS Normalizado (0=Melhor, 1=Pior)')
    plt.ylabel('Custo L1 Normalizado (0=Melhor, 1=Pior)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pareto_pre_normalized.png')
    print("Gráfico salvo como pareto_pre_normalized.png")
    
    print("\n--- Resultados (Raw vs Norm) ---")
    print(df_pareto[['lambda_1', 'J_WLS_Raw', 'J_L1_Raw', 'J_WLS_Norm']].head(3))
    print("...")
    print(df_pareto[['lambda_1', 'J_WLS_Raw', 'J_L1_Raw', 'J_WLS_Norm']].tail(3))
    plt.show()
else:
    print("Falha na convergência.")