import pandas as pd
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time

# ==============================================================================
# 1. FUNÇÕES AUXILIARES E RESTRIÇÕES (Idênticas ao original)
# ==============================================================================

def get_sigma_values(z_values, col_names):
    factors = []
    for name in col_names:
        if name == 'TOTAL_PLANTA_GLOBAL':
            factors.append(0.01)
        elif str(name).endswith('|TOTAL'):
            factors.append(0.02)
        else:
            factors.append(0.05)
    factors = np.array(factors)
    return np.maximum(np.abs(z_values) * factors, 1e-4)

def constraint_balance(x, map_indices):
    balance_errors = []
    for key, indices in map_indices.items():
        idx_components = indices['components']
        idx_total = indices['total']
        if idx_components and idx_total is not None:
            sum_components = np.sum(x[idx_components])
            val_total = x[idx_total]
            balance_errors.append(sum_components - val_total)
    return np.array(balance_errors)

# ==============================================================================
# 2. FUNÇÕES OBJETIVO PURAS (Mono-Objetivo)
# ==============================================================================

def objective_wls_only(x, z_medido, sigmas):
    """Minimiza APENAS o erro quadrado ponderado (WLS)"""
    eps = 1e-15
    return np.sum(((x - z_medido) ** 2) / (sigmas ** 2 + eps))

def objective_l1_only(x, z_medido, sigmas): # Sigmas passados apenas para manter assinatura, não usados no calculo
    """Minimiza APENAS o erro absoluto (L1)"""
    return np.sum(np.abs(x - z_medido))

def calcular_metricas(x, z_medido, sigmas):
    """Função auxiliar para retornar ambos os valores (WLS e L1) para um dado x"""
    eps = 1e-15
    val_wls = np.sum(((x - z_medido) ** 2) / (sigmas ** 2 + eps))
    val_l1 = np.sum(np.abs(x - z_medido))
    return val_wls, val_l1

# ==============================================================================
# 3. WORKER: Resolve os extremos para UMA linha
# ==============================================================================
def resolver_extremos(z_medido, colunas_dados, map_indices_numeric):
    sigmas = get_sigma_values(z_medido, colunas_dados)
    
    # Configuração comum do solver
    cons = {'type': 'eq', 'fun': constraint_balance, 'args': (map_indices_numeric,)}
    bounds = [(0, None) for _ in range(len(z_medido))]
    options = {'ftol': 1e-4, 'disp': True, 'maxiter': 500}

    # --- PASSO 1: Encontrar Utopia WLS (Minimizar WLS) ---
    res_wls = minimize(
        objective_wls_only,
        z_medido,
        args=(z_medido, sigmas),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options=options
    )

    # --- PASSO 2: Encontrar Utopia L1 (Minimizar L1) ---
    res_l1 = minimize(
        objective_l1_only,
        z_medido,
        args=(z_medido, sigmas), # Passamos sigmas apenas para compatibilidade de args se necessário
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options=options
    )

    # --- Coleta de Dados para Tabela de Pagamentos (Payoff) ---
    if res_wls.success and res_l1.success:
        # Ponto 1: Otimizado para WLS
        # wls_utopia -> O melhor valor possível de WLS
        # l1_nadir_est -> O valor de L1 quando WLS é ótimo (geralmente ruim)
        wls_utopia, l1_nadir_est = calcular_metricas(res_wls.x, z_medido, sigmas)

        # Ponto 2: Otimizado para L1
        # wls_nadir_est -> O valor de WLS quando L1 é ótimo (geralmente ruim)
        # l1_utopia -> O melhor valor possível de L1
        wls_nadir_est, l1_utopia = calcular_metricas(res_l1.x, z_medido, sigmas)

        return {
            'success': True,
            'wls_utopia': wls_utopia,
            'wls_nadir': wls_nadir_est,
            'l1_utopia': l1_utopia,
            'l1_nadir': l1_nadir_est
        }
    else:
        return {'success': False}

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    print("--- Carregando dados ---")
    nome_arquivo = "data/resultado_complexo_siderurgico.csv"
    df = pd.read_csv(nome_arquivo)
    colunas_dados = [c for c in df.columns if c != 'Data_Hora']

    # --- Mapeamento de Topologia (Cópia do original) ---
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

    print("--- Iniciando cálculo de Escala (Utopia/Nadir) ---")
    print(f"Processando {len(df)} linhas em paralelo...")
    
    start_t = time.time()
    data_matrix = df[colunas_dados].values.astype(float)

    # Executa o worker para cada linha
    results = Parallel(n_jobs=-1)(
        delayed(resolver_extremos)(
            row, colunas_dados, map_indices_numeric
        ) for row in data_matrix
    )

    # --- Processamento dos Resultados ---
    df_payoff = pd.DataFrame([r for r in results if r['success']])
    
    falhas = len(df) - len(df_payoff)
    print(f"Concluído em {time.time()-start_t:.2f}s. (Falhas de convergência: {falhas})")

    if not df_payoff.empty:
        # Médias Globais (Essa é a "Escala" média do seu processo)
        print(f"wls:\n{df_payoff['wls_utopia']}\nl1:\n{df_payoff['l1_utopia']}")
        avg_wls_utopia = df_payoff['wls_utopia'].min()
        avg_wls_nadir = df_payoff['wls_nadir'].max()
        
        avg_l1_utopia = df_payoff['l1_utopia'].min()
        avg_l1_nadir = df_payoff['l1_nadir'].max()

        # Amplitude (Range)
        range_wls = avg_wls_nadir - avg_wls_utopia
        range_l1 = avg_l1_nadir - avg_l1_utopia

        print("\n" + "="*60)
        print(" MATRIZ DE PAGAMENTOS (MÉDIA GLOBAL) ")
        print("="*60)
        print(f"{'Função':<10} | {'Utopia (Melhor)':<15} | {'Nadir (Pior - Est.)':<15} | {'Range (Escala)':<15}")
        print("-" * 60)
        print(f"{'WLS':<10} | {avg_wls_utopia:<15.4f} | {avg_wls_nadir:<15.4f} | {range_wls:<15.4f}")
        print(f"{'L1':<10} | {avg_l1_utopia:<15.4f} | {avg_l1_nadir:<15.4f} | {range_l1:<15.4f}")
        print("-" * 60)

        # Sugestão de Normalização
        # Se Range WLS for 1000 e Range L1 for 10, WLS domina por fator de 100.
        # Fator de normalização = 1 / Range
        
        norm_factor_wls = 1.0 / range_wls if range_wls > 0 else 1.0
        norm_factor_l1 = 1.0 / range_l1 if range_l1 > 0 else 1.0

        print("\nSUGESTÃO PARA O SCRIPT DE OTIMIZAÇÃO:")
        print("Normalizar as funções objetivo dividindo-as por estes valores (ou multiplicando pelo inverso):")
        print(f"-> Escala WLS: {range_wls:.4f}")
        print(f"-> Escala L1 : {range_l1:.4f}")
        
        print("\nOu use estes pesos normalizadores (k) na função custo:")
        print("J_total = w1 * (J_wls * k_wls) + w2 * (J_l1 * k_l1)")
        print(f"-> k_wls = {norm_factor_wls:.6f}")
        print(f"-> k_l1  = {norm_factor_l1:.6f}")

    else:
        print("Erro crítico: Nenhuma linha convergiu nos extremos.")