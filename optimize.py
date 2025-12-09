import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time

# ==============================================================================
# FUNÇÕES GLOBAIS (Necessárias para o Worker do Joblib)
# ==============================================================================

def get_sigma_values(z_values, col_names):
    """
    Versão adaptada de get_sigma para receber arrays diretos (mais rápido no paralelo)
    Mantém a mesma lógica de hierarquia.
    """
    factors = []
    for name in col_names:
        if name == 'TOTAL_PLANTA_GLOBAL':
            factors.append(0.01) # 1%
        elif str(name).endswith('|TOTAL'):
            factors.append(0.02) # 2%
        else:
            factors.append(0.05) # 5%
    factors = np.array(factors)
    return np.maximum(np.abs(z_values) * factors, 1e-4)

def objective_function(x, z_medido, sigmas, l1, l2):
    eps = 1e-15
    f_wls = np.sum(((x - z_medido) ** 2) / (sigmas ** 2 + eps))
    f_l1 = np.sum(np.abs(x - z_medido))
    return (l1 * f_wls) + (l2 * f_l1)

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

# --- WORKER: Função que será executada em paralelo para cada linha ---
def resolver_linha(z_medido, colunas_dados, l1, l2, map_indices_numeric):
    # Calcular sigmas
    sigmas = get_sigma_values(z_medido, colunas_dados)
    eps = 1e-15
    
    # Configurar Otimização
    cons = {'type': 'eq', 'fun': constraint_balance, 'args': (map_indices_numeric,)}
    bounds = [(0, None) for _ in range(len(z_medido))]
    
    res = minimize(
        objective_function,
        z_medido,
        args=(z_medido, sigmas, l1, l2),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-4, 'disp': False, 'maxiter': 500}
    )
    
    # Calcular resultados para retorno
    if res.success:
        wls_val = np.sum(((res.x - z_medido) ** 2) / ((sigmas ** 2) + eps))
        l1_val = np.sum(np.abs(res.x - z_medido))
        return res.x, wls_val, l1_val, True
    else:
        return z_medido, 0.0, 0.0, False

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Carregar os dados
    nome_arquivo = "data/resultado_complexo_siderurgico.csv"
    df = pd.read_csv(nome_arquivo)
    
    colunas_dados = [c for c in df.columns if c != 'Data_Hora']
    
    # 2. Identificação Automática da Topologia (Pré-processamento)
    topology_map = {}
    for col in colunas_dados:
        parts = col.split('|')
        if len(parts) == 3:
            planta, etapa, equipamento = parts
            key = (planta, etapa)
            if key not in topology_map:
                topology_map[key] = {'components': [], 'total': None}
            if equipamento == 'TOTAL':
                topology_map[key]['total'] = col
            else:
                topology_map[key]['components'].append(col)

    # Mapeamento Numérico
    col_to_index = {name: i for i, name in enumerate(colunas_dados)}
    map_indices_numeric = {}
    for key, val in topology_map.items():
        if val['total']:
            map_indices_numeric[key] = {
                'components': [col_to_index[c] for c in val['components']],
                'total': col_to_index[val['total']]
            }

    print("Iniciando reconciliação paralela...")
    
    # Preparar dados para o Joblib (Matriz numpy é mais leve que passar DataFrame)
    data_matrix = df[colunas_dados].values.astype(float)
    
    reconciled_rows = []
    # Configurar Lambdas (Log scale)
    ratios = np.logspace(-5, 5, 19)
    lambda_values = ratios / (1 + ratios)
    lambda_values = np.sort(np.unique(np.concatenate(([0.0, 1.0], lambda_values))))
    pareto_results = []
    
    # Loop Externo (Lambdas) - Sequencial
    # Loop Interno (Linhas) - Paralelo
    
    for l1 in lambda_values:
        l2 = 1 - l1
        print(f"Processando lambda_1: {l1:.6f} - lambda_2: {l2:.6f} ...", end=" ")
        start_t = time.time()
        
        # --- PARALELISMO AQUI ---
        # n_jobs=-1 usa todos os núcleos disponíveis
        results = Parallel(n_jobs=-1)(
            delayed(resolver_linha)(
                row, colunas_dados, l1, l2, map_indices_numeric
            ) for row in data_matrix
        )
        # ------------------------
        
        # Agregação dos resultados
        total_wls = 0
        total_l1 = 0
        failures = 0
        
        for x_res, wls_part, l1_part, success in results:
            reconciled_rows.append(x_res)
            
            if success:
                total_wls += wls_part
                total_l1 += l1_part
            else:
                failures += 1

        pareto_results.append({
            'lambda_1': l1,
            'J_WLS_Avg': total_wls / (len(df)-failures),
            'J_L1_Avg': total_l1 / (len(df)-failures)
        })
        
        print(f"Concluído em {time.time()-start_t:.2f}s. (Falhas: {failures})")

    # ==============================================================================
    # VISUALIZAÇÃO E NORMALIZAÇÃO (Código Original mantido)
    # ==============================================================================
    df_pareto = pd.DataFrame(pareto_results)

    if not df_pareto.empty:
        # 1. Normalizar os Custos MÉDIOS (0 a 1)
        min_wls = df_pareto['J_WLS_Avg'].min()
        max_wls = df_pareto['J_WLS_Avg'].max()
        min_l1 = df_pareto['J_L1_Avg'].min()
        max_l1 = df_pareto['J_L1_Avg'].max()
        
        print(f"df_pareto['J_WLS_Avg']:\n{df_pareto['J_WLS_Avg']}")
        print(f"df_pareto['J_L1_Avg']:\n{df_pareto['J_L1_Avg']}")

        denom_wls = max_wls if max_wls > 1e-9 else 1.0
        denom_l1 = max_l1 if max_l1 > 1e-9 else 1.0

        df_pareto['WLS_Norm'] = df_pareto['J_WLS_Avg']
        df_pareto['L1_Norm'] = df_pareto['J_L1_Avg']
        
        # 2. Encontrar o Melhor Ponto Médio (Distância Euclidiana)
        df_pareto['Dist'] = np.sqrt(df_pareto['WLS_Norm']**2 + df_pareto['L1_Norm']**2)
        print(f"df_pareto['Dist']:\n{df_pareto['Dist']}")
        best_point = df_pareto.loc[df_pareto['Dist'].idxmin()]

        # 3. Plotar
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(df_pareto['WLS_Norm'], df_pareto['L1_Norm'], 
                              c=df_pareto['lambda_1'], cmap='viridis', s=100, edgecolors='k', zorder=2)
        
        plt.plot(df_pareto['WLS_Norm'], df_pareto['L1_Norm'], 'k--', alpha=0.3, zorder=1)
        
        # Destaque
        plt.scatter(best_point['WLS_Norm'], best_point['L1_Norm'], 
                    color='red', s=250, marker='*', zorder=3,
                    label=f'Melhor Lambda Global: {best_point["lambda_1"]:.5f}')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Lambda 1 (Escala Linear)', rotation=270, labelpad=15)
        
        plt.title(f'Fronteira de Pareto Global (Média de {len(df)} amostras)\nRobustez Operacional do Algoritmo')
        plt.xlabel('Custo Estatístico Médio (WLS)')
        plt.ylabel('Consumo de Energia Médio (L1)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Margem
        plt.xlim(0.9*min_wls, 1.1*max_wls)
        plt.ylim(0.9*min_l1, 1.1*max_l1)
        
        plt.tight_layout()
        plt.savefig('pareto_global_average.png')
        plt.show()
        print("\nGráfico salvo como pareto_global_average.png")
        
        print("\n--- Resultados do Melhor Ponto Global ---")
        print(f"Lambda Ideal: {best_point['lambda_1']:.5f}")
        print(f"WLS Médio Esperado: {best_point['J_WLS_Avg']:.4f} (Chi-Quadrado)")
        print(f"Erro Energia Médio:   {best_point['J_L1_Avg']:.4f}")

    else:
        print("Falha: Nenhuma convergência obtida.")

    # Resultado
    # Nota: reconciled_rows conterá (21 * N) linhas, conforme lógica original
    df_reconciled = pd.DataFrame(reconciled_rows, columns=colunas_dados)
    print("\nExemplo de Dados Reconciliados (Planta 5 - Sinterização):")
    cols_demo = [c for c in colunas_dados if 'Planta 5' in c and 'Sinterizacao' in c]
    if cols_demo:
        print(df_reconciled[cols_demo].head())

    #Salvar
    df_reconciled.to_csv("data/resultado_complexo_siderurgico_reconciliado.csv")
    print("Arquivo salvo (data/resultado_complexo_siderurgico_reconciliado.csv)")