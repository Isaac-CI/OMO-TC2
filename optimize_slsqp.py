import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. Carregar os dados
# Usamos o novo arquivo que você enviou
nome_arquivo = "data/resultado_complexo_siderurgico.csv"
df = pd.read_csv(nome_arquivo)

# Parâmetros de Otimização
LAMBDA_1 = 0.5 #Peso para WLS (Quadrático)
LAMBDA_2 = 0.5 #Peso para L1 (Absoluto)

#Dúvida, como obter o Sigma do WLS por meio das medições ?
def get_sigma(z_values):
  #Definindo a priori um valor fixo de 2% para cada medidor
  return np.maximum(z_values * 0.02, 1e-4) #O maximum e o valor 1e-4 é para evitar uma divisão por zero, no pior caso o valor da incerteza será minimo (0,0001)

# Função Objetivo (Minimizar o erro de medição)
def objective_function(x, z_medido, sigmas, l1, l2):
    f_wls = np.sum(((x - z_medido) ** 2) / (sigmas ** 2))
    f_l1 = np.sum(np.abs(x - z_medido))
    return (l1 * f_wls) + (l2 * f_l1)

# 2. Identificação Automática da Topologia
# O script agrupa as colunas por Planta e Etapa para criar as equações de balanço
colunas_dados = [c for c in df.columns if c != 'Data_Hora']
topology_map = {}

for col in colunas_dados:
    parts = col.split('|')
    if len(parts) == 3: # Formato: Planta | Etapa | Equipamento
        planta, etapa, equipamento = parts
        key = (planta, etapa)
        
        if key not in topology_map:
            topology_map[key] = {'components': [], 'total': None}
        
        if equipamento == 'TOTAL':
            topology_map[key]['total'] = col
        else:
            topology_map[key]['components'].append(col)

# 3. Função de Restrição Dinâmica
def constraint_balance(x, map_indices):
    # Esta função retorna uma lista de "resíduos" de balanço.
    # O otimizador tentará fazer com que todos esses valores sejam ZERO.
    balance_errors = []
    
    for key, indices in map_indices.items():
        idx_components = indices['components']
        idx_total = indices['total']
        
        # Se temos componentes e um totalizador, a conta deve fechar
        if idx_components and idx_total is not None:
            # Soma dos equipamentos
            sum_components = np.sum(x[idx_components])
            # Valor do medidor total
            val_total = x[idx_total]
            
            # Restrição: Soma - Total = 0
            balance_errors.append(sum_components - val_total)
            
    return np.array(balance_errors)

# 4. Executar Reconciliação (Exemplo para a primeira linha de dados)
print("Iniciando reconciliação baseada na topologia dos arquivos .mmd...")

# Mapear nomes de colunas para índices numéricos (0, 1, 2...) para o otimizador
col_to_index = {name: i for i, name in enumerate(colunas_dados)}
map_indices_numeric = {}

for key, val in topology_map.items():
    if val['total']: # Só processa se houver coluna TOTAL
        map_indices_numeric[key] = {
            'components': [col_to_index[c] for c in val['components']],
            'total': col_to_index[val['total']]
        }

reconciled_rows = []

# Processando linha a linha (Exemplo com as 5 primeiras para rapidez)
for i in range(min(5, len(df))):
    z_medido = df.iloc[i][colunas_dados].values.astype(float)
    sigmas = get_sigma(z_medido)
    
    # Restrições para o SciPy
    cons = {'type': 'eq', 'fun': constraint_balance, 'args': (map_indices_numeric,)}
    bounds = [(0, None) for _ in range(len(z_medido))] # Energia não pode ser negativa
    
    res = minimize(
        objective_function,
        z_medido, # Chute inicial = valor medido
        args=(z_medido, sigmas, LAMBDA_1, LAMBDA_2),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-4, 'disp': False}
    )
    
    if res.success:
        reconciled_rows.append(res.x)
    else:
        reconciled_rows.append(z_medido) # Fallback
        print(f"Falha na linha {i}")

# Resultado
df_reconciled = pd.DataFrame(reconciled_rows, columns=colunas_dados)
print("\nExemplo de Dados Reconciliados (Planta 1 - Sinterização):")
cols_demo = [c for c in colunas_dados if 'Planta 1' in c and 'Sinterizacao' in c]
print(df_reconciled[cols_demo].head())