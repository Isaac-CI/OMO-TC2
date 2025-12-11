import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time

# ==============================================================================
# 1. DEFINIÇÃO DAS FUNÇÕES (Escopo Global para o Joblib)
# ==============================================================================

# Função para estimar Sigma (Mantida, caso queira usar depois)
def get_sigma(df):
    return df.std(numeric_only=True).values

# Função Objetivo
def objective_function_old(x, z_medido, sigmas, l1, l2):
    eps = 1e-15
    # Nota: Adicionada a divisão por (sigmas + eps) no L1 para normalização, 
    # conforme boas práticas discutidas, mas você pode remover se quiser manter o original.
    f_wls = np.sum(((x - z_medido) ** 2) / (sigmas ** 2 + eps))# + penalty(x)
    f_l1 = np.sum(np.abs(x - z_medido))# + penalty(x)
    return (l1 * f_wls) + (l2 * f_l1)
    #return f_wls


def objective_function(x_aug, z_medido, sigmas, l1, l2, n_vars):
    # Separar o vetor em duas partes
    x_fisico = x_aug[:n_vars] # Variáveis reais (índices 0 a 160)
    t_aux    = x_aug[n_vars:] # Variáveis auxiliares (índices 161 em diante)
    eps = 1e-15
    # WLS: Calculado sobre a parte física
    f_wls = np.sum(((x_fisico - z_medido) ** 2) / (sigmas ** 2 + eps))# + penalty(x_fisico)
    # L1: Agora é apenas a soma de t (Linearização!)
    # O módulo sumiu daqui. A "mágica" agora está nas restrições.
    f_l1 = np.sum(t_aux / (sigma + eps))# + penalty(x_fisico) 
    return (l1 * f_wls) + (l2 * f_l1)
    #return f_l1

def calculate_hypervolume(front_norm, ref_point=(1.1, 1.1)):
    """
    Calcula o Hipervolume para um problema de minimização bi-objetivo (2D).
    front_norm: Array N x 2 com dados normalizados e ORDENADOS pelo 1º objetivo.
    ref_point: Ponto de referência (deve ser maior que todos os pontos da fronteira).
    """
    # Garante que é array numpy
    front = np.array(front_norm)
    
    # O Hipervolume em 2D é a soma das áreas dos retângulos formados
    # entre cada ponto e o ponto de referência.
    # Algoritmo simples de varredura:
    volume = 0.0
    
    # Vamos iterar pelos pontos. Como estão ordenados por X (WLS),
    # calculamos a área "acima" da curva até o ref_point[1] (Y).
    
    # Altura inicial (do primeiro ponto até o teto)
    current_height = ref_point[1] - front[0, 1] 
    
    for i in range(len(front) - 1):
        # Largura entre o ponto atual e o próximo
        width = front[i+1, 0] - front[i, 0]
        
        # O retângulo formado usa a altura do ponto atual (que é "pior" em Y que o próximo)
        # Na verdade, em minimização pareto, se X aumenta, Y deve diminuir.
        # A área dominada é limitada pelo valor Y do ponto i.
        volume += width * (ref_point[1] - front[i, 1])
        
    # Último retângulo (do último ponto até a referência X)
    width_last = ref_point[0] - front[-1, 0]
    volume += width_last * (ref_point[1] - front[-1, 1])
    
    return volume

def calculate_delta_metric(front_norm):
    """
    Calcula a Métrica Delta (Uniformidade/Spread) de Deb.
    Delta baixo (próximo de 0) indica distribuição uniforme.
    """
    front = np.array(front_norm)
    
    # 1. Calcular distância Euclidiana entre pontos consecutivos
    # diffs[i] = ponto[i+1] - ponto[i]
    diffs = np.diff(front, axis=0)
    
    # dists[i] = norma(diffs[i])
    dists = np.linalg.norm(diffs, axis=1)
    
    if len(dists) == 0:
        return 0.0
    
    # 2. Média das distâncias
    d_mean = np.mean(dists)
    
    # 3. Extremos (distância do primeiro e último ponto aos "limites ideais")
    # Em problemas irrestritos/weighted sum, é comum assumir d_f e d_l como 0 
    # ou usar a distância até os eixos se houver bounds conhecidos.
    # Aqui usaremos a versão simplificada focada na uniformidade interna:
    d_f = 0.0 
    d_l = 0.0
    
    # 4. Cálculo do Delta
    delta = (d_f + d_l + np.sum(np.abs(dists - d_mean))) / (d_f + d_l + (len(front) - 1) * d_mean)
    
    return delta

# --- Definição das Restrições (Hardcoded conforme seu pedido) ---
THRESHOLD = 10
def cominuicao(x):
    return x[18] - np.sum(x[0:18])
def cominuicao_upper(x):
    return THRESHOLD - cominuicao(x)
def cominuicao_lower(x):
    return THRESHOLD + cominuicao(x)

def sinterizacao(x):
    return x[28] - np.sum(x[19:28])
def sinterizacao_upper(x):
    return THRESHOLD - sinterizacao(x)
def sinterizacao_lower(x):
    return THRESHOLD + sinterizacao(x)


def coqueificacao(x):
    return x[47] - np.sum(x[29:47])
def coqueificacao_upper(x):
    return THRESHOLD - coqueificacao(x)
def coqueificacao_lower(x):
    return THRESHOLD + coqueificacao(x)

def alto_forno(x):
    return x[78] - np.sum(x[48:78])
def alto_forno_upper(x):
    return THRESHOLD - alto_forno(x)
def alto_forno_lower(x):
    return THRESHOLD + alto_forno(x)


def aciaria(x):
    return x[91] - np.sum(x[79:91])
def aciaria_upper(x):
    return THRESHOLD - aciaria(x)
def aciaria_lower(x):
    return THRESHOLD + aciaria(x)

def lingotamento(x):
    return x[113] - np.sum(x[92:113])
def lingotamento_upper(x):
    return THRESHOLD - lingotamento(x)
def lingotamento_lower(x):
    return THRESHOLD + lingotamento(x)

def laminacao(x):
    return x[159] - np.sum(x[114:159])
def laminacao_upper(x):
    return THRESHOLD - laminacao(x)
def laminacao_lower(x):
    return THRESHOLD + laminacao(x)

def planta(x):
    return x[160] - (x[159] + x[113] + x[91] + x[78] + x[47] + x[28] + x[18])
def planta_upper(x):
    return THRESHOLD - planta(x)
def planta_lower(x):
    return THRESHOLD + planta(x)

def penalty(x):
    return np.abs(cominuicao(x) + sinterizacao(x) + coqueificacao(x) + alto_forno(x) + aciaria(x) + lingotamento(x) + laminacao(x) + planta(x))

# t >= x - z  -->  t - x + z >= 0
def constraint_linear_pos(x_aug, z_medido):
    n = len(z_medido)
    x_phys = x_aug[:n]
    t = x_aug[n:]
    return t - (x_phys - z_medido)

# t >= -(x - z) --> t + x - z >= 0
def constraint_linear_neg(x_aug, z_medido):
    n = len(z_medido)
    x_phys = x_aug[:n]
    t = x_aug[n:]
    return t + (x_phys - z_medido)

# Lista de restrições para passar ao otimizador
lista_restricoes_originais = []

# Lista de restrições para passar ao otimizador
lista_restricoes_originais = [
    {'type': 'eq', 'fun': cominuicao},
    {'type': 'eq', 'fun': sinterizacao},
    {'type': 'eq', 'fun': coqueificacao},
    {'type': 'eq', 'fun': alto_forno},
    {'type': 'eq', 'fun': aciaria},
    {'type': 'eq', 'fun': lingotamento},
    {'type': 'eq', 'fun': laminacao},
    {'type': 'eq', 'fun': planta},
]

# Lista de restrições para passar ao otimizador
# lista_restricoes_originais = [
#     {'type': 'ineq', 'fun': cominuicao_upper},
#     {'type': 'ineq', 'fun': sinterizacao_upper},
#     {'type': 'ineq', 'fun': coqueificacao_upper},
#     {'type': 'ineq', 'fun': alto_forno_upper},
#     {'type': 'ineq', 'fun': aciaria_upper},
#     {'type': 'ineq', 'fun': lingotamento_upper},
#     {'type': 'ineq', 'fun': laminacao_upper},
#     {'type': 'ineq', 'fun': planta_upper},
#     {'type': 'ineq', 'fun': cominuicao_lower},
#     {'type': 'ineq', 'fun': sinterizacao_lower},
#     {'type': 'ineq', 'fun': coqueificacao_lower},
#     {'type': 'ineq', 'fun': alto_forno_lower},
#     {'type': 'ineq', 'fun': aciaria_lower},
#     {'type': 'ineq', 'fun': lingotamento_lower},
#     {'type': 'ineq', 'fun': laminacao_lower},
#     {'type': 'ineq', 'fun': planta_lower},
# ]

# ==============================================================================
# 2. FUNÇÃO WORKER (Executa uma única linha)
# ==============================================================================
def resolver_linha(z_medido, l1, l2, sigma):
    # No seu código original, você usava sigma fixo de 0.05 dentro do loop
    # Se quiser usar o get_sigma, troque a linha abaixo por: sigmas = get_sigma(z_medido)
    n = len(z_medido)
    x0 = np.concatenate([z_medido, np.zeros(n) + 1e-6])
    bounds = [(0, None)] * (2 * n)

    minhas_restricoes = lista_restricoes_originais.copy()
    
    # Otimização do SLSQP aceita retorno vetorial (array) nas restrições
    minhas_restricoes.append({'type': 'ineq', 'fun': constraint_linear_pos, 'args': (z_medido,)})
    minhas_restricoes.append({'type': 'ineq', 'fun': constraint_linear_neg, 'args': (z_medido,)})

    res = minimize(
        objective_function,
        x0,
        args=(z_medido, sigma, l1, l2, n),
        method='SLSQP',
        bounds=bounds,
        constraints=minhas_restricoes,
        options={'ftol': 1e-9, 'disp': False, 'maxiter': 10000}
    )
    
    if res.success:
        return res.x[:n], True
    else:
        return z_medido, False # Fallback em caso de falha

def calcular_custos_individuais(x_reconciliado, z_medido, sigma):
    # Recalcula WLS e L1 separadamente para plotar o gráfico de Pareto
    # Usa a mesma definição de sigma do resolver_linha para consistência
    eps = 1e-15
    
    cost_wls = np.sum(((x_reconciliado - z_medido) ** 2) / (sigma ** 2 + eps))
    cost_l1 = np.sum(np.abs(x_reconciliado - z_medido)  / (sigma + eps))
    
    return cost_wls, cost_l1

# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # Carregar dados
    nome_arquivo = "data/resultado_complexo_siderurgico.csv"
    try:
        df = pd.read_csv(nome_arquivo)
    except FileNotFoundError:
        print("Arquivo não encontrado. Verifique o caminho.")
        exit()

    colunas_dados = [c for c in df.columns if c != 'Data_Hora']
    
    # Converter para matriz NumPy (Muito mais rápido para iterar)
    data_matrix = df[colunas_dados].values.astype(float)

    #Temp para arquivos pequenos
    sigma = get_sigma(df)
    #sigma = np.maximum(data_matrix * 0.02, 1e-4)
    
    print("Iniciando reconciliação paralela com rastreamento de pareto...")

    # Lista de dicionários: {'Hora': idx, 'Lambda': val, 'WLS': val, 'L1': val}
    pareto_records = []
    reconciled_rows = []
    
    # Configuração dos Lambdas
    # Logaritimo
    # ratios = np.logspace(-5, 5, 19)
    # lambda_values = ratios / (1 + ratios)
    # lambda_values = np.sort(np.unique(np.concatenate(([0.0, 1.0], lambda_values))))
    # Linear
    lambda_values = np.linspace(0,1,11)

    # Loop Externo (Lambdas)
    for l2 in lambda_values:
        l1 = 1 - l2
        print(f"Processando lambda_1: {l1:.6f} - lambda_2: {l2:.6f} ...", end=" ")
        start_t = time.time()
        
        # --- PARALELISMO AQUI ---
        # n_jobs=-1 usa todos os núcleos do processador
        resultados_completos = Parallel(n_jobs=-1)(
            # delayed(resolver_linha)(row, l1, l2, sigma) for row in data_matrix
            delayed(resolver_linha)(row, l1, l2, sigma) for row in data_matrix[0:1]
        )

        dados_dessa_rodada = [item[0] for item in resultados_completos]
        status_dessa_rodada = [item[1] for item in resultados_completos]
        
        # Pós-processamento imediato para extrair os custos WLS e L1
        for i, (x_res, sucesso) in enumerate(resultados_completos):
            # Se quiser salvar apenas sucessos no Pareto, descomente o if:
            if sucesso:
                # Recalcula os custos para este cenário específico
                c_wls, c_l1 = calcular_custos_individuais(x_res, data_matrix[i], sigma)
                
                pareto_records.append({
                    'Hora_Index': i,
                    'Lambda': l1,
                    'WLS': c_wls,
                    'L1': c_l1,
                    'Sucesso': sucesso,
                    'Solucao': x_res  # Guardamos a solução se quiser exportar depois
                })
        reconciled_rows.extend(dados_dessa_rodada)
        total_falhas = status_dessa_rodada.count(False)
        #registros_hora_0 = [rec for rec in pareto_records if rec['Hora_Index'] == 0]
        #print(f"registros_hora_0:\n{registros_hora_0}")
        
        print(f"Concluído em {time.time()-start_t:.2f}s. (Falhas: {total_falhas})")

    # Resultado Final
    df_reconciled = pd.DataFrame(reconciled_rows, columns=colunas_dados)
    
    print("\nExemplo de Dados Reconciliados (Planta 5 - Sinterização):")
    cols_demo = [c for c in colunas_dados if 'Planta 5' in c and 'Sinterizacao' in c]
    if cols_demo:
        print(df_reconciled[cols_demo].head())

    # Salvar
    output_path = "data/resultado_complexo_siderurgico_reconciliado.csv"
    df_reconciled.to_csv(output_path, index=False)
    print(f"Arquivo salvo em: {output_path}")

    # Criar DataFrame Mestre com todos os resultados
    df_pareto_full = pd.DataFrame(pareto_records)
    print(f"\nTotal de cenários calculados: {len(df_pareto_full)}")

  # ==========================================================================
    # 4. SELEÇÃO AUTOMÁTICA PELO ÓTIMO DE PARETO (LOCAL)
    # ==========================================================================
    
    # --- ESCOLHA AQUI A HORA ---
    HORA_ESCOLHIDA = 0  # Exemplo: Linha 100 do seu CSV
    # ---------------------------

    print(f"\nAnalisando Fronteira de Pareto para a Hora {HORA_ESCOLHIDA}...")
    
    # Filtrar dados para a hora escolhida
    df_hora = df_pareto_full[
        (df_pareto_full['Hora_Index'] == 0) & 
        (df_pareto_full['Sucesso'] == True)
    ].copy()
    print(df_hora)
    
    if df_hora.empty:
        print("Erro: Nenhuma solução convergiu para esta hora.")
    else:
        # --- LÓGICA DE SELEÇÃO AUTOMÁTICA ---
        # 1. Normalização Min-Max para esta hora específica
        min_wls, max_wls = df_hora['WLS'].min(), df_hora['WLS'].max()
        min_l1, max_l1 = df_hora['L1'].min(), df_hora['L1'].max()
        
        # Evitar divisão por zero
        denom_wls = (max_wls - min_wls) if (max_wls - min_wls) > 1e-9 else 1.0
        denom_l1 = (max_l1 - min_l1) if (max_l1 - min_l1) > 1e-9 else 1.0
        
        df_hora['WLS_Norm'] = (df_hora['WLS'] - min_wls) / denom_wls
        df_hora['L1_Norm'] = (df_hora['L1'] - min_l1) / denom_l1
        
        # 2. Calcular Distância Euclidiana até a origem (0,0) ideal
        df_hora['Distancia'] = np.sqrt(df_hora['WLS_Norm']**2 + df_hora['L1_Norm']**2)
        
        # 3. Encontrar o ponto com menor distância
        idx_best = df_hora['Distancia'].idxmin()
        ponto_destaque = df_hora.loc[idx_best]
        
        melhor_lambda = ponto_destaque['Lambda']
        
        print(f"\n>>> Melhor Lambda Local Identificado: {melhor_lambda:.5f}")
        print(f"    (Minimiza o compromisso entre erro estatístico e balanço de massa para esta hora)")

        # --- PLOTAGEM ---
        plt.figure(figsize=(10, 6))
        
        # Ordenar para plotar a linha corretamente
        df_hora_sorted = df_hora.sort_values('WLS')
        plt.plot(df_hora_sorted['WLS'], df_hora_sorted['L1'], 'b-', alpha=0.3, label='Fronteira de Pareto')
        
        # Pontos de todos os lambdas
        sc = plt.scatter(df_hora['WLS'], df_hora['L1'], c=df_hora['Lambda'], 
                         cmap='viridis', s=80, edgecolors='k', zorder=2)
        
        # Ponto Destaque (O Melhor Encontrado Automaticamente)
        plt.scatter(ponto_destaque['WLS'], ponto_destaque['L1'], 
                    color='red', s=300, marker='*', zorder=3,
                    label=f'Melhor Lambda: {melhor_lambda:.4f}')

        plt.title(f'Fronteira de Pareto Otimizada - Hora {HORA_ESCOLHIDA}')
        plt.xlabel('Custo WLS (Aderência à Incerteza)')
        plt.ylabel('Custo L1 (Massa Movimentada)')
        plt.colorbar(sc, label='Valor de Lambda 1')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        nome_img = f"pareto_auto_hora_{HORA_ESCOLHIDA}.png"
        plt.savefig(nome_img)
        plt.show()
        print(f"Gráfico salvo como: {nome_img}")
        
        print("\n--- Detalhes do Ponto Ótimo ---")
        print(f"Lambda:   {ponto_destaque['Lambda']:.6f}")
        print(f"Custo WLS: {ponto_destaque['WLS']:.4f}")
        print(f"Custo L1:  {ponto_destaque['L1']:.4f}")
        
        # Exportar a solução
        solucao_final = ponto_destaque['Solucao']
        df_solucao = pd.DataFrame([solucao_final], columns=colunas_dados)
        df_solucao.to_csv(f"solucao_otima_hora_{HORA_ESCOLHIDA}.csv", index=False)
        print(f"Solução ótima salva em CSV.")
        
        # --- CÁLCULO DAS MÉTRICAS ---
        pareto_front_norm = df_hora_sorted[['WLS_Norm', 'L1_Norm']].values
        # Hipervolume (Maior é melhor): Área coberta pela fronteira
        hv_value = calculate_hypervolume(pareto_front_norm, ref_point=(1.1, 1.1))
        
        # Delta (Menor é melhor): Uniformidade da distribuição dos pontos
        delta_value = calculate_delta_metric(pareto_front_norm)
        
        print(f"\n>>> Métricas da Fronteira de Pareto:")
        print(f"    Hipervolume (Ref: 1.1, 1.1): {hv_value:.5f} (Maior = Melhor convergência/cobertura)")
        print(f"    Delta Metric (Spread):       {delta_value:.5f} (Menor = Distribuição mais uniforme)")