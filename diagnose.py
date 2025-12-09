import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO
# ==============================================================================
# Caminhos dos arquivos (Ajuste se necessário)
file_measured = "data/resultado_complexo_siderurgico.csv"
file_reconciled = "data/resultado_complexo_siderurgico_reconciliado.csv"

# Carregar dados
try:
    df_meas = pd.read_csv(file_measured)
    df_rec_full = pd.read_csv(file_reconciled)
except FileNotFoundError:
    print("ERRO: Arquivos não encontrados. Verifique o caminho em 'file_measured' e 'file_reconciled'.")
    exit()

# Remover colunas não numéricas (Data_Hora, indices antigos) para cálculo
cols_data = [c for c in df_meas.columns if c not in ['Data_Hora', 'Unnamed: 0']]
df_meas_numeric = df_meas[cols_data]
df_rec_numeric = df_rec_full[cols_data]

# --- RECONSTRUÇÃO DOS INDICES DE LAMBDA ---
n_rows = len(df_meas)
total_rows_rec = len(df_rec_full)

# Proteção contra divisão por zero se arquivos estiverem vazios
if n_rows == 0:
    print("Erro: Arquivo de medição vazio.")
    exit()

n_lambdas_detected = total_rows_rec // n_rows

print(f"Dataset Original: {n_rows} linhas")
print(f"Dataset Reconciliado: {total_rows_rec} linhas")
print(f"Detectado {n_lambdas_detected} cenários de Lambda no arquivo.")

# ===> AQUI ESTÁ A ATUALIZAÇÃO SOLICITADA <===
# Recriar o vetor de Lambdas conforme sua nova definição
ratios = np.logspace(-5, 5, 19) # Agora usa 19 pontos internos
lambda_values = ratios / (1 + ratios)
lambda_values = np.sort(np.unique(np.concatenate(([0.0, 1.0], lambda_values))))

print(f"Lambdas definidos no script: {len(lambda_values)}")

# Validação de consistência
if len(lambda_values) != n_lambdas_detected:
    print(f"\n[ALERTA] Inconsistência detectada!")
    print(f"O CSV tem {n_lambdas_detected} iterações, mas a definição de lambda gerou {len(lambda_values)} valores.")
    print("Isso geralmente acontece se você mudou o código mas não rodou a otimização de novo.")
    print("Usando índices lineares genéricos para não quebrar o gráfico.")
    lambda_values = np.linspace(0, 1, n_lambdas_detected)

# ==============================================================================
# 2. CÁLCULO DE MÉTRICAS DE FALHA E AJUSTE
# ==============================================================================
metrics = []

print("Calculando métricas de diagnóstico...")

for i, l1 in enumerate(lambda_values):
    # Extrair o bloco correspondente a este Lambda
    start_idx = i * n_rows
    end_idx = (i + 1) * n_rows
    
    # Pegar o pedaço reconciliado e resetar index para comparar com o medido
    df_rec_chunk = df_rec_numeric.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Diferença absoluta
    diff = np.abs(df_rec_chunk - df_meas_numeric)
    
    # Soma dos ajustes por linha
    row_diffs = diff.sum(axis=1)
    
    # 1. DETECÇÃO DE FALHA (Heurística: Ajuste == 0 exato significa fallback)
    failures = (row_diffs < 1e-9).sum()
    
    # 2. MAGNITUDE DOS AJUSTES (Apenas para quem convergiu)
    valid_adjustments = row_diffs[row_diffs > 1e-9]
    avg_adjustment = valid_adjustments.mean() if not valid_adjustments.empty else 0
    
    metrics.append({
        'lambda_1': l1,
        'lambda_2': 1.0 - l1,
        'failures': failures,
        'failure_rate': (failures / n_rows) * 100,
        'avg_adjustment_mass': avg_adjustment
    })

df_metrics = pd.DataFrame(metrics)

# ==============================================================================
# 3. VISUALIZAÇÃO
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

# --- GRÁFICO A: TAXA DE FALHA VS LAMBDA ---
ax1 = fig.add_subplot(gs[0, 0])
sns.lineplot(data=df_metrics, x='lambda_1', y='failure_rate', marker='o', color='red', ax=ax1, linewidth=2)
ax1.fill_between(df_metrics['lambda_1'], df_metrics['failure_rate'], color='red', alpha=0.1)

ax1.set_title('Instabilidade Numérica: Taxa de Falha do Solver', fontsize=14, fontweight='bold')
ax1.set_xlabel('Lambda 1 (Peso WLS)', fontsize=12)
ax1.set_ylabel('% de Falhas (Fallback)', fontsize=12)
ax1.set_ylim(-5, 105)

# --- GRÁFICO B: AGRESSIVIDADE DOS AJUSTES ---
ax2 = fig.add_subplot(gs[0, 1])
sns.lineplot(data=df_metrics, x='lambda_1', y='avg_adjustment_mass', marker='s', color='blue', ax=ax2)
ax2.set_title('Agressividade Média: Magnitude das Correções', fontsize=14, fontweight='bold')
ax2.set_xlabel('Lambda 1', fontsize=12)
ax2.set_ylabel('Média de Ajuste (Energia/Massa Total)', fontsize=12)

# --- GRÁFICO C: HISTOGRAMA DE RESÍDUOS (L1 vs WLS) ---
ax3 = fig.add_subplot(gs[1, :])

# Pegar índices extremos e médio
idx_L1 = 0 # Lambda ~ 0
idx_Mix = len(lambda_values) // 2 # Lambda ~ 0.5

# Extrair dados para plotagem
df_L1 = df_rec_numeric.iloc[idx_L1*n_rows : (idx_L1+1)*n_rows].reset_index(drop=True)
df_Mix = df_rec_numeric.iloc[idx_Mix*n_rows : (idx_Mix+1)*n_rows].reset_index(drop=True)

# Calcular resíduos (Medido - Reconciliado)
residuos_L1 = (df_L1 - df_meas_numeric).values.flatten()
residuos_Mix = (df_Mix - df_meas_numeric).values.flatten()

# Filtrar zeros (falhas) para não distorcer o histograma
residuos_L1 = residuos_L1[np.abs(residuos_L1) > 1e-3]
residuos_Mix = residuos_Mix[np.abs(residuos_Mix) > 1e-3]

if len(residuos_L1) > 0 and len(residuos_Mix) > 0:
    sns.kdeplot(residuos_L1, ax=ax3, label=f'Lambda={lambda_values[idx_L1]:.4f} (L1 - Esparso)', fill=True, alpha=0.3, color='green')
    sns.kdeplot(residuos_Mix, ax=ax3, label=f'Lambda={lambda_values[idx_Mix]:.4f} (Híbrido)', fill=True, alpha=0.3, color='purple')
    
    # Zoom inteligente no eixo X para ignorar outliers extremos no plot
    limit = np.percentile(np.abs(residuos_Mix), 98) 
    ax3.set_xlim(-limit, limit)
    ax3.set_title('Distribuição dos Ajustes: Como o solver corrige os dados?', fontsize=14)
    ax3.set_xlabel('Magnitude do Ajuste (Medido - Reconciliado)', fontsize=12)
    ax3.legend()
else:
    ax3.text(0.5, 0.5, "Dados insuficientes para histograma (Muitas falhas ou ajustes zero)", 
             ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('diagnostico_falhas_lambda_v2.png')
print("\nGráfico salvo: diagnostico_falhas_lambda_v2.png")
print("\n=== RELATÓRIO SIMPLIFICADO ===")
print(df_metrics[['lambda_1', 'failure_rate', 'avg_adjustment_mass']].round(4).to_string(index=False))

plt.show()