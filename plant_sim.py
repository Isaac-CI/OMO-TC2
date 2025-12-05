import pandas as pd
import sys

# Importa a função de simulação
try:
    from data_gen import simulate_multi_line_consumption
except ImportError:
    print("Erro Crítico: Arquivo 'data_gen.py' não encontrado.")
    sys.exit(1)

# Importa a configuração das plantas
try:
    from plant_config import COMPLEXO_SIDERURGICO_CONFIG
except ImportError:
    print("Erro Crítico: Arquivo 'plant_config.py' não encontrado.")
    sys.exit(1)

# ==============================================================================
# PARÂMETROS DA SIMULAÇÃO
# ==============================================================================
TONELADAS_ANO = 5_000_000 # 5 Milhões de toneladas
ANO_SIMULACAO = 2025

print(f"--- Iniciando Simulação do Complexo Siderúrgico ({ANO_SIMULACAO}) ---")
print(f"Capacidade Total: {TONELADAS_ANO/1e6} Mt/ano")

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
try:
    df_resultado = simulate_multi_line_consumption(
        plant_config=COMPLEXO_SIDERURGICO_CONFIG,
        annual_tonnage=TONELADAS_ANO,
        year=ANO_SIMULACAO,
        # Variabilidades do sistema
        overall_variability=0.08,    
        inter_unit_variability=0.04, 
        temporal_variability=0.15,   
        stage_noise_skew=-3.0,       
        stage_noise_scale=0.05
    )

    # Salvar resultados
    nome_arquivo = "data/resultado_complexo_siderurgico.csv"
    df_resultado.to_csv(nome_arquivo, index_label="Data_Hora")
    print(f"\n[SUCESSO] Dados salvos em '{nome_arquivo}'")

    # ==============================================================================
    # RELATÓRIO DE CONSUMO
    # ==============================================================================
    
    # Cálculos de tempo e produção efetiva no período simulado
    horas_simuladas = len(df_resultado)
    horas_no_ano = 8784 if (ANO_SIMULACAO % 4 == 0 and ANO_SIMULACAO % 100 != 0) or (ANO_SIMULACAO % 400 == 0) else 8760
    taxa_producao_horaria = TONELADAS_ANO / horas_no_ano
    toneladas_periodo = taxa_producao_horaria * horas_simuladas

    print(f"\n--- Relatório Executivo ({horas_simuladas} horas simuladas) ---")
    print(f"Produção Estimada no Período: {toneladas_periodo:.2f} toneladas")
    
    print("\n" + "="*100)
    print(f"{'PLANTA':<30} | {'PROD. (t)':<12} | {'CONSUMO ETAPAS (MWh)':<22} | {'SPECIFIC (kWh/t)':<18}")
    print("="*100)

    totais = df_resultado.sum()
    plantas = list(COMPLEXO_SIDERURGICO_CONFIG.keys())
    
    total_energia_global = 0

    for planta in plantas:
        # Filtra colunas de totais de etapa para esta planta
        cols_etapas = [c for c in totais.index if c.startswith(planta) and "TOTAL" in c]
        
        if not cols_etapas:
            print(f"Aviso: Nenhum dado encontrado para {planta}")
            continue

        consumo_mwh = sum(totais[c] for c in cols_etapas) / 1000
        
        # Calcula produção específica desta planta
        share = COMPLEXO_SIDERURGICO_CONFIG[planta]['production_share']
        prod_planta = toneladas_periodo * share
        
        spec_kwh_t = (consumo_mwh * 1000) / prod_planta if prod_planta > 0 else 0
        
        print(f"{planta:<30} | {prod_planta:<12.0f} | {consumo_mwh:<22.2f} | {spec_kwh_t:<18.2f}")
        
        total_energia_global += consumo_mwh

    print("-" * 100)
    spec_global = (total_energia_global * 1000) / toneladas_periodo
    print(f"{'TOTAL COMPLEXO':<30} | {toneladas_periodo:<12.0f} | {total_energia_global:<22.2f} | {spec_global:<18.2f}")
    print("=" * 100)

    # Detalhamento rápido da Planta 4 para verificar a customização
    print(f"\n--- Validação Customização: {plantas[3]} ---")
    lam_cols = [c for c in totais.index if plantas[3] in c and "Laminador Morgan" in c]
    print(f"Máquinas de Laminação encontradas: {len(lam_cols)}")
    for col in lam_cols:
        val = totais[col] / 1000
        print(f"  > {col.split('|')[-1]}: {val:.2f} MWh")

except Exception as e:
    print(f"\nOcorreu um erro durante a simulação:\n{e}")