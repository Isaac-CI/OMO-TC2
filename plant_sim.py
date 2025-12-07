import pandas as pd
import sys

# Importa a função de simulação
try:
    from data_gen import simulate_multi_line_consumption
except ImportError:
    print("Erro Crítico: Arquivo 'multi_line_simulator.py' não encontrado.")
    sys.exit(1)

# Importa a configuração das plantas
try:
    from plant_config import COMPLEXO_SIDERURGICO_CONFIG
except ImportError:
    print("Erro Crítico: Arquivo 'plants_config.py' não encontrado.")
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
        overall_variability=0.15,    
        inter_unit_variability=0.11, 
        temporal_variability=0.15,   
        stage_noise_scale=0.5,
        plant_noise_scale=0.35 # Variabilidade do medidor global da planta
    )

    # Salvar resultados
    nome_arquivo = "resultado_complexo_siderurgico.csv"
    df_resultado.to_csv(nome_arquivo, index_label="Data_Hora")
    print(f"\n[SUCESSO] Dados salvos em '{nome_arquivo}'")

    # ==============================================================================
    # RELATÓRIO EXECUTIVO (MACRO)
    # ==============================================================================
    
    # Cálculos de tempo e produção efetiva no período simulado
    horas_simuladas = len(df_resultado)
    horas_no_ano = 8784 if (ANO_SIMULACAO % 4 == 0 and ANO_SIMULACAO % 100 != 0) or (ANO_SIMULACAO % 400 == 0) else 8760
    taxa_producao_horaria = TONELADAS_ANO / horas_no_ano
    toneladas_periodo = taxa_producao_horaria * horas_simuladas

    print(f"\n--- Relatório Executivo ({horas_simuladas} horas simuladas) ---")
    print(f"Produção Estimada no Período: {toneladas_periodo:.2f} toneladas")
    
    print("\n" + "="*125)
    print(f"{'PLANTA':<25} | {'PROD. (t)':<10} | {'SOMA ETAPAS':<14} | {'TOTAL PLANTA':<14} | {'DIF (%)':<8} | {'SPEC (kWh/t)':<12}")
    print("="*125)

    totais = df_resultado.sum()
    plantas = list(COMPLEXO_SIDERURGICO_CONFIG.keys())
    
    # Totais globais para o complexo inteiro
    complexo_soma_etapas = 0
    complexo_total_planta = 0

    for planta in plantas:
        # 1. Soma das estimativas das ETAPAS (já com ruído de etapa)
        cols_etapas = [c for c in totais.index if c.startswith(planta) and "|TOTAL" in c]
        
        if not cols_etapas:
            print(f"{planta:<25} | {'N/A':<10} | {'Sem Dados':<14} | ...")
            continue

        soma_etapas_mwh = sum(totais[c] for c in cols_etapas) / 1000
        
        # 2. Consumo GLOBAL da planta (simulado independentemente - usando soma de equipamentos como proxy base física)
        cols_equip = [c for c in totais.index if c.startswith(planta) and "|TOTAL" not in c]
        consumo_planta_estimado_mwh = sum(totais[c] for c in cols_equip) / 1000 
        
        # Calcula produção específica desta planta
        share = COMPLEXO_SIDERURGICO_CONFIG[planta]['production_share']
        prod_planta = toneladas_periodo * share
        
        # Cálculo de diferença e específico
        spec_kwh_t = (consumo_planta_estimado_mwh * 1000) / prod_planta if prod_planta > 0 else 0
        diff_percent = ((consumo_planta_estimado_mwh - soma_etapas_mwh) / soma_etapas_mwh) * 100 if soma_etapas_mwh > 0 else 0
        
        print(f"{planta:<25} | {prod_planta:<10.0f} | {soma_etapas_mwh:<14.2f} | {consumo_planta_estimado_mwh:<14.2f} | {diff_percent:>7.2f}% | {spec_kwh_t:<12.2f}")
        
        complexo_soma_etapas += soma_etapas_mwh
        complexo_total_planta += consumo_planta_estimado_mwh

    print("-" * 125)
    
    # Totais Globais do Complexo
    if "TOTAL_PLANTA_GLOBAL" in totais:
        total_complexo_medido = totais["TOTAL_PLANTA_GLOBAL"] / 1000
    else:
        total_complexo_medido = complexo_total_planta 

    spec_global = (total_complexo_medido * 1000) / toneladas_periodo
    diff_global = ((total_complexo_medido - complexo_soma_etapas) / complexo_soma_etapas) * 100
    
    print(f"{'TOTAL COMPLEXO':<25} | {toneladas_periodo:<10.0f} | {complexo_soma_etapas:<14.2f} | {total_complexo_medido:<14.2f} | {diff_global:>7.2f}% | {spec_global:<12.2f}")
    print("=" * 125)

    # ==============================================================================
    # RELATÓRIO DETALHADO POR ETAPA E EQUIPAMENTO
    # ==============================================================================
    print("\n" + "="*80)
    print("DETALHAMENTO TÉCNICO: ETAPAS vs EQUIPAMENTOS")
    print("="*80)
    
    for planta in plantas:
        print(f"\n>>> {planta}")
        print("-" * 80)
        
        # Identificar etapas únicas para esta planta
        # As colunas são nomeadas como "Planta|Etapa|Equipamento"
        cols_planta = [c for c in totais.index if c.startswith(planta)]
        
        # Extrai nomes das etapas (segundo elemento do split)
        etapas_identificadas = sorted(list(set(c.split('|')[1] for c in cols_planta if len(c.split('|')) >= 2)))
        
        for etapa in etapas_identificadas:
            col_total_etapa = f"{planta}|{etapa}|TOTAL"
            
            # Filtra colunas de equipamentos desta etapa específica
            # Deve conter a etapa, pertencer à planta, e NÃO ser a coluna de total
            cols_eq = [c for c in cols_planta if f"|{etapa}|" in c and c != col_total_etapa]
            
            if not cols_eq: continue # Pula se não houver equipamentos (caso de erro)

            # Valores em MWh
            val_etapa = totais.get(col_total_etapa, 0) / 1000
            val_soma_eq = sum(totais[c] for c in cols_eq) / 1000
            
            diferenca = val_etapa - val_soma_eq
            diff_pct = (diferenca / val_soma_eq * 100) if val_soma_eq > 0 else 0
            
            print(f"  [{etapa}]")
            print(f"    Estimativa Etapa (Simulada): {val_etapa:10.2f} MWh")
            print(f"    Soma Equipamentos (Física):  {val_soma_eq:10.2f} MWh")
            print(f"    Diferença (Perdas/Ruído):    {diferenca:10.2f} MWh ({diff_pct:+.2f}%)")
            print(f"    ... Detalhe Equipamentos:")
            
            for c_eq in cols_eq:
                nome_equip = c_eq.split('|')[-1]
                val_eq = totais[c_eq] / 1000
                print(f"        - {nome_equip:<30}: {val_eq:8.2f} MWh")
            print("")

except Exception as e:
    print(f"\nOcorreu um erro durante a simulação:\n{e}")