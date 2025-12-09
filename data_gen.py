import pandas as pd
import numpy as np
import random

def simulate_multi_line_consumption(plant_config: dict, 
                                    annual_tonnage: float, 
                                    year: int = 2024,
                                    overall_variability: float = 0.10, 
                                    inter_unit_variability: float = 0.05,
                                    temporal_variability: float = 0.15,
                                    stage_noise_scale: float = 0.05,
                                    plant_noise_scale: float = 0.02) -> pd.DataFrame:
    """
    Gera série temporal horária para Múltiplas Linhas de Produção independentes.
    Calcula também o TOTAL GLOBAL da planta com ruído independente.

    Args:
        plant_config (dict): Configuração hierárquica da planta.
        annual_tonnage (float): Produção total da planta inteira.
        year (int): Ano da simulação.
        overall_variability (float): Variabilidade do tipo de equipamento.
        inter_unit_variability (float): Variabilidade entre unidades iguais.
        temporal_variability (float): Ruído horário das máquinas.
        stage_noise_scale (float): Desvio padrão do ruído normal da etapa.
        plant_noise_scale (float): Desvio padrão do ruído normal aplicado ao total da planta.

    Returns:
        pd.DataFrame: DataFrame com colunas de máquinas, totais de etapa e TOTAL_PLANTA_GLOBAL.
    """
    
    # 1. Configuração de Tempo
    start_date = f'{year}-01-01'
    end_date = f'{year}-01-07 23:00'
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')
    hours_in_simulation = len(time_index)
    hours_in_year = 8784 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 8760
    
    # Taxa Global da Planta (t/h)
    global_hourly_rate = annual_tonnage / hours_in_year
    
    data_dict = {}
    
    # Acumulador para a soma pura de todas as máquinas da planta
    plant_aggregated_series = np.zeros(hours_in_simulation)
    
    print(f"--- Iniciando Simulação Multi-Linhas (Distruibuição Normal) ---")
    print(f"Produção Global: {annual_tonnage} t/ano ({global_hourly_rate:.2f} t/h)")

    # 2. Calcular Distribuição de Carga entre Linhas
    lines = list(plant_config.keys())
    defined_shares = {k: v.get('production_share') for k, v in plant_config.items() if v.get('production_share') is not None}
    
    sum_defined = sum(defined_shares.values())
    count_undefined = len(lines) - len(defined_shares)
    
    if sum_defined > 1.0:
        print("AVISO: Soma das proporções definidas > 100%. Normalizando...")
    
    default_share = (1.0 - sum_defined) / count_undefined if count_undefined > 0 else 0.0

    # ==========================================================================
    # LOOP DAS LINHAS DE PRODUÇÃO
    # ==========================================================================
    for line_name, line_data in plant_config.items():
        share = line_data.get('production_share', default_share)
        line_hourly_rate = global_hourly_rate * share
        stages_config = line_data.get('stages', {})
        
        print(f"Simulando '{line_name}': {share*100:.1f}% da carga ({line_hourly_rate:.2f} t/h)")

        if line_hourly_rate <= 0:
            continue

        # LOOP DAS ETAPAS
        for stage_name, equipment_dict in stages_config.items():
            
            stage_machines_series = []
            
            # LOOP DOS EQUIPAMENTOS
            for equip_name, props in equipment_dict.items():
                quantity = props.get('quantity', 0)
                base_spec = props.get('base_kWh_per_tonne', 0)
                schedule = props.get('schedule', [1.0] * 24)
                
                if len(schedule) != 24: schedule = [1.0] * 24
                if quantity <= 0 or base_spec <= 0: continue

                # Sazonalidade
                seasonality_vector = np.array([schedule[t.hour] for t in time_index])

                # Taxa de produção por UNIDADE
                unit_production_rate = line_hourly_rate / quantity

                # Variabilidade Geral do Tipo
                std_dev_overall = base_spec * overall_variability
                type_simulated_spec = max(0, random.normalvariate(base_spec, std_dev_overall))
                
                # Carga Base Elétrica por Unidade (kWh/h)
                unit_base_load_hourly = type_simulated_spec * unit_production_rate

                for i in range(1, quantity + 1):
                    col_name = f"{line_name}|{stage_name}|{equip_name} #{i}"
                    
                    # Variabilidade Inter-Unidade
                    std_dev_inter = unit_base_load_hourly * inter_unit_variability
                    this_unit_base = max(0, random.normalvariate(unit_base_load_hourly, std_dev_inter))
                    
                    # Ruído Temporal
                    noise = np.random.normal(loc=0, scale=temporal_variability, size=hours_in_simulation)
                    
                    # Cálculo Final Unidade
                    hourly_series = this_unit_base * seasonality_vector * (1 + noise)
                    hourly_series = np.maximum(hourly_series, 0)
                    
                    # Máscara de desligamento
                    mask_off = seasonality_vector == 0
                    hourly_series[mask_off] = 0
                    
                    # Armazenar dados
                    data_dict[col_name] = hourly_series
                    stage_machines_series.append(hourly_series)
                    
                    # Acumular no total global da planta
                    plant_aggregated_series += hourly_series

            # ---------------------------------------------------------
            # Cálculo da Etapa Agregada
            # ---------------------------------------------------------
            if stage_machines_series:
                raw_stage_sum = np.sum(stage_machines_series, axis=0)
                
                # Ruído Normal da Etapa (Simétrico)
                # Centrado em 0, com desvio padrão definido por stage_noise_scale
                normal_noise = np.random.normal(loc=0, scale=stage_noise_scale, size=hours_in_simulation)
                normal_noise = np.maximum(normal_noise, -0.95) # Satura valor negativo máximo para normal noise em -95% impedindo 0s espúrios
                
                stage_final_series = raw_stage_sum * (1 + normal_noise)
                stage_final_series = np.maximum(stage_final_series, 0)
                stage_final_series[raw_stage_sum == 0] = 0
                
                stage_col_name = f"{line_name}|{stage_name}|TOTAL"
                data_dict[stage_col_name] = stage_final_series

    # ==========================================================================
    # CÁLCULO DO TOTAL GLOBAL DA PLANTA
    # ==========================================================================
    
    # Gera ruído normal (Gaussiano)
    plant_noise = np.random.normal(loc=0, scale=plant_noise_scale, size=hours_in_simulation)
    plant_noise = np.maximum(plant_noise, -0.95) # Satura valor negativo máximo para plant noise em -95% impedindo 0s espúrios
    
    # Aplica o ruído: Total = Soma_Equipamentos * (1 + Ruído)
    plant_final_total = plant_aggregated_series * (1 + plant_noise)
    plant_final_total = np.maximum(plant_final_total, 0) # Garante não-negativo
    
    # Onde a soma dos equipamentos é 0 (planta parada), o total deve ser 0
    plant_final_total[plant_aggregated_series == 0] = 0
    
    data_dict["TOTAL_PLANTA_GLOBAL"] = plant_final_total

    # Cria DataFrame final
    df = pd.DataFrame(data_dict, index=time_index)
    return df