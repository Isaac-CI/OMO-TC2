import pandas as pd
import numpy as np
import random
from scipy.stats import skewnorm

def simulate_multi_line_consumption(plant_config: dict, 
                                    annual_tonnage: float, 
                                    year: int = 2024,
                                    overall_variability: float = 0.10, 
                                    inter_unit_variability: float = 0.05,
                                    temporal_variability: float = 0.15,
                                    stage_noise_skew: float = -5.0, 
                                    stage_noise_scale: float = 0.05) -> pd.DataFrame:
    """
    Gera série temporal horária para Múltiplas Linhas de Produção independentes.

    Args:
        plant_config (dict): Configuração hierárquica:
                             {
                                'Nome Linha': {
                                    'production_share': 0.X (Opcional, float 0.0-1.0),
                                    'stages': {
                                        'Nome Etapa': {
                                            'Nome Maquina': {props...},
                                            ...
                                        }
                                    }
                                },
                                ...
                             }
        annual_tonnage (float): Produção total da planta inteira.
        ... (outros parâmetros de variabilidade iguais ao anterior)

    Returns:
        pd.DataFrame: DataFrame com colunas nomeadas como 'Linha_Etapa_Maquina #ID'.
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
    
    print(f"--- Iniciando Simulação Multi-Linhas ---")
    print(f"Produção Global: {annual_tonnage} t/ano ({global_hourly_rate:.2f} t/h)")

    # 2. Calcular Distribuição de Carga entre Linhas
    lines = list(plant_config.keys())
    defined_shares = {k: v.get('production_share') for k, v in plant_config.items() if v.get('production_share') is not None}
    
    sum_defined = sum(defined_shares.values())
    count_undefined = len(lines) - len(defined_shares)
    
    if sum_defined > 1.0:
        print("AVISO: Soma das proporções definidas > 100%. Normalizando...")
        # Lógica simples de normalização poderia ser adicionada aqui, mas vamos alertar.
    
    # Define o share padrão para quem não tem 'production_share' definido
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

        # LOOP DAS ETAPAS (Dentro da Linha)
        for stage_name, equipment_dict in stages_config.items():
            
            stage_machines_series = []
            
            # LOOP DOS EQUIPAMENTOS (Dentro da Etapa)
            for equip_name, props in equipment_dict.items():
                quantity = props.get('quantity', 0)
                base_spec = props.get('base_kWh_per_tonne', 0)
                schedule = props.get('schedule', [1.0] * 24)
                
                if len(schedule) != 24: schedule = [1.0] * 24
                if quantity <= 0 or base_spec <= 0: continue

                # Sazonalidade
                seasonality_vector = np.array([schedule[t.hour] for t in time_index])

                # Taxa de produção por UNIDADE desta máquina
                # (A carga da linha é dividida entre as máquinas paralelas)
                unit_production_rate = line_hourly_rate / quantity

                # Variabilidade Geral do Tipo
                std_dev_overall = base_spec * overall_variability
                type_simulated_spec = max(0, random.normalvariate(base_spec, std_dev_overall))
                
                # Carga Base Elétrica por Unidade (kWh/h)
                unit_base_load_hourly = type_simulated_spec * unit_production_rate

                for i in range(1, quantity + 1):
                    # Nomenclatura hierárquica: Linha -> Etapa -> Equipamento
                    col_name = f"{line_name}|{stage_name}|{equip_name} #{i}"
                    
                    # Variabilidade Inter-Unidade
                    std_dev_inter = unit_base_load_hourly * inter_unit_variability
                    this_unit_base = max(0, random.normalvariate(unit_base_load_hourly, std_dev_inter))
                    
                    # Ruído Temporal
                    noise = np.random.normal(loc=0, scale=temporal_variability, size=hours_in_simulation)
                    
                    # Cálculo Final Unidade
                    hourly_series = this_unit_base * seasonality_vector * (1 + noise)
                    hourly_series = np.maximum(hourly_series, 0)
                    
                    # Mascara de desligamento
                    mask_off = seasonality_vector == 0
                    hourly_series[mask_off] = 0
                    
                    data_dict[col_name] = hourly_series
                    stage_machines_series.append(hourly_series)

            # ---------------------------------------------------------
            # Cálculo da Etapa Agregada (Para esta Linha Específica)
            # ---------------------------------------------------------
            if stage_machines_series:
                raw_stage_sum = np.sum(stage_machines_series, axis=0)
                
                # Ruído Enviesado da Etapa
                skewed_noise = skewnorm.rvs(a=stage_noise_skew, loc=0, scale=stage_noise_scale, size=hours_in_simulation)
                
                stage_final_series = raw_stage_sum * (1 + skewed_noise)
                stage_final_series = np.maximum(stage_final_series, 0)
                stage_final_series[raw_stage_sum == 0] = 0
                
                # Nome da coluna de total da etapa na linha
                stage_col_name = f"{line_name}|{stage_name}|TOTAL"
                data_dict[stage_col_name] = stage_final_series

    # Cria DataFrame final
    df = pd.DataFrame(data_dict, index=time_index)
    return df