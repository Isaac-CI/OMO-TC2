"""
Arquivo de configuração para o Complexo Siderúrgico (Apenas Planta 5).
Define perfis de operação e a estrutura detalhada de equipamentos para a Planta 5 assumindo toda a produção.
"""

# ==============================================================================
# 1. PERFIS DE OPERAÇÃO (SAZONALIDADE)
# ==============================================================================

# Perfil 24h Contínuo (Alto Forno, Coqueria)
profile_cont = [1.0] * 24

# Perfil Turnos (12h operação: 06h as 18h)
profile_day_shift = [0.0]*6 + [1.0]*12 + [0.0]*6 

# Perfil Cíclico (Aciaria - Variação de carga)
profile_batch = [0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 0.8, 1.2] * 3

# ==============================================================================
# 2. FUNÇÕES GERADORAS DE ESTRUTURA
# ==============================================================================

def get_base_structure(efficiency_factor=1.0):
    """
    Retorna a estrutura PADRÃO da Rota 1.
    efficiency_factor: Multiplicador para o consumo (1.0 = padrão, >1.0 = menos eficiente).
    """
    ef = efficiency_factor
    
    return {
        # --- ETAPA 1: COMINUIÇÃO ---
        'Cominuicao': {
            'Britador Blake (Mandibulas)': {'quantity': 1, 'base_kWh_per_tonne': 80.0 * ef, 'schedule': profile_day_shift},
            'Britador Conico':             {'quantity': 1, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_day_shift},
            'Britador Bradford (Carvao)':  {'quantity': 1, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_day_shift},
            'Moinho de Gaiolas':           {'quantity': 1, 'base_kWh_per_tonne': 112.0 * ef, 'schedule': profile_day_shift},
            'Moinho de Bolas':             {'quantity': 2, 'base_kWh_per_tonne': 564.0 * ef, 'schedule': profile_day_shift}
        },
        # --- ETAPA 2: SINTERIZAÇÃO ---
        'Sinterizacao': {
            'Grelha de Sinterizacao':      {'quantity': 1, 'base_kWh_per_tonne': 170.0 * ef, 'schedule': profile_cont},
            'Exaustores Principais':       {'quantity': 2, 'base_kWh_per_tonne': 1000.0 * ef, 'schedule': profile_cont}
        },
        # --- ETAPA 3: COQUEIFICAÇÃO ---
        'Coqueificacao': {
            'Correia Transp. Bunker':      {'quantity': 1, 'base_kWh_per_tonne': 23.0 * ef, 'schedule': profile_cont},
            'Maquinas da Bateria':         {'quantity': 2, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_cont},
            'Silos de Carregamento':       {'quantity': 1, 'base_kWh_per_tonne': 17.0 * ef, 'schedule': profile_cont},
            'Extincao (Bombas Spray)':     {'quantity': 2, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_batch}
        },
        # --- ETAPA 4: ALTO FORNO ---
        'Alto_Forno': {
            'Sistemas de Carga':           {'quantity': 1, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_cont},
            'Sopradores Ventaneiras':      {'quantity': 3, 'base_kWh_per_tonne': 732.0 * ef, 'schedule': profile_cont},
            'Fogoes Cowper':               {'quantity': 3, 'base_kWh_per_tonne': 23.0 * ef, 'schedule': profile_cont},
            'Refrigeracao (Staves)':       {'quantity': 2, 'base_kWh_per_tonne': 169.0 * ef, 'schedule': profile_cont},
            'Sensores e Controladora':     {'quantity': 1, 'base_kWh_per_tonne': 6.0 * ef, 'schedule': profile_cont}
        },
        # --- ETAPA 5: ACIARIA ---
        'Aciaria': {
            'Convertedor LD':              {'quantity': 2, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_batch},
            'Forno Panela':                {'quantity': 2, 'base_kWh_per_tonne': 450.0 * ef, 'schedule': profile_batch}
        },
        # --- ETAPA 6: LINGOTAMENTO ---
        'Lingotamento': {
            'Torre de Panela':             {'quantity': 1, 'base_kWh_per_tonne': 11.0 * ef, 'schedule': profile_cont},
            'Distribuidor/Lingoteira':     {'quantity': 2, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_cont},
            'Sprays Resfriamento':         {'quantity': 2, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_cont},
            'Maquina de Corte':            {'quantity': 2, 'base_kWh_per_tonne': 34.0 * ef, 'schedule': profile_cont}
        },
        # --- ETAPA 7: LAMINAÇÃO ---
        'Laminacao': {
            'Laminador Morgan #1 motor':            {'quantity': 3, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #2 motor':            {'quantity': 3, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #3 motor':            {'quantity': 3, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #4 motor':            {'quantity': 3, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #5 motor':            {'quantity': 3, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont}
        }
    }

def get_scaled_structure(efficiency_factor=1.0):
    """
    Retorna a estrutura ESCALADA (aprox. 3x maior) da Rota 1.
    """
    ef = efficiency_factor
    
    return {
        # --- ETAPA 1: COMINUIÇÃO ---
        'Cominuicao': {
            'Britador Blake (Mandibulas)': {'quantity': 3, 'base_kWh_per_tonne': 80.0 * ef, 'schedule': profile_day_shift},
            'Britador Conico':             {'quantity': 3, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_day_shift},
            'Britador Bradford (Carvao)':  {'quantity': 3, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_day_shift},
            'Moinho de Gaiolas':           {'quantity': 3, 'base_kWh_per_tonne': 112.0 * ef, 'schedule': profile_day_shift},
            'Moinho de Bolas':             {'quantity': 6, 'base_kWh_per_tonne': 564.0 * ef, 'schedule': profile_day_shift}
        },
        # --- ETAPA 2: SINTERIZAÇÃO ---
        'Sinterizacao': {
            'Grelha de Sinterizacao':      {'quantity': 3, 'base_kWh_per_tonne': 170.0 * ef, 'schedule': profile_cont},
            'Exaustores Principais':       {'quantity': 6, 'base_kWh_per_tonne': 1000.0 * ef, 'schedule': profile_cont}
        },
        # --- ETAPA 3: COQUEIFICAÇÃO ---
        'Coqueificacao': {
            'Correia Transp. Bunker':      {'quantity': 3, 'base_kWh_per_tonne': 23.0 * ef, 'schedule': profile_cont},
            'Maquinas da Bateria':         {'quantity': 6, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_cont},
            'Silos de Carregamento':       {'quantity': 3, 'base_kWh_per_tonne': 17.0 * ef, 'schedule': profile_cont},
            'Extincao (Bombas Spray)':     {'quantity': 6, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_batch}
        },
        # --- ETAPA 4: ALTO FORNO ---
        'Alto_Forno': {
            'Sistemas de Carga':           {'quantity': 3, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_cont},
            'Sopradores Ventaneiras':      {'quantity': 9, 'base_kWh_per_tonne': 732.0 * ef, 'schedule': profile_cont},
            'Fogoes Cowper':               {'quantity': 9, 'base_kWh_per_tonne': 23.0 * ef, 'schedule': profile_cont},
            'Refrigeracao (Staves)':       {'quantity': 6, 'base_kWh_per_tonne': 169.0 * ef, 'schedule': profile_cont},
            'Sensores e Controladora':     {'quantity': 3, 'base_kWh_per_tonne': 6.0 * ef, 'schedule': profile_cont}
        },
        # --- ETAPA 5: ACIARIA ---
        'Aciaria': {
            'Convertedor LD':              {'quantity': 6, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_batch},
            'Forno Panela':                {'quantity': 6, 'base_kWh_per_tonne': 450.0 * ef, 'schedule': profile_batch}
        },
        # --- ETAPA 6: LINGOTAMENTO ---
        'Lingotamento': {
            'Torre de Panela':             {'quantity': 3, 'base_kWh_per_tonne': 11.0 * ef, 'schedule': profile_cont},
            'Distribuidor/Lingoteira':     {'quantity': 6, 'base_kWh_per_tonne': 56.0 * ef, 'schedule': profile_cont},
            'Sprays Resfriamento':         {'quantity': 6, 'base_kWh_per_tonne': 90.0 * ef, 'schedule': profile_cont},
            'Maquina de Corte':            {'quantity': 6, 'base_kWh_per_tonne': 34.0 * ef, 'schedule': profile_cont}
        },
        # --- ETAPA 7: LAMINAÇÃO ---
        'Laminacao': {
            'Laminador Morgan #1 motor':            {'quantity': 9, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #2 motor':            {'quantity': 9, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #3 motor':            {'quantity': 9, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #4 motor':            {'quantity': 9, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont},
            'Laminador Morgan #5 motor':            {'quantity': 9, 'base_kWh_per_tonne': 110.0 * ef, 'schedule': profile_cont}
        }
    }

# ==============================================================================
# 3. DEFINIÇÃO DAS PLANTAS (CUSTOMIZAÇÕES)
# ==============================================================================

# Planta 1: Referência (Eficiência Padrão)
planta_matriz = get_base_structure(efficiency_factor=1.0)

# Planta 2: Norte (Planta Antiga - 10% menos eficiente no consumo)
planta_norte = get_base_structure(efficiency_factor=1.10) # 10% menos eficiente
planta_norte['Alto_Forno']['Sopradores Ventaneiras']['quantity'] = 4

# Planta 3: Sul (Planta Moderna/Otimizada - 5% mais eficiente)
planta_sul = get_base_structure(efficiency_factor=0.95) # 5% mais eficiente
planta_sul['Aciaria']['Convertedor LD']['quantity'] = 3

# Planta 4: Leste (Focada em Acabamento)
planta_leste = get_scaled_structure(efficiency_factor=1.0) # Planta Escalada (Grande porte)

# --- NOVAS PLANTAS ADICIONADAS ---
planta_oeste = get_scaled_structure(efficiency_factor=0.92) # Planta Gigante e muito moderna
planta_sudeste = get_base_structure(efficiency_factor=1.0) # Planta Padrão
planta_noroeste = get_base_structure(efficiency_factor=1.20) # Planta Antiga e Ineficiente
planta_centro = get_scaled_structure(efficiency_factor=1.05) # Planta Grande e levemente ineficiente

# ==============================================================================
# 4. CONFIGURAÇÃO FINAL EXPORTADA
# ==============================================================================

COMPLEXO_SIDERURGICO_CONFIG = {
    # 1. Planta Base - DESATIVADA (0%)
    'Planta 1 (Matriz)': {
        'production_share': 0.00,
        'stages': planta_matriz
    },
    # 2. Planta Baixa Eficiência - DESATIVADA (0%)
    'Planta 2 (Norte - Antiga)': {
        'production_share': 0.00,
        'stages': planta_norte
    },
    # 3. Planta Alta Eficiência - DESATIVADA (0%)
    'Planta 3 (Sul - Moderna)': {
        'production_share': 0.00,
        'stages': planta_sul
    },
    # 4. Planta Escalada - DESATIVADA (0%)
    'Planta 4 (Leste - Escalada)': {
        'production_share': 0.00,
        'stages': planta_leste
    },
    # 5. Planta Gigante Moderna - ATIVA (100%)
    'Planta 5 (Oeste - Gigante)': {
        'production_share': 1.00,
        'stages': planta_oeste
    },
    # 6. Planta Padrão Auxiliar - DESATIVADA (0%)
    'Planta 6 (Sudeste)': {
        'production_share': 0.00,
        'stages': planta_sudeste
    },
    # 7. Planta Legada - DESATIVADA (0%)
    'Planta 7 (Noroeste - Legado)': {
        'production_share': 0.00,
        'stages': planta_noroeste
    },
    # 8. Planta Exportação - DESATIVADA (0%)
    'Planta 8 (Centro - Export)': {
        'production_share': 0.00,
        'stages': planta_centro
    }
}