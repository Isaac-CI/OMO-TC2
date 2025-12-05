import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração de Estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_and_structure_data(csv_file):
    """
    Lê o CSV e cria uma estrutura de metadados para facilitar o agrupamento.
    """
    if not os.path.exists(csv_file):
        print(f"Erro: Arquivo '{csv_file}' não encontrado.")
        return None, None

    print(f"Lendo dados de {csv_file}...")
    df = pd.read_csv(csv_file, index_col='Data_Hora', parse_dates=True)
    
    # Filtrar colunas de TOTAL pré-calculado para evitar dupla contagem
    # Vamos trabalhar com a soma dos equipamentos individuais
    machine_cols = [c for c in df.columns if "|TOTAL" not in c]
    df_machines = df[machine_cols]

    # Criar um DataFrame auxiliar de metadados das colunas
    # Estrutura: Coluna_Original | Planta | Etapa | Equipamento
    meta_data = []
    for col in machine_cols:
        parts = col.split('|')
        if len(parts) == 3:
            meta_data.append({
                'Col': col,
                'Planta': parts[0],
                'Etapa': parts[1],
                'Equipamento': parts[2]
            })
    
    df_meta = pd.DataFrame(meta_data)
    return df_machines, df_meta

def plot_total_energy_by_plant(df, df_meta):
    """Gráfico de Barras: Energia Total consumida por cada Planta no período."""
    print("Gerando gráfico: Energia Total por Planta...")
    
    # Agrupar colunas por Planta
    plant_groups = df_meta.groupby('Planta')['Col'].apply(list)
    
    plant_totals = {}
    for plant, cols in plant_groups.items():
        # Soma (kWh) -> Converte para GWh
        plant_totals[plant] = df[cols].sum().sum() / 1e6 

    ser_totals = pd.Series(plant_totals).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=ser_totals.values, y=ser_totals.index, palette="viridis")
    
    plt.title('Consumo Total de Energia por Planta (GWh)', fontsize=16)
    plt.xlabel('Energia (GWh)')
    plt.ylabel('Planta')
    
    # Adicionar valores nas barras
    for i, v in enumerate(ser_totals.values):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig('images/grafico_1_total_por_planta.png')
    plt.show()

def plot_hourly_profile_by_plant(df, df_meta):
    """Gráfico de Linha: Perfil Médio Horário (Curva de Carga Diária)."""
    print("Gerando gráfico: Perfil de Carga Diário...")
    
    # Criar DataFrame agregado por planta
    df_plants = pd.DataFrame(index=df.index)
    for plant in df_meta['Planta'].unique():
        cols = df_meta[df_meta['Planta'] == plant]['Col']
        df_plants[plant] = df[cols].sum(axis=1) / 1000 # MW

    # Agrupar por hora do dia (0-23) e tirar a média
    df_hourly_avg = df_plants.groupby(df_plants.index.hour).mean()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_hourly_avg, dashes=False, linewidth=2.5)
    
    plt.title('Perfil de Carga Médio Diário por Planta (Sazonalidade)', fontsize=16)
    plt.xlabel('Hora do Dia')
    plt.ylabel('Potência Média (MW)')
    plt.xticks(range(0, 24))
    plt.legend(title='Planta', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('images/grafico_2_perfil_diario.png')
    plt.show()

def plot_energy_heatmap(df, df_meta):
    """Heatmap: Consumo por Planta vs Etapa."""
    print("Gerando gráfico: Mapa de Calor Planta x Etapa...")
    
    # Calcular total por (Planta, Etapa)
    heatmap_data = []
    
    groups = df_meta.groupby(['Planta', 'Etapa'])['Col'].apply(list)
    
    for (plant, stage), cols in groups.items():
        total_mwh = df[cols].sum().sum() / 1000 # MWh
        heatmap_data.append({'Planta': plant, 'Etapa': stage, 'Consumo_MWh': total_mwh})
    
    df_heat = pd.DataFrame(heatmap_data)
    
    # Pivotar para formato matriz
    df_matrix = df_heat.pivot(index="Planta", columns="Etapa", values="Consumo_MWh")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_matrix, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=.5)
    
    plt.title('Mapa de Calor: Consumo de Energia (MWh) por Etapa', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/grafico_3_heatmap_etapas.png')
    plt.show()

def plot_load_duration_curve(df, df_meta):
    """Curva de Duração de Carga (Load Duration Curve) do Complexo Inteiro."""
    print("Gerando gráfico: Curva de Duração de Carga...")
    
    # Soma total de todas as máquinas a cada hora
    total_load_complex = df.sum(axis=1) / 1000 # MW
    
    # Ordenar do maior para o menor
    sorted_load = total_load_complex.sort_values(ascending=False).reset_index(drop=True)
    
    # Eixo X em porcentagem do tempo
    x_axis = (sorted_load.index / len(sorted_load)) * 100
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(x_axis, sorted_load, color="skyblue", alpha=0.4)
    plt.plot(x_axis, sorted_load, color="Slateblue", alpha=0.6, linewidth=2)
    
    plt.title('Curva de Duração de Carga - Complexo Siderúrgico', fontsize=16)
    plt.xlabel('Duração (% do Tempo)')
    plt.ylabel('Demanda Total (MW)')
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    
    # Anotações
    peak = sorted_load.max()
    base = sorted_load.min()
    plt.annotate(f'Pico: {peak:.1f} MW', xy=(0, peak), xytext=(10, peak),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Base: {base:.1f} MW', xy=(100, base), xytext=(80, base*1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('images/grafico_4_curva_duracao.png')
    plt.show()

def main():
    # Nome do arquivo gerado pelo script de simulação
    # Certifique-se de que este nome bate com o gerado no 'run_rota1_detailed_simulation.py'
    csv_filename = "data/resultado_complexo_siderurgico.csv" 
    
    df, df_meta = load_and_structure_data(csv_filename)
    
    if df is not None:
        # 1. Total por Planta (Ranking)
        plot_total_energy_by_plant(df, df_meta)
        
        # 2. Perfil Diário (Mostra turnos e horários de ponta)
        plot_hourly_profile_by_plant(df, df_meta)
        
        # 3. Heatmap (Onde está o gasto em cada planta?)
        plot_energy_heatmap(df, df_meta)
        
        # 4. Curva de Carga (Dimensionamento elétrico)
        plot_load_duration_curve(df, df_meta)
        
        print("\nVisualização concluída. Gráficos salvos como PNG.")

if __name__ == "__main__":
    main()