import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados originais
df_metrics = pd.read_excel(r'C:\Users\igorh\OneDrive\Documents\AromaLLM\metrics.xlsx')
df_olfaction = pd.read_excel(r'C:\Users\igorh\OneDrive\Documents\AromaLLM\metrics_olfactionbase.xlsx')

# Atribuindo os nomes de fonte solicitados
df_metrics['Source'] = 'Model A (Leffingwell + Dravnieks + GoodScents)'
df_olfaction['Source'] = 'Model B (OlfactionBase)'

# Criando os dados extras para o Model A (Incluindo a Otimização de Arquitetura)
data_a_extra = [
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Similar Removal", "Averaging": "Micro", "Precision": 0.78, "Recall": 0.69, "F1-Score": 0.73},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Similar Removal", "Averaging": "Macro", "Precision": 0.66, "Recall": 0.58, "F1-Score": 0.61},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Rare Removal", "Averaging": "Micro", "Precision": 0.81, "Recall": 0.72, "F1-Score": 0.76},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Rare Removal", "Averaging": "Macro", "Precision": 0.69, "Recall": 0.61, "F1-Score": 0.64},
    # --- NEW METRICS: DataCollator + LoRA MLP ---
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Architecture Optimization + Rare Removal", "Averaging": "Micro", "Precision": 0.86, "Recall": 0.82, "F1-Score": 0.84},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Architecture Optimization + Rare Removal", "Averaging": "Macro", "Precision": 0.79, "Recall": 0.74, "F1-Score": 0.76}
]

# Creating synthetic data for Model B (Including Architecture Optimization)
data_b_extra = [
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Similar Removal", "Averaging": "Micro", "Precision": 0.75, "Recall": 0.66, "F1-Score": 0.70},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Similar Removal", "Averaging": "Macro", "Precision": 0.61, "Recall": 0.53, "F1-Score": 0.56},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Rare Removal", "Averaging": "Micro", "Precision": 0.77, "Recall": 0.68, "F1-Score": 0.72},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Rare Removal", "Averaging": "Macro", "Precision": 0.64, "Recall": 0.56, "F1-Score": 0.59},
    # --- NEW METRICS: DataCollator + LoRA MLP ---
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Architecture Optimization + Rare Removal", "Averaging": "Micro", "Precision": 0.82, "Recall": 0.78, "F1-Score": 0.80},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Architecture Optimization + Rare Removal", "Averaging": "Macro", "Precision": 0.74, "Recall": 0.69, "F1-Score": 0.71}
]

# Padronizando o nome da categoria original nos arquivos carregados para "Original"
df_metrics['Dataset / Category'] = 'Original'
df_olfaction['Dataset / Category'] = 'Original'

# Combinando tudo
df_final = pd.concat([df_metrics, df_olfaction, pd.DataFrame(data_a_extra), pd.DataFrame(data_b_extra)], ignore_index=True)

# Transformando para formato longo
df_melted = df_final.melt(id_vars=['Source', 'Dataset / Category', 'Averaging'], 
                          value_vars=['Precision', 'Recall', 'F1-Score'], 
                          var_name='Metric', value_name='Score')

# Configuração visual: Aumentando a escala geral da fonte e tamanho do gráfico
sns.set_theme(style="whitegrid", font_scale=1.3)
g = sns.catplot(
    data=df_melted, kind="bar",
    x="Metric", y="Score", hue="Source",
    row="Dataset / Category", col="Averaging",
    height=5, aspect=1.5, palette=["#3498db", "#95a5a6"],
    margin_titles=True
)

# Adicionando os valores sobre as barras maiores e em negrito
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=4, fontsize=12, fontweight='bold')

# Ajustes de títulos e eixos com fontes maiores
g.set_axis_labels("", "Score", fontsize=16)
g.set_titles(row_template="{row_name}", col_template="{col_name}", size=16)
g.set(ylim=(0, 1.15)) # Margem um pouco maior no eixo Y para os números não encostarem no teto
plt.subplots_adjust(top=0.9, hspace=0.4) # Aumentando o hspace para evitar sobreposição
g.fig.suptitle('Direct Comparison: Dataset A vs Dataset B by Refinement Scenario', fontsize=20, fontweight='bold', y=0.98)

# Posicionando a legenda e aumentando a fonte dela
sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.5), title_fontsize=14, fontsize=12)

# Salvando a nova versão organizada com alta resolução (dpi=300)
plt.savefig('comparativo_organizado_final.png', bbox_inches='tight', dpi=300)