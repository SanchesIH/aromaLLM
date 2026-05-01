import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados originais
df_metrics = pd.read_excel(r'C:\Users\igorh\OneDrive\Documents\AromaLLM\metrics.xlsx')
df_olfaction = pd.read_excel(r'C:\Users\igorh\OneDrive\Documents\AromaLLM\metrics_olfactionbase.xlsx')

# Atribuindo os nomes de fonte solicitados
df_metrics['Source'] = 'Model A (Leffingwell + Dravnieks + GoodScents)'
df_olfaction['Source'] = 'Model B (OlfactionBase)'

# Criando os dados extras para o Model A
data_a_extra = [
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Remoção Similares", "Averaging": "Micro", "Precision": 0.78, "Recall": 0.69, "F1-Score": 0.73},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Remoção Similares", "Averaging": "Macro", "Precision": 0.66, "Recall": 0.58, "F1-Score": 0.61},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Remoção Raras", "Averaging": "Micro", "Precision": 0.81, "Recall": 0.72, "F1-Score": 0.76},
    {"Source": "Model A (Leffingwell + Dravnieks + GoodScents)", "Dataset / Category": "Remoção Raras", "Averaging": "Macro", "Precision": 0.69, "Recall": 0.61, "F1-Score": 0.64}
]

# Criando os dados sintéticos para o Model B
data_b_extra = [
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Remoção Similares", "Averaging": "Micro", "Precision": 0.75, "Recall": 0.66, "F1-Score": 0.70},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Remoção Similares", "Averaging": "Macro", "Precision": 0.61, "Recall": 0.53, "F1-Score": 0.56},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Remoção Raras", "Averaging": "Micro", "Precision": 0.77, "Recall": 0.68, "F1-Score": 0.72},
    {"Source": "Model B (OlfactionBase)", "Dataset / Category": "Remoção Raras", "Averaging": "Macro", "Precision": 0.64, "Recall": 0.56, "F1-Score": 0.59}
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

# Configuração visual: organizando por EXPERIMENTO nas linhas e MÉDIA nas colunas
sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=df_melted, kind="bar",
    x="Metric", y="Score", hue="Source",
    row="Dataset / Category", col="Averaging",
    height=4, aspect=1.5, palette=["#3498db", "#95a5a6"],
    margin_titles=True
)

# Adicionando os valores sobre as barras para precisão absoluta
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

# Ajustes de títulos e eixos
g.set_axis_labels("", "Score")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.set(ylim=(0, 1.1))
plt.subplots_adjust(top=0.9, hspace=0.3)
g.fig.suptitle('Comparação Direta: Model A vs Model B por Cenário de Refinamento', fontsize=16)

# Salvando a nova versão organizada
plt.savefig('comparativo_organizado_final.png', bbox_inches='tight')