import pandas as pd

def get_odors_dict(csv_file):
    # Carrega a tabela
    df = pd.read_csv(csv_file)
    mapping = {}
    for index, row in df.iterrows():
        smiles = row['standardized_smiles']
        # Extrai as colunas de odores, remove NAs e limpa espaços vazios
        odors = row.drop('standardized_smiles').dropna().astype(str).str.strip().tolist()
        odors = [o for o in odors if o]
        # Salva no formato "Odor1, Odor2, Odor3" ou "None" se estiver vazio
        mapping[smiles] = ", ".join(odors) if odors else "None"
    return mapping

# 1. Carrega os dicionários mapeando SMILES -> Odores para ambas as bases
prim_map = get_odors_dict('curated_PrimaryOdor.csv')
sub_map = get_odors_dict('curated_SubOdor.csv')

# 2. Pega todas as moléculas (SMILES) únicas juntando as duas bases
all_smiles = set(prim_map.keys()).union(set(sub_map.keys()))

prompts = []
completions = []
chemdfm_formats = []
smiles_list = []

# 3. Cria as frases combinando odores e sub-odores na mesma conversa
for smiles in all_smiles:
    prim_odors = prim_map.get(smiles, "None")
    sub_odors = sub_map.get(smiles, "None")
    
    # Ignora se por algum motivo ambos estiverem vazios para a molécula
    if prim_odors == "None" and sub_odors == "None":
        continue
        
    # Prompt focado em QSOR (Estrutura -> Odor)
    prompt = f"What are the primary odors and sub-odors of this molecule: {smiles}?"
    completion = f"The primary odors present in this molecule are: {prim_odors}. The sub-odors are: {sub_odors}."
    
    # Formatação exata exigida pelo ChemDFM
    chemdfm_format = f"[Round 0]\nHuman: {prompt}\nAssistant: {completion}"
    
    smiles_list.append(smiles)
    prompts.append(prompt)
    completions.append(completion)
    chemdfm_formats.append(chemdfm_format)
    
# 4. Salva tudo em uma nova tabela final
out_df = pd.DataFrame({
    'smiles': smiles_list,
    'prompt': prompts,
    'completion': completions,
    'chemdfm_text': chemdfm_formats
})

output_file = 'combined_odor_prompts.csv'
out_df.to_csv(output_file, index=False)

print(f"Dataset combinado criado com sucesso com {len(out_df)} exemplos.")