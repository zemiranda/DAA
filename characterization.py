"""Caracterizacao de cada faixa etaria (idade REAL do treino) a nivel de saude, estilo de vida e socioeconomico.
Requer out/features2.pkl (python fe2.py). Gera out/caracterizacao_faixas.csv e imprime resumos.
Nota: as perguntas PC e HA (autonomia) so foram feitas a quem tem 55+ anos (encaminhamento do inquerito);
por isso a gravidade PC/HA so e reportada a partir dos 55-59.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
OUT = "out"
LAB = ["15-17","18-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
tr = pd.read_csv("dataset_train.csv"); n = len(tr)
f = pd.read_pickle(f"{OUT}/features2.pkl").iloc[:n].reset_index(drop=True)
y = tr.AGE_RANGE.values
pct = lambda m: 100 * pd.Series(m).groupby(y).mean()

D = pd.DataFrame(index=range(16)); D.index = LAB
D["n"] = np.bincount(y)
D["%mulheres"] = pct(tr.SEX == "Mulher").values
# --- saude
D["%saude_ma_muito_ma"] = pct(tr.HS1.isin(["Mau", "Muito mau"])).values
D["n_doencas_cronicas"] = pd.Series(f.N_DOENCAS.values).groupby(y).mean().values
D["%3+_doencas"] = pct(f.N_DOENCAS >= 3).values
D["%dif_sensorial_motora"] = pct(f.DIF_N_PL > 0).values            # PL: dificuldades reais (alguma/muita/nao consegue)
D["gravidade_PL"] = pd.Series(f.DIF_SEV_PL.values).groupby(y).mean().values
g = pd.Series(f.DIF_SEV_PCHA.values).where(y >= 9).groupby(y).mean()    # so 55+
D["gravidade_PC_HA(55+)"] = g.reindex(range(16)).values
imc = f.IMC.where((f.IMC > 12) & (f.IMC < 60))
D["IMC_medio"] = pd.Series(imc.values).groupby(y).mean().values
D["%obesidade(IMC>=30)"] = 100 * pd.Series((imc >= 30).where(imc.notna())).groupby(y).mean().values
# --- estilo de vida
D["%fuma_diariamente"] = pct(tr.IN43 == "Fuma diariamente").values
D["%alcool_diario"] = pct(tr.AL1 == "Todos os dias ou quase todos os dias").values
D["%rede_social_<=2"] = pct(tr.SS1.isin(["Nenhuma", "1 ou 2"])).values
D["%trabalho_fisico_pesado"] = pct(tr.PE1 == "Em trabalhos pesados ou fisicamente exigentes").values
# --- socioeconomico
basico = ["Nenhum nível completado", "Básico 1º e 2º ciclo", "Básico 3º ciclo"]
sup = ["Bacharelato ou licenciatura 1º ciclo", "Licenciatura pré-bolonha ou mestrado 2º ciclo", "Doutoramento", "C.T.Superior Profissional"]
D["%ate_ensino_basico"] = pct(tr.HATLEVEL.isin(basico)).values
D["%ensino_superior"] = pct(tr.HATLEVEL.isin(sup)).values
D["%1o_quintil_rend"] = pct(tr.HHINCOME == "1º quintil").values
D["%5o_quintil_rend"] = pct(tr.HHINCOME == "5º quintil").values
D["%empregado"] = pct(tr.MAINSTAT.str.startswith("Tem um emprego")).values
D["%reformado"] = pct(tr.MAINSTAT.str.startswith("Reformado")).values
D["%desempregado"] = pct(tr.MAINSTAT == "Desempregado").values
D["%estudante"] = pct(tr.MAINSTAT.str.startswith("Aluno")).values
D["%viuvo"] = pct(tr.MARSTALEGAL == "Viúvo").values
D["%casado"] = pct(tr.MARSTALEGAL == "Casado").values
D["%vive_so"] = pct(tr.HHTYPE == "Agregado unipessoal").values
D["%zona_densa"] = pct(tr.DEG_URB == "Áreas densamente povoadas").values
D = D.round(1); D.to_csv(f"{OUT}/caracterizacao_faixas.csv", index_label="faixa")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
blocos = {
 "SAUDE": ["n", "%mulheres", "%saude_ma_muito_ma", "n_doencas_cronicas", "%3+_doencas", "%dif_sensorial_motora", "gravidade_PL", "gravidade_PC_HA(55+)"],
 "CORPO E ESTILO DE VIDA": ["IMC_medio", "%obesidade(IMC>=30)", "%fuma_diariamente", "%alcool_diario", "%rede_social_<=2", "%trabalho_fisico_pesado"],
 "SOCIOECONOMICO": ["%ate_ensino_basico", "%ensino_superior", "%1o_quintil_rend", "%5o_quintil_rend", "%empregado", "%reformado", "%desempregado", "%estudante", "%viuvo", "%casado", "%vive_so", "%zona_densa"],
}
for k, cols in blocos.items():
    print(f"\n== {k} =="); print(D[cols].to_string())
