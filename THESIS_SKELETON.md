# Thesis Skeleton — Bootstrapping Text Anomaly Detection with LLM-Generated Weak Supervision

---

## Tese Central

> **"Um LLM local compacto, usado como anotador fraco sobre dados reais, é suficiente para bootstrappear um detector de anomalias textuais de baixo custo que generaliza através de tarefas e línguas — e sua eficácia é limitada pela expressabilidade semântica do critério de normalidade, não pelo volume de labels gerados."**

---

## Linha de Raciocínio (narrative arc)

```
[Paper 1] O problema existe e é estrutural
    → Supervisão melhora muito anomaly detection em texto
    → Mas o gargalo é a anotação: rara, cara, dependente de especialista
    → Encoder importa mais que o modelo — o que sugere onde intervir
    → Conclusão: precisamos de uma fonte de supervisão sem custo humano

          ↓

[Gap] O que nenhum benchmark fez:
    eliminar o anotador humano sem perder a capacidade de generalizar

          ↓

[Paper 2] A arquitetura proposta fecha esse gap
    → LLM local como oráculo fraco: anota uma vez, treina um detector leve
    → Funciona: 56–79% do gap para o oracle, zero anotação humana
    → Generaliza: 2 tarefas × 2 línguas × 2 modelos
    → Mas tem limites — e os limites são explicáveis

          ↓

[Teoria dos Failure Modes] Onde e por quê falha
    → Ruído quantitativo (muitos FP): o pipeline tolera
    → Ruído direcional (inversão de polaridade): o pipeline amplifica via SetFit
    → Calibração cross-lingual: modelo maior ≠ sempre melhor
    → Conclusão: a eficácia é função da expressabilidade do critério de normalidade

          ↓

[Tese] A limitação fundamental não é o modelo nem o volume de dados
       — é a capacidade do LLM de inferir a fronteira de normalidade
       a partir de uma descrição em linguagem natural
```

---

## Estrutura Proposta da Dissertação

### Capítulo 1 — Introdução
- Motivação: anomaly detection em texto é importante e difícil
- O bottleneck real: não é o modelo, é o rótulo
- Hipótese central e contribuições
- Roadmap da dissertação

### Capítulo 2 — Revisão da Literatura
- Anomaly detection em texto (CVDD, DATE, DeepSAD, MLP)
- Semi-supervised AD: quanto label é suficiente?
- LLMs como anotadores: o que já foi feito e onde falta
- Weak supervision e few-shot fine-tuning (SetFit)
- Multilingual NLP: encoders e generalização cross-lingual

### Capítulo 3 — Benchmark Multilíngue (Paper 1)
*"A supervisão importa — mas de onde ela vem?"*
- Setup experimental: 6 datasets, 2 línguas, múltiplos encoders e modelos
- RQ1: Supervisão melhora? → Sim, dramaticamente em hate speech
- RQ2: Encoder vs. modelo: o que domina? → Encoder
- RQ3: Qual modelo é mais robusto? → MLP e DevNet
- RQ4: Quanto label é necessário? → Satura rápido (< 1% em alguns casos)
- **Takeaway:** o gargalo é a anotação, não a arquitetura

### Capítulo 4 — Pipeline com Weak Supervision via LLM (Paper 2)
*"Podemos eliminar o anotador humano?"*
- Arquitetura proposta: LLM → weak labels → SetFit → MLP/DeepSAD
- Estratégias de sampling (random, diversity)
- Resultados principais: 56–79% do gap recuperado
- Ablation 2×2: MLP vs DeepSAD, original vs SetFit
- **Takeaway:** funciona, mas com limites identificáveis

### Capítulo 5 — Teoria dos Failure Modes
*"Quando e por quê o pipeline falha?"*
- Regime 1: Ruído quantitativo (20 Newsgroups) — tolerado pelo MLP
- Regime 2: Inversão de polaridade (WikiNews 7B) — amplificado pelo SetFit
- Regime 3: Calibração cross-lingual (HateBR) — modelo maior ≠ melhor
- Hipótese unificadora: expressabilidade semântica do critério de normalidade
- **Takeaway:** o limite não é o volume de labels — é a qualidade da fronteira descrita no prompt

### Capítulo 6 — Conclusão e Trabalho Futuro
- Síntese da contribuição
- Limitações honestas
- Direções futuras

---

## Materialidade Atual (o que já temos)

| Item | Status |
|---|---|
| Paper 1 publicado (STIL anterior) | ✅ Sólido |
| Paper 2 aceito (STIL 2026) | ✅ Sólido |
| Resultados em 4 datasets, 2 línguas, 2 tarefas | ✅ |
| Ablation 2×2 (modelo × embedding) | ✅ |
| Análise de annotation quality por dataset/modelo | ✅ |
| Identificação de 3 failure modes | ✅ |
| Código público e reproduzível | ✅ |
| Comparação com 7 baselines unsupervised + 4 oracles | ✅ |

**Conclusão:** há materialidade suficiente para uma quali sólida sem novos experimentos.

---

## Recomendações por Prioridade

### SHOULD (reforça a defesa, baixo custo)

**S1 — Adicionar baseline "MiniLM sem fine-tune"**
- Por que: o ablation atual confunde ganho do SetFit com ganho do backbone melhor
- Como: rodar `--no_setfit --setfit_model paraphrase-multilingual-MiniLM-L12-v2` (o encoder do SetFit sem treinar)
- Esforço: 1 dia de experimentos, mesmos datasets
- Impacto: fecha a pergunta mais óbvia de qualquer revisor/banca sobre o Capítulo 4

**S2 — Formalizar a hipótese de "expressabilidade semântica"**
- Por que: os failure modes são observação empírica — precisam virar teoria
- Como: propor uma métrica proxy (e.g., separabilidade do embedding original antes do SetFit — silhouette score das classes) e mostrar correlação com downstream AUC
- Esforço: análise sobre dados já existentes, sem novos experimentos
- Impacto: transforma o Capítulo 5 de "achamos isso" para "prevemos isso"

**S3 — Teste de significância estatística nos resultados do Paper 2**
- Por que: 6 runs por configuração tem sobreposição de intervalos de confiança em alguns casos
- Como: Wilcoxon signed-rank entre MLP e DeepSAD (já feito no Paper 1, replicar no 2)
- Esforço: 1 hora de análise
- Impacto: protege contra "isso é ruído experimental" na banca

---

### COULD (robustez adicional, se houver tempo)

**C1 — Testar um segundo LLM (ex: Mistral ou Llama)**
- Por que: hoje tudo é Qwen — banca pode questionar se os failure modes são do método ou do modelo
- Esforço: médio (rodar o pipeline com outro modelo GGUF)
- Impacto: generaliza os failure modes além da família Qwen

**C2 — Curva N × AUC por dataset**
- Por que: mostra empiricamente o ponto de colapso do método (abaixo de qual N ele falha)
- Esforço: rodar com N = 50, 100, 150, 200 — dados do pipeline já existem para N=200
- Impacto: conecta Paper 1 (quanto label é suficiente?) com Paper 2 (quanto LLM call é suficiente?)

**C3 — Análise qualitativa dos prompts por failure mode**
- Por que: mostra que o problema é na fronteira descrita no prompt, não no LLM em si
- Esforço: manual, sem código novo
- Impacto: suporta a hipótese de expressabilidade com evidência qualitativa

---

### WOULD (tese completa / doutorado)

**W1 — Soft-label distillation**
- Usar o score contínuo do LLM (0.0–1.0) ao invés de binarizar em τ=0.5
- Potencialmente resolve o failure mode de inversão de polaridade (WikiNews 7B)
- Exige mudança no SetFit e no MLP

**W2 — Sampling semanticamente guiado**
- Selecionar candidatos próximos à fronteira de normalidade (boundary sampling)
- Mais eficiente que random/diversity para N pequeno

**W3 — Few-shot no prompt do LLM**
- Incluir 2–3 exemplos no prompt (selecionados por diversity sampling)
- Potencialmente melhora hate speech implícito sem violar a premissa de zero human labels

**W4 — Expansão multilíngue**
- Espanhol, Francês, Árabe — teste real de generalização cross-lingual
- Crítico para validar a tese de generalização além de PT/EN

---

## Remarks sobre Experimentos Recentes

- **score_guided descartado:** estratégia removida do pipeline final (não publica resultado melhor que random/diversity, e requer DeepSVDD extra)
- **told_br descartado:** problema estrutural no ground truth + hate implícito PT-BR irresolvível com zero-shot
- **diversity vs random:** indistinguíveis nos experimentos com 3 seeds — resultado em si (robustez à estratégia de sampling)
- **Qwen 7B vs 14B em HateBR:** 7B melhor que 14B — calibração cultural, não capacidade de modelo
- **SetFit ativado em média com 7–16 anomalias por run:** limiar mínimo de 8 por classe é crítico

---

## Checklist para Quali

- [ ] Tese central articulada (✅ acima)
- [ ] Narrative arc conectando os dois papers (✅ acima)
- [ ] Capítulos mapeados com função clara
- [ ] Baseline "MiniLM sem fine-tune" rodada (S1)
- [ ] Teste de significância estatística (S3)
- [ ] Hipótese de expressabilidade formalizada (S2)
- [ ] Slides com a linha de raciocínio clara
- [ ] Resposta preparada para: "qual é a contribuição nova?"
- [ ] Resposta preparada para: "por que só 4 datasets?"
- [ ] Resposta preparada para: "o confound do SetFit"
