# Plano de Validação Clínica e Submissão Regulatória (CardioAI Nexus)

## 1. Visão Geral do Sistema
O CardioAI Nexus é um sistema híbrido de inteligência artificial (Wavelet DSP + CNN para classificação beat-level + Temporal Transformer para rhythm-level) projetado para a interpretação automatizada de Eletrocardiogramas (ECG) de 12 derivações.

## 2. Objetivos do Estudo Prospectivo
- **Objetivo Primário:** Validar a sensibilidade (SE) e especificidade (SP) do algoritmo na detecção de arritmias críticas (ex: Fibrilação Ventricular, TV sustentada, BAVT) e isquemia aguda (IAMCSST, Padrão de de Winter, Wellens) em comparação com o padrão-ouro (consenso de 3 cardiologistas especialistas).
- **Objetivo Secundário:** Avaliar a taxa de falsos positivos em ambiente de mundo real e o impacto no fluxo de trabalho clínico (tempo porta-ECG-diagnóstico).

## 3. Desenho do Estudo
- **Tipo:** Estudo clínico prospectivo, multicêntrico, duplo-cego.
- **Centros:** 3 hospitais terciários com unidades de dor torácica e emergência cardiológica.
- **Tamanho da Amostra (N):** 5.000 ECGs consecutivos (visando poder estatístico de 90% para classes raras com prevalência < 1%).
- **Critérios de Inclusão:** Pacientes > 18 anos admitidos na emergência ou UTI com indicação clínica para ECG de 12 derivações.
- **Critérios de Exclusão:** ECGs com qualidade técnica inaceitável (ruído > 50% do traçado) onde a interpretação humana é impossível.

## 4. Metodologia de Adjudicação (Human-in-the-Loop)
1. O ECG é adquirido e processado pelo CardioAI Nexus (latência < 200ms).
2. O laudo da IA é ocultado dos adjudicadores iniciais.
3. Três cardiologistas (cegos entre si e cegos para a IA) laudam o ECG.
4. O "Padrão-Ouro" é definido por concordância de pelo menos 2/3 dos cardiologistas.
5. Em caso de discordância total, um eletrofisiologista sênior atua como tie-breaker.

## 5. Métricas de Avaliação
- **Sensibilidade (Recall):** TP / (TP + FN) - Meta: > 98% para ritmos críticos.
- **Especificidade:** TN / (TN + FP) - Meta: > 95% para reduzir fadiga de alarmes.
- **Valor Preditivo Positivo (PPV):** TP / (TP + FP) - Meta: > 90% para IAMCSST.
- **Valor Preditivo Negativo (NPV):** TN / (TN + FN) - Meta: > 99% para triagem de emergência.

## 6. Estratégia Regulatória (FDA / ANVISA / CE Mark)
- **Classificação de Risco:** Software as a Medical Device (SaMD) - Classe II (FDA) / Classe III (ANVISA).
- **Submissão FDA:** Via 510(k) demonstrando equivalência substancial a predicados existentes (ex: algoritmos GE Marquette 12SL ou Philips DXL), com a superioridade da IA como diferencial.
- **Normas Aplicáveis:**
  - ISO 13485: Sistema de Gestão de Qualidade para Dispositivos Médicos.
  - ISO 14971: Gestão de Risco (mitigação de falsos negativos em condições críticas).
  - IEC 62304: Processos de Ciclo de Vida de Software Médico.
  - LGPD / HIPAA: Anonimização de dados no pipeline de re-treinamento.

## 7. Governança de Dados e Re-treinamento Contínuo
- Os dados do Dashboard Clínico (onde o médico clica em "Approve", "Modify" ou "Reject") são anonimizados e enviados para um Data Lake seguro.
- **Pipeline de Re-treino:** Modelos são re-treinados trimestralmente usando *holdout datasets* rigorosos para evitar *catastrophic forgetting*.
- **Monitoramento Pós-Mercado:** Rastreamento contínuo da taxa de alarmes falsos e *data drift* em diferentes demografias.
