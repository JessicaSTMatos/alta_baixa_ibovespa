# 📈 Previsão de Alta/Baixa do Ibovespa com Streamlit

Este projeto consiste no desenvolvimento de uma aplicação web em **Streamlit** para monitoramento do Ibovespa e **previsão da direção do índice (Alta ou Baixa)** por meio de um **modelo de Machine Learning de classificação**.  

A aplicação disponibiliza um dashboard interativo que apresenta métricas de desempenho do modelo, matriz de confusão e visualizações temporais, permitindo avaliar o comportamento do índice e a qualidade das previsões realizadas.

🔗 **Aplicação em produção:**  
https://previsao-ibovespa.streamlit.app/

---

## 🎯 Objetivo do Projeto

O objetivo principal do projeto é aplicar conceitos de **ciência de dados e aprendizado de máquina** em um problema real do mercado financeiro, contemplando todo o ciclo de desenvolvimento, desde a modelagem até o deploy de uma aplicação funcional.

Entre os objetivos específicos, destacam-se:
- Prever movimentos de **alta ou baixa** do Ibovespa
- Avaliar o desempenho de um modelo de classificação por meio de métricas estatísticas
- Criar um dashboard interativo para análise dos resultados
- Realizar o deploy da aplicação em ambiente cloud

O projeto foi desenvolvido **em grupo**, com fins acadêmicos e exploratórios.

---

## 🧠 Modelagem e Avaliação

A modelagem foi realizada a partir de dados históricos do Ibovespa, envolvendo etapas de preparação dos dados, definição da variável alvo e treinamento de um modelo de classificação.

A avaliação do modelo foi conduzida por meio de métricas como:
- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão

Durante o processo de avaliação, foi identificado **overfitting**, caracterizado por desempenho superior nos dados de treino em relação aos dados de teste. Esse comportamento é reconhecido como uma limitação do modelo e indica oportunidades de melhoria, como o uso de técnicas de regularização, validação temporal mais robusta e testes com outros algoritmos.

---

## 🚀 Deploy da Aplicação

O deploy da aplicação foi realizado utilizando **GitHub** para versionamento do código e **Streamlit Community Cloud** para publicação da aplicação.

O fluxo adotado incluiu:
- Criação de um repositório no GitHub
- Envio dos arquivos do projeto utilizando **Git Bash**
- Integração do repositório com o Streamlit Community Cloud
- Configuração do arquivo principal da aplicação
- Publicação da aplicação em ambiente cloud

Após esse processo, a aplicação passou a estar disponível publicamente no seguinte endereço:

🔗 https://previsao-ibovespa.streamlit.app/

---

## 🔮 Considerações Finais

O projeto permitiu consolidar conhecimentos práticos em modelagem preditiva, avaliação de modelos de classificação e deploy de aplicações de ciência de dados.  

Apesar das limitações identificadas, como o overfitting, a solução desenvolvida cumpre o objetivo proposto e estabelece uma base sólida para evoluções futuras, tanto em termos de modelagem quanto de análise e monitoramento do mercado financeiro.

---

## 👥 Autoria

Projeto desenvolvido em grupo para fins acadêmicos.
