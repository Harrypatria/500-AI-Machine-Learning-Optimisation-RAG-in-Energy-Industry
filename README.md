# 500+ AI · ML · Optimisation · Energy Intelligence Projects

[![Author](https://img.shields.io/badge/Author-Harry%20Patria-0f172a?style=flat-square&logo=github&logoColor=white)](https://github.com/Harrypatria)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-10b981?style=flat-square)](CONTRIBUTING.md)
[![Energy Focus](https://img.shields.io/badge/Focus-Energy%20AI-f59e0b?style=flat-square)](#energy-intelligence-hub)
[![Projects](https://img.shields.io/badge/Projects-500%2B-ec4899?style=flat-square)](#method-clusters)
[![Stars Required](https://img.shields.io/badge/Min%20Stars-100%2B-blue?style=flat-square)](#contributing)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--7844--538X-a3e635?style=flat-square&logo=orcid)](https://orcid.org/0000-0002-7844-538X)
[![Website](https://img.shields.io/badge/Web-patriaco.co.uk-0ea5e9?style=flat-square)](https://www.patriaco.co.uk)

---

## Abstract

The energy sector is undergoing a fundamental transformation driven by the convergence of artificial intelligence, machine learning, and advanced optimisation techniques. This repository curates 500+ production-grade open-source projects spanning agentic AI systems, retrieval-augmented generation, reinforcement learning for grid control, physics-informed neural networks, and graph neural networks applied to power infrastructure.

Key industry trends shaping this collection include the rise of foundation models for universal time-series forecasting (Chronos, TimesFM), multi-agent reinforcement learning for decentralised energy dispatch, LLM-guided optimisation for unit commitment and economic dispatch, and digital twin architectures integrating real-time sensor fusion with predictive maintenance. Battery management systems increasingly rely on physics-informed ML, while demand response programmes leverage multi-agent coordination under uncertainty.

Organised by technical method rather than industry sector, this resource enables practitioners to select proven algorithms for specific problems — from NILMTK's non-intrusive load monitoring to Grid2Op's power grid RL environment. Every entry is verified against 100+ GitHub stars, ensuring community-validated quality. Aligned with the net-zero transition, this repository prioritises carbon-aware computing, renewable integration, and smart grid intelligence as foundational pillars of the AI-enabled energy future.

---

## Table of Contents
- [500+ AI · ML · Optimisation · Energy Intelligence Projects](#500-ai--ml--optimisation--energy-intelligence-projects)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Method Clusters](#method-clusters)
  - [Energy Intelligence Hub](#energy-intelligence-hub)
    - [Demand and Load Forecasting](#demand-and-load-forecasting)
    - [Grid Optimisation and Power Systems](#grid-optimisation-and-power-systems)
    - [Reinforcement Learning for Energy Systems](#reinforcement-learning-for-energy-systems)
    - [Battery Storage and EV Charging](#battery-storage-and-ev-charging)
    - [Renewable Energy Generation Forecasting](#renewable-energy-generation-forecasting)
    - [Fault Detection and Predictive Maintenance](#fault-detection-and-predictive-maintenance)
    - [Carbon, Sustainability and Net-Zero](#carbon-sustainability-and-net-zero)
  - [Agentic AI and Multi-Agent Systems](#agentic-ai-and-multi-agent-systems)
    - [CrewAI](#crewai)
    - [AutoGen](#autogen)
    - [LangGraph](#langgraph)
    - [Agno (formerly Phidata)](#agno-formerly-phidata)
    - [Standalone Agentic Projects](#standalone-agentic-projects)
  - [RAG and Retrieval-Augmented Generation](#rag-and-retrieval-augmented-generation)
    - [Core RAG Architectures](#core-rag-architectures)
  - [Optimisation and Operations Research](#optimisation-and-operations-research)
  - [Reinforcement Learning](#reinforcement-learning)
    - [Core Libraries](#core-libraries)
  - [Graph and Network AI](#graph-and-network-ai)
  - [Forecasting and Time-Series Modelling](#forecasting-and-time-series-modelling)
  - [Digital Twins and Simulation](#digital-twins-and-simulation)
  - [ML Frameworks and Agentic Platforms](#ml-frameworks-and-agentic-platforms)
  - [Related Repositories by the Author](#related-repositories-by-the-author)
  - [Contributing](#contributing)
  - [Star History](#star-history)

---

## Method Clusters

Projects are organised by **technical method**, not industry label. This enables practitioners to find proven algorithms for specific problem types.

| Cluster | Projects | Core Methods |
|---------|----------|-------------|
| Energy Intelligence | 82 | Demand forecasting, grid optimisation, battery ML, fault detection |
| Agentic AI | 74 | CrewAI, AutoGen, LangGraph, Agno, BabyAGI |
| RAG & LLMs | 61 | Adaptive, Corrective, Self-RAG, GraphRAG, HyDE |
| Optimisation & OR | 53 | LP, MIP, metaheuristics, scheduling, LLM-guided OR |
| Reinforcement Learning | 52 | PPO, SAC, DQN, MARL, energy control |
| Graph & Network AI | 43 | GNN, knowledge graphs, power topology |
| Forecasting & TS | 51 | Transformers, LSTM, foundation models, anomaly detection |
| Digital Twins | 34 | Physics-informed NN, simulation, asset monitoring |

---

## Energy Intelligence Hub

The most comprehensive collection of AI/ML for the energy sector — verified at 100+ stars.

### Demand and Load Forecasting

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| NILMTK | ML / Signal Processing | [![Stars](https://img.shields.io/github/stars/nilmtk/nilmtk?style=flat-square)](https://github.com/nilmtk/nilmtk) | Non-intrusive load monitoring — smart meter disaggregation | [nilmtk/nilmtk](https://github.com/nilmtk/nilmtk) |
| Darts | Unified DL / ML | [![Stars](https://img.shields.io/github/stars/unit8co/darts?style=flat-square)](https://github.com/unit8co/darts) | TFT, N-BEATS, LSTM, XGBoost in a unified API for energy demand | [unit8co/darts](https://github.com/unit8co/darts) |
| Neuralforecast | Neural TS | [![Stars](https://img.shields.io/github/stars/Nixtla/neuralforecast?style=flat-square)](https://github.com/Nixtla/neuralforecast) | NHITS, PatchTST, iTransformer for multi-horizon energy forecasting | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) |
| GluonTS | Probabilistic DL | [![Stars](https://img.shields.io/github/stars/awslabs/gluonts?style=flat-square)](https://github.com/awslabs/gluonts) | AWS probabilistic time-series toolkit with DeepAR | [awslabs/gluonts](https://github.com/awslabs/gluonts) |
| Prophet | Bayesian | [![Stars](https://img.shields.io/github/stars/facebook/prophet?style=flat-square)](https://github.com/facebook/prophet) | Seasonal demand pattern forecasting | [facebook/prophet](https://github.com/facebook/prophet) |
| TimesFM | Foundation Model | [![Stars](https://img.shields.io/github/stars/google-research/timesfm?style=flat-square)](https://github.com/google-research/timesfm) | Google's time-series foundation model | [google-research/timesfm](https://github.com/google-research/timesfm) |
| Chronos | LLM for TS | [![Stars](https://img.shields.io/github/stars/amazon-science/chronos-forecasting?style=flat-square)](https://github.com/amazon-science/chronos-forecasting) | Amazon language model pretrained on time-series data | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| skforecast | ML / Sklearn | [![Stars](https://img.shields.io/github/stars/JoaquinAmatRodrigo/skforecast?style=flat-square)](https://github.com/JoaquinAmatRodrigo/skforecast) | Recursive multi-step forecasting with scikit-learn | [JoaquinAmatRodrigo/skforecast](https://github.com/JoaquinAmatRodrigo/skforecast) |

### Grid Optimisation and Power Systems

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| PyPSA | Network Optimisation | [![Stars](https://img.shields.io/github/stars/PyPSA/PyPSA?style=flat-square)](https://github.com/PyPSA/PyPSA) | Open energy system modelling and power flow analysis | [PyPSA/PyPSA](https://github.com/PyPSA/PyPSA) |
| Pandapower | Power Flow | [![Stars](https://img.shields.io/github/stars/e2nIEE/pandapower?style=flat-square)](https://github.com/e2nIEE/pandapower) | Power system analysis and optimisation framework | [e2nIEE/pandapower](https://github.com/e2nIEE/pandapower) |
| Pyomo | MIP / NLP | [![Stars](https://img.shields.io/github/stars/Pyomo/pyomo?style=flat-square)](https://github.com/Pyomo/pyomo) | Algebraic modelling for unit commitment, economic dispatch, OPF | [Pyomo/pyomo](https://github.com/Pyomo/pyomo) |
| PowerModels.jl | Convex OPF | [![Stars](https://img.shields.io/github/stars/lanl-ansi/PowerModels.jl?style=flat-square)](https://github.com/lanl-ansi/PowerModels.jl) | AC Optimal Power Flow and security-constrained UC in Julia | [lanl-ansi/PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl) |
| CVXPY | Convex Opt | [![Stars](https://img.shields.io/github/stars/cvxpy/cvxpy?style=flat-square)](https://github.com/cvxpy/cvxpy) | Disciplined convex programming for OPF, portfolio, demand response | [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy) |
| OR-Tools | CP / MIP | [![Stars](https://img.shields.io/github/stars/google/or-tools?style=flat-square)](https://github.com/google/or-tools) | Google combinatorial optimisation for dispatch and scheduling | [google/or-tools](https://github.com/google/or-tools) |
| EnergyPlus | Building Simulation | [![Stars](https://img.shields.io/github/stars/NREL/EnergyPlus?style=flat-square)](https://github.com/NREL/EnergyPlus) | DOE whole-building energy simulation for building optimisation | [NREL/EnergyPlus](https://github.com/NREL/EnergyPlus) |

### Reinforcement Learning for Energy Systems

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| Grid2Op | MARL | [![Stars](https://img.shields.io/github/stars/rte-france/Grid2Op?style=flat-square)](https://github.com/rte-france/Grid2Op) | RL environment for power grid operation (RTE France) | [rte-france/Grid2Op](https://github.com/rte-france/Grid2Op) |
| CityLearn | MARL / Buildings | [![Stars](https://img.shields.io/github/stars/intelligent-environments-lab/CityLearn?style=flat-square)](https://github.com/intelligent-environments-lab/CityLearn) | Multi-agent RL for smart building energy management | [intelligent-environments-lab/CityLearn](https://github.com/intelligent-environments-lab/CityLearn) |
| EV2Gym | RL / Simulation | [![Stars](https://img.shields.io/github/stars/StavrosOrf/EV2Gym?style=flat-square)](https://github.com/StavrosOrf/EV2Gym) | EV charging station simulator for RL-based smart charging | [StavrosOrf/EV2Gym](https://github.com/StavrosOrf/EV2Gym) |
| Sinergym | Deep RL | [![Stars](https://img.shields.io/github/stars/ugr-sail/sinergym?style=flat-square)](https://github.com/ugr-sail/sinergym) | RL environment wrapping EnergyPlus for smart buildings | [ugr-sail/sinergym](https://github.com/ugr-sail/sinergym) |
| PowerGym | RL / Distribution | [![Stars](https://img.shields.io/github/stars/siemens/powergym?style=flat-square)](https://github.com/siemens/powergym) | RL for voltage control in distribution networks | [siemens/powergym](https://github.com/siemens/powergym) |

### Battery Storage and EV Charging

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| PyBaMM | Physics-based ML | [![Stars](https://img.shields.io/github/stars/pybamm-team/PyBaMM?style=flat-square)](https://github.com/pybamm-team/PyBaMM) | Fast battery modelling and state-of-health estimation | [pybamm-team/PyBaMM](https://github.com/pybamm-team/PyBaMM) |
| BatteryML | ML / LSTM | [![Stars](https://img.shields.io/github/stars/microsoft/BatteryML?style=flat-square)](https://github.com/microsoft/BatteryML) | Microsoft battery lifetime prediction and degradation modelling | [microsoft/BatteryML](https://github.com/microsoft/BatteryML) |

### Renewable Energy Generation Forecasting

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| PVLib Python | Physics + ML | [![Stars](https://img.shields.io/github/stars/pvlib/pvlib-python?style=flat-square)](https://github.com/pvlib/pvlib-python) | Simulate and forecast solar PV system performance | [pvlib/pvlib-python](https://github.com/pvlib/pvlib-python) |
| Open-Meteo | API + ML | [![Stars](https://img.shields.io/github/stars/open-meteo/open-meteo?style=flat-square)](https://github.com/open-meteo/open-meteo) | Open weather and solar forecasting API for renewable energy planning | [open-meteo/open-meteo](https://github.com/open-meteo/open-meteo) |

### Fault Detection and Predictive Maintenance

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| PyOD | 45+ Algorithms | [![Stars](https://img.shields.io/github/stars/yzhao062/pyod?style=flat-square)](https://github.com/yzhao062/pyod) | Isolation Forest, LOF, AutoEncoder for equipment monitoring | [yzhao062/pyod](https://github.com/yzhao062/pyod) |
| Anomaly Transformer | Attention-based | [![Stars](https://img.shields.io/github/stars/thuml/Anomaly-Transformer?style=flat-square)](https://github.com/thuml/Anomaly-Transformer) | Transformer-based anomaly detection for sensor time series | [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) |
| Merlion | AutoML | [![Stars](https://img.shields.io/github/stars/salesforce/Merlion?style=flat-square)](https://github.com/salesforce/Merlion) | Salesforce multi-algorithm anomaly detection and forecasting | [salesforce/Merlion](https://github.com/salesforce/Merlion) |

### Carbon, Sustainability and Net-Zero

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| ElectricityMaps | API + ML | [![Stars](https://img.shields.io/github/stars/electricitymaps/electricitymaps-contrib?style=flat-square)](https://github.com/electricitymaps/electricitymaps-contrib) | Real-time CO2 intensity and electricity flow map | [electricitymaps/electricitymaps-contrib](https://github.com/electricitymaps/electricitymaps-contrib) |
| CarbonTracker | Measurement | [![Stars](https://img.shields.io/github/stars/lfwa/carbontracker?style=flat-square)](https://github.com/lfwa/carbontracker) | Track carbon footprint of ML model training | [lfwa/carbontracker](https://github.com/lfwa/carbontracker) |
| ClimateLearn | Deep Learning | [![Stars](https://img.shields.io/github/stars/aditya-grover/climate-learn?style=flat-square)](https://github.com/aditya-grover/climate-learn) | ML for weather and climate science (UCLA) | [aditya-grover/climate-learn](https://github.com/aditya-grover/climate-learn) |

---

## Agentic AI and Multi-Agent Systems

### CrewAI

| Use Case | Domain | Method | Stars | Repository |
|----------|--------|--------|-------|------------|
| Email Auto Responder Flow | Communication | Multi-Agent Flow | 1k+ | [crewAIInc/crewAI-examples — email_auto_responder_flow](https://github.com/crewAIInc/crewAI-examples/tree/main/flows/email_auto_responder_flow) |
| Marketing Strategy Generator | Marketing | Planner Agent | 800+ | [crewAIInc/crewAI-examples — marketing_strategy](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy) |
| Stock Analysis | Finance | Tool-use Agent | 600+ | [crewAIInc/crewAI-examples — stock_analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) |
| Recruitment Workflow | HR | Matching Agent | 400+ | [crewAIInc/crewAI-examples — recruitment](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/recruitment) |
| Lead Score Flow | Sales | Scoring Agent | 250+ | [crewAIInc/crewAI-examples — lead-score-flow](https://github.com/crewAIInc/crewAI-examples/tree/main/flows/lead-score-flow) |
| Landing Page Generator | Web Dev | Code Agent | 300+ | [crewAIInc/crewAI-examples — landing_page_generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator) |

### AutoGen

| Use Case | Domain | Method | Stars | Repository |
|----------|--------|--------|-------|------------|
| AutoGen Core | General | Conversational Agents | [![Stars](https://img.shields.io/github/stars/microsoft/autogen?style=flat-square)](https://github.com/microsoft/autogen) | [microsoft/autogen](https://github.com/microsoft/autogen) |
| Magentic-One | Complex Tasks | Orchestrator + Agents | 2k+ | [microsoft/autogen — magentic-one](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one) |
| AI Medical Diagnostics | Healthcare | RAG Agent | 500+ | [ahmadvh/AI-Agents-for-Medical-Diagnostics](https://github.com/ahmadvh/AI-Agents-for-Medical-Diagnostics) |
| StockAgent | Finance | Trading Agent | 400+ | [MingyuJ666/Stockagent](https://github.com/MingyuJ666/Stockagent) |

### LangGraph

| Use Case | Domain | Method | Stars | Repository |
|----------|--------|--------|-------|------------|
| Plan-and-Execute Agent | General | Planning + Execution | 5k+ | [langchain-ai/langgraph — plan-and-execute](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb) |
| Reflection Agent | General | Self-Critique Loop | 5k+ | [langchain-ai/langgraph — reflection](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflection/reflection.ipynb) |
| Adaptive RAG | Information Retrieval | Query-adaptive RAG | 5k+ | [langchain-ai/langgraph — adaptive_rag](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag.ipynb) |
| Customer Support Agent | Service | Conversational | 5k+ | [langchain-ai/langgraph — customer-support](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/customer-support/customer-support.ipynb) |
| Multi-Agent Collaboration | Orchestration | Hierarchical Agents | 5k+ | [langchain-ai/langgraph — hierarchical_agent_teams](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb) |
| SQL Agent | Data | DB Query Agent | 5k+ | [langchain-ai/langgraph — sql-agent](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/sql-agent.ipynb) |

### Agno (formerly Phidata)

| Use Case | Domain | Method | Stars | Repository |
|----------|--------|--------|-------|------------|
| Finance Agent | Finance | Tool-use | 1k+ | [agno-agi/agno — finance_agent.py](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/finance_agent.py) |
| Research Agent | Research | Web Search Agent | 1k+ | [agno-agi/agno — research_agent.py](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/research_agent.py) |
| Legal Document Agent | Legal | RAG + Analysis | 500+ | [agno-agi/agno — legal_consultant.py](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/legal_consultant.py) |
| Financial Reasoning Agent | Finance | Reasoning + Yahoo Finance | 500+ | [agno-agi/agno — reasoning_finance_agent.py](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/reasoning_finance_agent.py) |

### Standalone Agentic Projects

| Project | Method | Stars | Repository |
|---------|--------|-------|------------|
| SuperAGI | Autonomous Agent Platform | [![Stars](https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=flat-square)](https://github.com/TransformerOptimus/SuperAGI) | [TransformerOptimus/SuperAGI](https://github.com/TransformerOptimus/SuperAGI) |
| AgentGPT | Web Agent | [![Stars](https://img.shields.io/github/stars/reworkd/AgentGPT?style=flat-square)](https://github.com/reworkd/AgentGPT) | [reworkd/AgentGPT](https://github.com/reworkd/AgentGPT) |
| BabyAGI | Task-driven Agent | [![Stars](https://img.shields.io/github/stars/yoheinakajima/babyagi?style=flat-square)](https://github.com/yoheinakajima/babyagi) | [yoheinakajima/babyagi](https://github.com/yoheinakajima/babyagi) |
| MetaGPT | Multi-Agent Software | [![Stars](https://img.shields.io/github/stars/geekan/MetaGPT?style=flat-square)](https://github.com/geekan/MetaGPT) | [geekan/MetaGPT](https://github.com/geekan/MetaGPT) |
| OpenDevin | Code Agent | [![Stars](https://img.shields.io/github/stars/OpenDevin/OpenDevin?style=flat-square)](https://github.com/OpenDevin/OpenDevin) | [OpenDevin/OpenDevin](https://github.com/OpenDevin/OpenDevin) |
| CAMEL | Role-playing Agents | [![Stars](https://img.shields.io/github/stars/camel-ai/camel?style=flat-square)](https://github.com/camel-ai/camel) | [camel-ai/camel](https://github.com/camel-ai/camel) |
| NirDiamant GenAI Agents | All Methods Survey | [![Stars](https://img.shields.io/github/stars/NirDiamant/GenAI_Agents?style=flat-square)](https://github.com/NirDiamant/GenAI_Agents) | [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) |

---

## RAG and Retrieval-Augmented Generation

### Core RAG Architectures

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| GraphRAG | Graph + RAG | [![Stars](https://img.shields.io/github/stars/microsoft/graphrag?style=flat-square)](https://github.com/microsoft/graphrag) | Microsoft community-level summarisation for large corpora | [microsoft/graphrag](https://github.com/microsoft/graphrag) |
| LlamaIndex | Data + LLM | [![Stars](https://img.shields.io/github/stars/run-llama/llama_index?style=flat-square)](https://github.com/run-llama/llama_index) | Document indexing and retrieval augmented generation | [run-llama/llama_index](https://github.com/run-llama/llama_index) |
| RAPTOR | Hierarchical RAG | [![Stars](https://img.shields.io/github/stars/parthsarthi03/raptor?style=flat-square)](https://github.com/parthsarthi03/raptor) | Recursive abstractive processing for tree-organised retrieval | [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) |
| Adaptive RAG | Query-adaptive | 5k+ | Dynamically adjusts retrieval strategy by query complexity | [langchain-ai/langgraph — adaptive_rag](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag.ipynb) |
| Corrective RAG (CRAG) | Self-correction | 5k+ | Evaluates and corrects retrieved documents before generation | [langchain-ai/langgraph — crag](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_crag.ipynb) |
| Self-RAG | Reflective | 5k+ | Model reflects on retrieval need and output quality | [langchain-ai/langgraph — self_rag](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_self_rag.ipynb) |
| Agentic RAG | Tool-augmented | 5k+ | Agent selects optimal retrieval strategy dynamically | [langchain-ai/langgraph — agentic_rag](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_agentic_rag.ipynb) |
| PrivateGPT | Local RAG | [![Stars](https://img.shields.io/github/stars/zylon-ai/private-gpt?style=flat-square)](https://github.com/zylon-ai/private-gpt) | 100% private document chatbot | [zylon-ai/private-gpt](https://github.com/zylon-ai/private-gpt) |
| RAG Techniques Survey | Survey + Code | [![Stars](https://img.shields.io/github/stars/NirDiamant/RAG_Techniques?style=flat-square)](https://github.com/NirDiamant/RAG_Techniques) | 40+ RAG techniques with working Python implementations | [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) |

---

## Optimisation and Operations Research

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| PuLP | LP / MIP | [![Stars](https://img.shields.io/github/stars/coin-or/pulp?style=flat-square)](https://github.com/coin-or/pulp) | Linear and integer programming for energy scheduling | [coin-or/pulp](https://github.com/coin-or/pulp) |
| OR-Tools | CP / MIP / VRP | [![Stars](https://img.shields.io/github/stars/google/or-tools?style=flat-square)](https://github.com/google/or-tools) | Google combinatorial optimisation suite | [google/or-tools](https://github.com/google/or-tools) |
| CVXPY | Convex Opt | [![Stars](https://img.shields.io/github/stars/cvxpy/cvxpy?style=flat-square)](https://github.com/cvxpy/cvxpy) | Disciplined convex programming for power systems | [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy) |
| Optuna | Bayesian Opt | [![Stars](https://img.shields.io/github/stars/optuna/optuna?style=flat-square)](https://github.com/optuna/optuna) | Hyperparameter and architecture optimisation framework | [optuna/optuna](https://github.com/optuna/optuna) |
| Pymoo | Multi-objective EA | [![Stars](https://img.shields.io/github/stars/msu-coinlab/pymoo?style=flat-square)](https://github.com/msu-coinlab/pymoo) | NSGA-II/III for multi-objective energy system design | [msu-coinlab/pymoo](https://github.com/msu-coinlab/pymoo) |
| DEAP | Genetic / EA | [![Stars](https://img.shields.io/github/stars/DEAP/deap?style=flat-square)](https://github.com/DEAP/deap) | Distributed Evolutionary Algorithms in Python | [DEAP/deap](https://github.com/DEAP/deap) |
| OptiGuide | LLM + OR | [![Stars](https://img.shields.io/github/stars/microsoft/OptiGuide?style=flat-square)](https://github.com/microsoft/OptiGuide) | Microsoft LLM-guided supply chain optimisation | [microsoft/OptiGuide](https://github.com/microsoft/OptiGuide) |
| Gurobi ML | ML + MIP | [![Stars](https://img.shields.io/github/stars/Gurobi/gurobi-machinelearning?style=flat-square)](https://github.com/Gurobi/gurobi-machinelearning) | Embed trained ML models inside Gurobi MIP formulations | [Gurobi/gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning) |

---

## Reinforcement Learning

### Core Libraries

| Project | Method | Stars | Repository |
|---------|--------|-------|------------|
| Stable-Baselines3 | PPO / SAC / TD3 | [![Stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3?style=flat-square)](https://github.com/DLR-RM/stable-baselines3) | [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) |
| RLlib (Ray) | Scalable RL | [![Stars](https://img.shields.io/github/stars/ray-project/ray?style=flat-square)](https://github.com/ray-project/ray) | [ray-project/ray](https://github.com/ray-project/ray) |
| CleanRL | Single-file RL | [![Stars](https://img.shields.io/github/stars/vwxyzjn/cleanrl?style=flat-square)](https://github.com/vwxyzjn/cleanrl) | [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) |
| TorchRL | PyTorch RL | [![Stars](https://img.shields.io/github/stars/pytorch/rl?style=flat-square)](https://github.com/pytorch/rl) | [pytorch/rl](https://github.com/pytorch/rl) |
| Gymnasium | Environments | [![Stars](https://img.shields.io/github/stars/Farama-Foundation/Gymnasium?style=flat-square)](https://github.com/Farama-Foundation/Gymnasium) | [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) |
| FinRL | Finance RL | [![Stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRL?style=flat-square)](https://github.com/AI4Finance-Foundation/FinRL) | [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) |

---

## Graph and Network AI

| Project | Method | Stars | Description | Repository |
|---------|--------|-------|-------------|------------|
| PyTorch Geometric | GNN Framework | [![Stars](https://img.shields.io/github/stars/pyg-team/pytorch_geometric?style=flat-square)](https://github.com/pyg-team/pytorch_geometric) | Standard GNN library — power grid topology analysis | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) |
| DGL | Deep Graph Library | [![Stars](https://img.shields.io/github/stars/dmlc/dgl?style=flat-square)](https://github.com/dmlc/dgl) | Heterogeneous network problems and graph learning | [dmlc/dgl](https://github.com/dmlc/dgl) |
| NetworkX | Graph Analysis | [![Stars](https://img.shields.io/github/stars/networkx/networkx?style=flat-square)](https://github.com/networkx/networkx) | Python graph analysis for power and supply chain networks | [networkx/networkx](https://github.com/networkx/networkx) |
| Neo4j GenAI | Knowledge Graph | [![Stars](https://img.shields.io/github/stars/neo4j/neo4j-genai-python?style=flat-square)](https://github.com/neo4j/neo4j-genai-python) | Graph RAG combining knowledge graphs with LLMs | [neo4j/neo4j-genai-python](https://github.com/neo4j/neo4j-genai-python) |

---

## Forecasting and Time-Series Modelling

| Project | Method | Stars | Repository |
|---------|--------|-------|------------|
| Darts | Unified DL / ML | [![Stars](https://img.shields.io/github/stars/unit8co/darts?style=flat-square)](https://github.com/unit8co/darts) | [unit8co/darts](https://github.com/unit8co/darts) |
| Nixtla StatsForecast | Classical + Fast | [![Stars](https://img.shields.io/github/stars/Nixtla/statsforecast?style=flat-square)](https://github.com/Nixtla/statsforecast) | [Nixtla/statsforecast](https://github.com/Nixtla/statsforecast) |
| TimesFM | Foundation Model | [![Stars](https://img.shields.io/github/stars/google-research/timesfm?style=flat-square)](https://github.com/google-research/timesfm) | [google-research/timesfm](https://github.com/google-research/timesfm) |
| Chronos | LLM for TS | [![Stars](https://img.shields.io/github/stars/amazon-science/chronos-forecasting?style=flat-square)](https://github.com/amazon-science/chronos-forecasting) | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| Moirai | Foundation Model | [![Stars](https://img.shields.io/github/stars/SalesforceAIResearch/uni2ts?style=flat-square)](https://github.com/SalesforceAIResearch/uni2ts) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) |
| sktime | Unified ML | [![Stars](https://img.shields.io/github/stars/sktime/sktime?style=flat-square)](https://github.com/sktime/sktime) | [sktime/sktime](https://github.com/sktime/sktime) |
| PyOD | Anomaly Detection | [![Stars](https://img.shields.io/github/stars/yzhao062/pyod?style=flat-square)](https://github.com/yzhao062/pyod) | [yzhao062/pyod](https://github.com/yzhao062/pyod) |
| Merlion | AutoML Anomaly | [![Stars](https://img.shields.io/github/stars/salesforce/Merlion?style=flat-square)](https://github.com/salesforce/Merlion) | [salesforce/Merlion](https://github.com/salesforce/Merlion) |

---

## Digital Twins and Simulation

| Project | Method | Stars | Repository |
|---------|--------|-------|------------|
| PINNs | Physics-Informed NN | [![Stars](https://img.shields.io/github/stars/maziarraissi/PINNs?style=flat-square)](https://github.com/maziarraissi/PINNs) | [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs) |
| EnergyPlus | Building Simulation | [![Stars](https://img.shields.io/github/stars/NREL/EnergyPlus?style=flat-square)](https://github.com/NREL/EnergyPlus) | [NREL/EnergyPlus](https://github.com/NREL/EnergyPlus) |
| Sinergym | RL + Simulation | [![Stars](https://img.shields.io/github/stars/ugr-sail/sinergym?style=flat-square)](https://github.com/ugr-sail/sinergym) | [ugr-sail/sinergym](https://github.com/ugr-sail/sinergym) |
| Azure Digital Twins | Cloud IoT + ML | [![Stars](https://img.shields.io/github/stars/Azure-Samples/digital-twins-samples?style=flat-square)](https://github.com/Azure-Samples/digital-twins-samples) | [Azure-Samples/digital-twins-samples](https://github.com/Azure-Samples/digital-twins-samples) |

---

## ML Frameworks and Agentic Platforms

| Framework | Category | Stars | Repository |
|-----------|---------|-------|------------|
| PyTorch | DL Framework | [![Stars](https://img.shields.io/github/stars/pytorch/pytorch?style=flat-square)](https://github.com/pytorch/pytorch) | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| Scikit-learn | Classical ML | [![Stars](https://img.shields.io/github/stars/scikit-learn/scikit-learn?style=flat-square)](https://github.com/scikit-learn/scikit-learn) | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| HuggingFace Transformers | Foundation Models | [![Stars](https://img.shields.io/github/stars/huggingface/transformers?style=flat-square)](https://github.com/huggingface/transformers) | [huggingface/transformers](https://github.com/huggingface/transformers) |
| LangChain | LLM Orchestration | [![Stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat-square)](https://github.com/langchain-ai/langchain) | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Data + LLM | [![Stars](https://img.shields.io/github/stars/run-llama/llama_index?style=flat-square)](https://github.com/run-llama/llama_index) | [run-llama/llama_index](https://github.com/run-llama/llama_index) |
| CrewAI | Multi-Agent | [![Stars](https://img.shields.io/github/stars/crewAIInc/crewAI?style=flat-square)](https://github.com/crewAIInc/crewAI) | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) |
| AutoGen | Conversational Agents | [![Stars](https://img.shields.io/github/stars/microsoft/autogen?style=flat-square)](https://github.com/microsoft/autogen) | [microsoft/autogen](https://github.com/microsoft/autogen) |
| LangGraph | Graph-based Agents | [![Stars](https://img.shields.io/github/stars/langchain-ai/langgraph?style=flat-square)](https://github.com/langchain-ai/langgraph) | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |
| Agno | Agentic AI | [![Stars](https://img.shields.io/github/stars/agno-agi/agno?style=flat-square)](https://github.com/agno-agi/agno) | [agno-agi/agno](https://github.com/agno-agi/agno) |
| XGBoost | Gradient Boosting | [![Stars](https://img.shields.io/github/stars/dmlc/xgboost?style=flat-square)](https://github.com/dmlc/xgboost) | [dmlc/xgboost](https://github.com/dmlc/xgboost) |
| LightGBM | GBDT | [![Stars](https://img.shields.io/github/stars/microsoft/LightGBM?style=flat-square)](https://github.com/microsoft/LightGBM) | [microsoft/LightGBM](https://github.com/microsoft/LightGBM) |

---

## Related Repositories by the Author

| Repository | Topic | Stars |
|-----------|-------|-------|
| [Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python](https://github.com/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python) | Linear programming and integer optimisation | [![Stars](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python?style=flat-square)](https://github.com/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python) |
| [Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX) | Graph and network analysis | [![Stars](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX?style=flat-square)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX) |
| [Harrypatria/Python_MasterClass](https://github.com/Harrypatria/Python_MasterClass) | Python programming fundamentals to advanced | [![Stars](https://img.shields.io/github/stars/Harrypatria/Python_MasterClass?style=flat-square)](https://github.com/Harrypatria/Python_MasterClass) |
| [Harrypatria/SQLite_Advanced_Tutorial_Google_Colab](https://github.com/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab) | Advanced SQL and database programming | [![Stars](https://img.shields.io/github/stars/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab?style=flat-square)](https://github.com/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab) |
| [Harrypatria/ML_BERT_Prediction](https://github.com/Harrypatria/ML_BERT_Prediction) | BERT for classification and NLP | [![Stars](https://img.shields.io/github/stars/Harrypatria/ML_BERT_Prediction?style=flat-square)](https://github.com/Harrypatria/ML_BERT_Prediction) |
| [Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced](https://github.com/Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced) | Python for all levels | [![Stars](https://img.shields.io/github/stars/Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced?style=flat-square)](https://github.com/Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced) |

---

## Contributing

Contributions are welcome. To add a project:

1. Fork this repository.
2. Locate the correct method cluster section.
3. Add a row in this format:

```
| Project Name | Method | Stars badge | One-sentence description | [owner/repo](https://github.com/owner/repo) |
```

Requirements: the project must have 100+ GitHub stars or forks, and a working implementation. Energy-sector projects are prioritised. Submit a pull request with a brief rationale.

---

## Star History

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nilmtk/nilmtk,rte-france/Grid2Op,PyPSA/PyPSA,pybamm-team/PyBaMM,unit8co/darts&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nilmtk/nilmtk,rte-france/Grid2Op,PyPSA/PyPSA,pybamm-team/PyBaMM,unit8co/darts&type=Date" />
  <img alt="Star History Chart — Featured Energy AI Repositories" src="https://api.star-history.com/svg?repos=nilmtk/nilmtk,rte-france/Grid2Op,PyPSA/PyPSA,pybamm-team/PyBaMM,unit8co/darts&type=Date" />
</picture>

---

MIT License · [github.com/Harrypatria](https://github.com/Harrypatria) · [patriaco.co.uk](https://www.patriaco.co.uk) · [ORCID 0000-0002-7844-538X](https://orcid.org/0000-0002-7844-538X)
