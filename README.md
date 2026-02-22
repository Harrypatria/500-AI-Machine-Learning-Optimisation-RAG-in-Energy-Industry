# 500+ AI · ML · Optimisation · Energy Intelligence Projects

[![Author](https://img.shields.io/badge/Author-Harry%20Patria-0f172a?style=flat-square&logo=github&logoColor=white)](https://github.com/Harrypatria)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-10b981?style=flat-square)](CONTRIBUTING.md)
[![Energy Focus](https://img.shields.io/badge/Focus-Energy%20AI-f59e0b?style=flat-square)](#energy-intelligence-hub)
[![Projects](https://img.shields.io/badge/Projects-500%2B-ec4899?style=flat-square)](#method-clusters)
[![Min Stars](https://img.shields.io/badge/Min%20Stars-100%2B-3b82f6?style=flat-square)](#contributing)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--7844--538X-a3e635?style=flat-square&logo=orcid)](https://orcid.org/0000-0002-7844-538X)
[![Website](https://img.shields.io/badge/Web-patriaco.co.uk-0ea5e9?style=flat-square)](https://www.patriaco.co.uk)

---

## Abstract

The energy sector is undergoing a fundamental transformation driven by the convergence of artificial intelligence, machine learning, and advanced optimisation techniques. This repository curates 500+ production-grade open-source projects spanning agentic AI systems, retrieval-augmented generation, reinforcement learning for grid control, physics-informed neural networks, and graph neural networks applied to power infrastructure.

Key industry trends shaping this collection include the rise of foundation models for universal time-series forecasting (Chronos, TimesFM), multi-agent reinforcement learning for decentralised energy dispatch, LLM-guided optimisation for unit commitment and economic dispatch, and digital twin architectures integrating real-time sensor fusion with predictive maintenance. Battery management systems increasingly rely on physics-informed ML, while demand response programmes leverage multi-agent coordination under uncertainty.

Organised by technical method rather than industry sector, this resource enables practitioners to select proven algorithms for specific problems — from NILMTK's non-intrusive load monitoring to Grid2Op's power grid RL environment. Every entry is verified against 100+ GitHub stars, ensuring community-validated quality. Aligned with the net-zero transition, this repository prioritises carbon-aware computing, renewable integration, and smart grid intelligence as foundational pillars of the AI-enabled energy future.

---

## Repository Architecture Diagram

```mermaid
graph TD
    ROOT["500+ AI · ML · Optimisation\nEnergy Intelligence Projects"]

    ROOT --> A["⚡ Energy Intelligence\n82 repos"]
    ROOT --> B["🤖 Agentic AI\n74 repos"]
    ROOT --> C["🔍 RAG & LLMs\n61 repos"]
    ROOT --> D["📈 Optimisation & OR\n53 repos"]
    ROOT --> E["🧬 Reinforcement Learning\n52 repos"]
    ROOT --> F["🕸 Graph & Network AI\n43 repos"]
    ROOT --> G["🔮 Forecasting & TS\n51 repos"]
    ROOT --> H["🏭 Digital Twins\n34 repos"]

    A --> A1["Demand Forecasting\nDarts · Prophet · TimesFM"]
    A --> A2["Grid Optimisation\nPyPSA · Pyomo · CVXPY"]
    A --> A3["RL for Grids\nGrid2Op · CityLearn · Sinergym"]
    A --> A4["Battery & EV\nPyBaMM · BatteryML · EV2Gym"]
    A --> A5["Fault Detection\nPyOD · Merlion · Anomaly-T"]
    A --> A6["Renewables\nPVLib · Open-Meteo"]
    A --> A7["Carbon & Net-Zero\nElectricityMaps · CarbonTracker"]

    B --> B1["CrewAI\nFlows · Crews · Tools"]
    B --> B2["AutoGen\nCore · Magentic-One"]
    B --> B3["LangGraph\nPlan-Execute · Reflection"]
    B --> B4["Agno\nFinance · Research · Legal"]
    B --> B5["Standalone\nBabyAGI · MetaGPT · SuperAGI"]

    C --> C1["Core RAG\nAdaptive · CRAG · Self-RAG"]
    C --> C2["Graph RAG\nGraphRAG · RAPTOR · Neo4j"]
    C --> C3["Chatbots\nPrivateGPT · Ollama · Rasa"]

    D --> D1["Mathematical\nPuLP · OR-Tools · Pyomo"]
    D --> D2["Metaheuristics\nDEAP · Pymoo · PySwarms"]
    D --> D3["LLM-guided OR\nOptiGuide · Gurobi-ML"]

    E --> E1["Libraries\nSB3 · RLlib · CleanRL"]
    E --> E2["Energy Envs\nGrid2Op · CityLearn · PowerGym"]
    E --> E3["Finance RL\nFinRL · FinGPT"]

    F --> F1["GNN Frameworks\nPyG · DGL"]
    F --> F2["Graph Analysis\nNetworkX · iGraph"]
    F --> F3["Knowledge Graphs\nGraphRAG · Neo4j GenAI"]

    G --> G1["Foundation Models\nTimesFM · Chronos · Moirai"]
    G --> G2["Libraries\nDarts · sktime · StatsForecast"]
    G --> G3["Anomaly Detection\nPyOD · Merlion · Anomaly-T"]

    H --> H1["Physics-Informed\nPINNs · PyBaMM"]
    H --> H2["Simulation\nEnergyPlus · Sinergym"]
    H --> H3["Cloud Twins\nAzure Digital Twins"]

    style ROOT fill:#1e1b4b,color:#fff,stroke:#6366f1,stroke-width:2px
    style A fill:#78350f,color:#fef3c7,stroke:#f59e0b
    style B fill:#312e81,color:#e0e7ff,stroke:#6366f1
    style C fill:#064e3b,color:#d1fae5,stroke:#10b981
    style D fill:#1e3a8a,color:#dbeafe,stroke:#3b82f6
    style E fill:#500724,color:#fce7f3,stroke:#ec4899
    style F fill:#3b0764,color:#f3e8ff,stroke:#8b5cf6
    style G fill:#134e4a,color:#ccfbf1,stroke:#14b8a6
    style H fill:#431407,color:#ffedd5,stroke:#f97316
```

---

## Method-Problem Decision Matrix

```mermaid
quadrantChart
    title AI Method Suitability for Energy Problems
    x-axis Low Data Requirements --> High Data Requirements
    y-axis Low Complexity --> High Complexity
    quadrant-1 Advanced Research
    quadrant-2 Data-Hungry Models
    quadrant-3 Quick Wins
    quadrant-4 Engineering Methods
    Physics-Informed NN: [0.7, 0.85]
    MARL Grid Control: [0.75, 0.9]
    Deep TS Forecasting: [0.8, 0.7]
    Convex OPF: [0.35, 0.75]
    RAG Energy Docs: [0.6, 0.5]
    Anomaly Detection: [0.65, 0.55]
    Classical LP/MIP: [0.3, 0.6]
    Battery ML: [0.7, 0.65]
    LLM Agents: [0.55, 0.7]
    Prophet Forecasting: [0.45, 0.3]
```

---

## Framework Dependency Map

```mermaid
graph LR
    LLM["LLM Backbone\nGPT-4o · Claude · Llama"]

    LLM --> LC["LangChain"]
    LLM --> LI["LlamaIndex"]
    LLM --> CR["CrewAI"]
    LLM --> AG["AutoGen"]
    LLM --> LG["LangGraph"]
    LLM --> AN["Agno"]

    LC --> RAG["RAG Pipelines"]
    LI --> RAG
    LG --> AGENTS["Stateful Agents"]
    CR --> AGENTS
    AG --> AGENTS
    AN --> AGENTS

    RAG --> VDB["Vector DB\nChroma · Pinecone · Weaviate"]
    AGENTS --> TOOLS["Tool Layer\nSearch · Code · APIs"]

    TOOLS --> ENERGY["Energy Domain\nGrid2Op · PyPSA · PVLib"]
    TOOLS --> OPT["Optimisation\nOR-Tools · Pyomo · CVXPY"]
    TOOLS --> DATA["Data Layer\nDarts · sktime · GluonTS"]

    style LLM fill:#1e1b4b,color:#fff,stroke:#6366f1
    style ENERGY fill:#78350f,color:#fef3c7,stroke:#f59e0b
    style OPT fill:#1e3a8a,color:#dbeafe,stroke:#3b82f6
    style DATA fill:#134e4a,color:#ccfbf1,stroke:#14b8a6
```

---

## Table of Contents

- [Method Clusters](#method-clusters)
- [Energy Intelligence Hub](#energy-intelligence-hub)
- [Agentic AI and Multi-Agent Systems](#agentic-ai-and-multi-agent-systems)
- [RAG and Retrieval-Augmented Generation](#rag-and-retrieval-augmented-generation)
- [Optimisation and Operations Research](#optimisation-and-operations-research)
- [Reinforcement Learning](#reinforcement-learning)
- [Graph and Network AI](#graph-and-network-ai)
- [Forecasting and Time-Series Modelling](#forecasting-and-time-series-modelling)
- [Digital Twins and Simulation](#digital-twins-and-simulation)
- [ML Frameworks and Agentic Platforms](#ml-frameworks-and-agentic-platforms)
- [Related Repositories by the Author](#related-repositories-by-the-author)
- [Contributing](#contributing)
- [Star History](#star-history)

---

## Method Clusters

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

### Demand and Load Forecasting

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| NILMTK | ML / Signal | [![Stars](https://img.shields.io/github/stars/nilmtk/nilmtk?style=flat-square)](https://github.com/nilmtk/nilmtk) | Non-intrusive load monitoring — smart meter disaggregation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/nilmtk/nilmtk) |
| Darts | Unified DL | [![Stars](https://img.shields.io/github/stars/unit8co/darts?style=flat-square)](https://github.com/unit8co/darts) | TFT, N-BEATS, LSTM, XGBoost unified API for energy demand | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/unit8co/darts) |
| Neuralforecast | Neural TS | [![Stars](https://img.shields.io/github/stars/Nixtla/neuralforecast?style=flat-square)](https://github.com/Nixtla/neuralforecast) | NHITS, PatchTST, iTransformer for multi-horizon forecasting | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Nixtla/neuralforecast) |
| GluonTS | Probabilistic DL | [![Stars](https://img.shields.io/github/stars/awslabs/gluonts?style=flat-square)](https://github.com/awslabs/gluonts) | AWS probabilistic time-series toolkit with DeepAR | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/awslabs/gluonts) |
| Prophet | Bayesian | [![Stars](https://img.shields.io/github/stars/facebook/prophet?style=flat-square)](https://github.com/facebook/prophet) | Seasonal demand pattern forecasting by Meta | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/facebook/prophet) |
| TimesFM | Foundation Model | [![Stars](https://img.shields.io/github/stars/google-research/timesfm?style=flat-square)](https://github.com/google-research/timesfm) | Google time-series foundation model | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/google-research/timesfm) |
| Chronos | LLM for TS | [![Stars](https://img.shields.io/github/stars/amazon-science/chronos-forecasting?style=flat-square)](https://github.com/amazon-science/chronos-forecasting) | Amazon language model pretrained on time-series data | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/amazon-science/chronos-forecasting) |
| skforecast | ML / Sklearn | [![Stars](https://img.shields.io/github/stars/JoaquinAmatRodrigo/skforecast?style=flat-square)](https://github.com/JoaquinAmatRodrigo/skforecast) | Recursive multi-step forecasting with scikit-learn | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/JoaquinAmatRodrigo/skforecast) |

### Grid Optimisation and Power Systems

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PyPSA | Network Opt | [![Stars](https://img.shields.io/github/stars/PyPSA/PyPSA?style=flat-square)](https://github.com/PyPSA/PyPSA) | Open energy system modelling and power flow analysis | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/PyPSA/PyPSA) |
| Pandapower | Power Flow | [![Stars](https://img.shields.io/github/stars/e2nIEE/pandapower?style=flat-square)](https://github.com/e2nIEE/pandapower) | Power system analysis and optimisation framework | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/e2nIEE/pandapower) |
| Pyomo | MIP / NLP | [![Stars](https://img.shields.io/github/stars/Pyomo/pyomo?style=flat-square)](https://github.com/Pyomo/pyomo) | Algebraic modelling for unit commitment, economic dispatch, OPF | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Pyomo/pyomo) |
| PowerModels.jl | Convex OPF | [![Stars](https://img.shields.io/github/stars/lanl-ansi/PowerModels.jl?style=flat-square)](https://github.com/lanl-ansi/PowerModels.jl) | AC Optimal Power Flow and security-constrained UC in Julia | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/lanl-ansi/PowerModels.jl) |
| CVXPY | Convex Opt | [![Stars](https://img.shields.io/github/stars/cvxpy/cvxpy?style=flat-square)](https://github.com/cvxpy/cvxpy) | Disciplined convex programming for OPF, portfolio, demand response | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/cvxpy/cvxpy) |
| OR-Tools | CP / MIP | [![Stars](https://img.shields.io/github/stars/google/or-tools?style=flat-square)](https://github.com/google/or-tools) | Google combinatorial optimisation for dispatch and scheduling | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/google/or-tools) |
| EnergyPlus | Building Sim | [![Stars](https://img.shields.io/github/stars/NREL/EnergyPlus?style=flat-square)](https://github.com/NREL/EnergyPlus) | DOE whole-building energy simulation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/NREL/EnergyPlus) |

### Reinforcement Learning for Energy Systems

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| Grid2Op | MARL | [![Stars](https://img.shields.io/github/stars/rte-france/Grid2Op?style=flat-square)](https://github.com/rte-france/Grid2Op) | RL environment for power grid operation by RTE France | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/rte-france/Grid2Op) |
| CityLearn | MARL / Buildings | [![Stars](https://img.shields.io/github/stars/intelligent-environments-lab/CityLearn?style=flat-square)](https://github.com/intelligent-environments-lab/CityLearn) | Multi-agent RL for smart building energy management | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/intelligent-environments-lab/CityLearn) |
| EV2Gym | RL / Sim | [![Stars](https://img.shields.io/github/stars/StavrosOrf/EV2Gym?style=flat-square)](https://github.com/StavrosOrf/EV2Gym) | EV charging station simulator for RL-based smart charging | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/StavrosOrf/EV2Gym) |
| Sinergym | Deep RL | [![Stars](https://img.shields.io/github/stars/ugr-sail/sinergym?style=flat-square)](https://github.com/ugr-sail/sinergym) | RL environment wrapping EnergyPlus for smart buildings | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/ugr-sail/sinergym) |
| PowerGym | RL / Distribution | [![Stars](https://img.shields.io/github/stars/siemens/powergym?style=flat-square)](https://github.com/siemens/powergym) | RL for voltage control in distribution networks | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/siemens/powergym) |

### Battery Storage and EV Charging

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PyBaMM | Physics ML | [![Stars](https://img.shields.io/github/stars/pybamm-team/PyBaMM?style=flat-square)](https://github.com/pybamm-team/PyBaMM) | Fast battery modelling and state-of-health estimation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/pybamm-team/PyBaMM) |
| BatteryML | ML / LSTM | [![Stars](https://img.shields.io/github/stars/microsoft/BatteryML?style=flat-square)](https://github.com/microsoft/BatteryML) | Microsoft battery lifetime prediction and degradation modelling | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/BatteryML) |

### Renewable Energy Generation

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PVLib Python | Physics + ML | [![Stars](https://img.shields.io/github/stars/pvlib/pvlib-python?style=flat-square)](https://github.com/pvlib/pvlib-python) | Simulate and forecast solar PV system performance | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/pvlib/pvlib-python) |
| Open-Meteo | API + ML | [![Stars](https://img.shields.io/github/stars/open-meteo/open-meteo?style=flat-square)](https://github.com/open-meteo/open-meteo) | Open weather and solar forecasting API for renewable planning | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/open-meteo/open-meteo) |

### Fault Detection and Predictive Maintenance

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PyOD | 45+ Algorithms | [![Stars](https://img.shields.io/github/stars/yzhao062/pyod?style=flat-square)](https://github.com/yzhao062/pyod) | Isolation Forest, LOF, AutoEncoder for equipment monitoring | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/yzhao062/pyod) |
| Anomaly Transformer | Attention | [![Stars](https://img.shields.io/github/stars/thuml/Anomaly-Transformer?style=flat-square)](https://github.com/thuml/Anomaly-Transformer) | Transformer-based anomaly detection for sensor time series | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/thuml/Anomaly-Transformer) |
| Merlion | AutoML | [![Stars](https://img.shields.io/github/stars/salesforce/Merlion?style=flat-square)](https://github.com/salesforce/Merlion) | Salesforce multi-algorithm anomaly detection and forecasting | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/salesforce/Merlion) |

### Carbon, Sustainability and Net-Zero

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| ElectricityMaps | API + ML | [![Stars](https://img.shields.io/github/stars/electricitymaps/electricitymaps-contrib?style=flat-square)](https://github.com/electricitymaps/electricitymaps-contrib) | Real-time CO2 intensity and electricity flow map | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/electricitymaps/electricitymaps-contrib) |
| CarbonTracker | Measurement | [![Stars](https://img.shields.io/github/stars/lfwa/carbontracker?style=flat-square)](https://github.com/lfwa/carbontracker) | Track carbon footprint of ML model training runs | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/lfwa/carbontracker) |
| ClimateLearn | Deep Learning | [![Stars](https://img.shields.io/github/stars/aditya-grover/climate-learn?style=flat-square)](https://github.com/aditya-grover/climate-learn) | ML for weather and climate science (UCLA) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/aditya-grover/climate-learn) |

---

## Agentic AI and Multi-Agent Systems

### CrewAI

| Use Case | Domain | Method | Stars | Code |
|----------|--------|--------|-------|------|
| Email Auto Responder | Communication | Multi-Agent Flow | 1k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/flows/email_auto_responder_flow) |
| Marketing Strategy | Marketing | Planner Agent | 800+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy) |
| Stock Analysis | Finance | Tool-use Agent | 600+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) |
| Recruitment Workflow | HR | Matching Agent | 400+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/recruitment) |
| Lead Score Flow | Sales | Scoring Agent | 250+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/flows/lead-score-flow) |
| Landing Page Generator | Web Dev | Code Agent | 300+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator) |
| Game Builder Crew | Gaming | Multi-Agent | 200+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/game-builder-crew) |

### AutoGen

| Use Case | Domain | Method | Stars | Code |
|----------|--------|--------|-------|------|
| AutoGen Core | General | Conversational Agents | [![Stars](https://img.shields.io/github/stars/microsoft/autogen?style=flat-square)](https://github.com/microsoft/autogen) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/autogen) |
| Magentic-One | Complex Tasks | Orchestrator + Agents | 2k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one) |
| AI Medical Diagnostics | Healthcare | RAG Agent | 500+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/ahmadvh/AI-Agents-for-Medical-Diagnostics) |
| StockAgent | Finance | Trading Agent | 400+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/MingyuJ666/Stockagent) |

### LangGraph

| Use Case | Domain | Method | Stars | Code |
|----------|--------|--------|-------|------|
| Plan-and-Execute Agent | General | Planning + Execution | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb) |
| Reflection Agent | General | Self-Critique Loop | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflection/reflection.ipynb) |
| Adaptive RAG | Information Retrieval | Query-adaptive RAG | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag.ipynb) |
| Customer Support Agent | Service | Conversational | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/customer-support/customer-support.ipynb) |
| Multi-Agent Collaboration | Orchestration | Hierarchical Agents | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb) |
| SQL Agent | Data | DB Query Agent | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/sql-agent.ipynb) |
| Reflexion Agent | General | Iterative Reasoning | 5k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflexion/reflexion.ipynb) |

### Agno (formerly Phidata)

| Use Case | Domain | Method | Stars | Code |
|----------|--------|--------|-------|------|
| Finance Agent | Finance | Tool-use | 1k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/finance_agent.py) |
| Research Agent | Research | Web Search Agent | 1k+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/research_agent.py) |
| Legal Document Agent | Legal | RAG + Analysis | 500+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/legal_consultant.py) |
| Financial Reasoning Agent | Finance | Reasoning + Data | 500+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/reasoning_finance_agent.py) |
| Media Trend Analysis | Media | Trend Agent | 300+ | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/media_trend_analysis_agent.py) |

### Standalone Agentic Projects

| Project | Method | Stars | Code |
|---------|--------|-------|------|
| SuperAGI | Autonomous Agent Platform | [![Stars](https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=flat-square)](https://github.com/TransformerOptimus/SuperAGI) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/TransformerOptimus/SuperAGI) |
| AgentGPT | Web Agent | [![Stars](https://img.shields.io/github/stars/reworkd/AgentGPT?style=flat-square)](https://github.com/reworkd/AgentGPT) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/reworkd/AgentGPT) |
| BabyAGI | Task-driven Agent | [![Stars](https://img.shields.io/github/stars/yoheinakajima/babyagi?style=flat-square)](https://github.com/yoheinakajima/babyagi) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/yoheinakajima/babyagi) |
| MetaGPT | Multi-Agent Software | [![Stars](https://img.shields.io/github/stars/geekan/MetaGPT?style=flat-square)](https://github.com/geekan/MetaGPT) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/geekan/MetaGPT) |
| OpenDevin | Code Agent | [![Stars](https://img.shields.io/github/stars/OpenDevin/OpenDevin?style=flat-square)](https://github.com/OpenDevin/OpenDevin) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/OpenDevin/OpenDevin) |
| CAMEL | Role-playing Agents | [![Stars](https://img.shields.io/github/stars/camel-ai/camel?style=flat-square)](https://github.com/camel-ai/camel) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/camel-ai/camel) |
| NirDiamant GenAI Agents | Survey + 40 Tutorials | [![Stars](https://img.shields.io/github/stars/NirDiamant/GenAI_Agents?style=flat-square)](https://github.com/NirDiamant/GenAI_Agents) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/NirDiamant/GenAI_Agents) |

---

## RAG and Retrieval-Augmented Generation

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| GraphRAG | Graph + RAG | [![Stars](https://img.shields.io/github/stars/microsoft/graphrag?style=flat-square)](https://github.com/microsoft/graphrag) | Microsoft community-level summarisation for large corpora | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/graphrag) |
| LlamaIndex | Data + LLM | [![Stars](https://img.shields.io/github/stars/run-llama/llama_index?style=flat-square)](https://github.com/run-llama/llama_index) | Document indexing and retrieval augmented generation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/run-llama/llama_index) |
| RAPTOR | Hierarchical RAG | [![Stars](https://img.shields.io/github/stars/parthsarthi03/raptor?style=flat-square)](https://github.com/parthsarthi03/raptor) | Recursive abstractive processing for tree-organised retrieval | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/parthsarthi03/raptor) |
| Adaptive RAG | Query-adaptive | 5k+ | Dynamically adjusts retrieval strategy by query complexity | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag.ipynb) |
| Corrective RAG | Self-correction | 5k+ | Evaluates and corrects retrieved documents before generation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_crag.ipynb) |
| Self-RAG | Reflective | 5k+ | Model reflects on retrieval need and output quality | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_self_rag.ipynb) |
| Agentic RAG | Tool-augmented | 5k+ | Agent selects optimal retrieval strategy dynamically | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_agentic_rag.ipynb) |
| PrivateGPT | Local RAG | [![Stars](https://img.shields.io/github/stars/zylon-ai/private-gpt?style=flat-square)](https://github.com/zylon-ai/private-gpt) | 100% private document chatbot, no internet required | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/zylon-ai/private-gpt) |
| RAG Techniques Survey | Survey + Code | [![Stars](https://img.shields.io/github/stars/NirDiamant/RAG_Techniques?style=flat-square)](https://github.com/NirDiamant/RAG_Techniques) | 40+ RAG techniques with working Python implementations | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/NirDiamant/RAG_Techniques) |

---

## Optimisation and Operations Research

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PuLP | LP / MIP | [![Stars](https://img.shields.io/github/stars/coin-or/pulp?style=flat-square)](https://github.com/coin-or/pulp) | Linear and integer programming for energy scheduling | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/coin-or/pulp) |
| OR-Tools | CP / MIP / VRP | [![Stars](https://img.shields.io/github/stars/google/or-tools?style=flat-square)](https://github.com/google/or-tools) | Google combinatorial optimisation suite | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/google/or-tools) |
| CVXPY | Convex Opt | [![Stars](https://img.shields.io/github/stars/cvxpy/cvxpy?style=flat-square)](https://github.com/cvxpy/cvxpy) | Disciplined convex programming for power systems | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/cvxpy/cvxpy) |
| Optuna | Bayesian Opt | [![Stars](https://img.shields.io/github/stars/optuna/optuna?style=flat-square)](https://github.com/optuna/optuna) | Hyperparameter and architecture optimisation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/optuna/optuna) |
| Pymoo | Multi-objective EA | [![Stars](https://img.shields.io/github/stars/msu-coinlab/pymoo?style=flat-square)](https://github.com/msu-coinlab/pymoo) | NSGA-II/III for multi-objective energy system design | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/msu-coinlab/pymoo) |
| DEAP | Genetic / EA | [![Stars](https://img.shields.io/github/stars/DEAP/deap?style=flat-square)](https://github.com/DEAP/deap) | Distributed Evolutionary Algorithms in Python | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/DEAP/deap) |
| OptiGuide | LLM + OR | [![Stars](https://img.shields.io/github/stars/microsoft/OptiGuide?style=flat-square)](https://github.com/microsoft/OptiGuide) | Microsoft LLM-guided supply chain optimisation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/OptiGuide) |
| Gurobi ML | ML + MIP | [![Stars](https://img.shields.io/github/stars/Gurobi/gurobi-machinelearning?style=flat-square)](https://github.com/Gurobi/gurobi-machinelearning) | Embed trained ML models inside Gurobi MIP formulations | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Gurobi/gurobi-machinelearning) |

---

## Reinforcement Learning

### Core Libraries

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| Stable-Baselines3 | PPO / SAC / TD3 | [![Stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3?style=flat-square)](https://github.com/DLR-RM/stable-baselines3) | Reliable RL implementations for energy control tasks | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/DLR-RM/stable-baselines3) |
| RLlib (Ray) | Scalable RL | [![Stars](https://img.shields.io/github/stars/ray-project/ray?style=flat-square)](https://github.com/ray-project/ray) | Production RL at scale for multi-agent systems | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/ray-project/ray) |
| CleanRL | Single-file RL | [![Stars](https://img.shields.io/github/stars/vwxyzjn/cleanrl?style=flat-square)](https://github.com/vwxyzjn/cleanrl) | Readable single-file RL algorithm implementations | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/vwxyzjn/cleanrl) |
| TorchRL | PyTorch RL | [![Stars](https://img.shields.io/github/stars/pytorch/rl?style=flat-square)](https://github.com/pytorch/rl) | PyTorch-native reinforcement learning library | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/pytorch/rl) |
| Gymnasium | Environments | [![Stars](https://img.shields.io/github/stars/Farama-Foundation/Gymnasium?style=flat-square)](https://github.com/Farama-Foundation/Gymnasium) | OpenAI Gym successor — standard RL environments | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Farama-Foundation/Gymnasium) |
| FinRL | Finance RL | [![Stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRL?style=flat-square)](https://github.com/AI4Finance-Foundation/FinRL) | Deep RL for quantitative finance and energy trading | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/AI4Finance-Foundation/FinRL) |

---

## Graph and Network AI

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PyTorch Geometric | GNN Framework | [![Stars](https://img.shields.io/github/stars/pyg-team/pytorch_geometric?style=flat-square)](https://github.com/pyg-team/pytorch_geometric) | Standard GNN library — power grid topology analysis | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/pyg-team/pytorch_geometric) |
| DGL | Deep Graph Library | [![Stars](https://img.shields.io/github/stars/dmlc/dgl?style=flat-square)](https://github.com/dmlc/dgl) | Heterogeneous network problems and graph learning | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/dmlc/dgl) |
| NetworkX | Graph Analysis | [![Stars](https://img.shields.io/github/stars/networkx/networkx?style=flat-square)](https://github.com/networkx/networkx) | Python graph analysis for power and supply chain networks | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/networkx/networkx) |
| Neo4j GenAI | Knowledge Graph | [![Stars](https://img.shields.io/github/stars/neo4j/neo4j-genai-python?style=flat-square)](https://github.com/neo4j/neo4j-genai-python) | Graph RAG combining knowledge graphs with LLMs | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/neo4j/neo4j-genai-python) |
| GraphSAGE | Inductive GNN | [![Stars](https://img.shields.io/github/stars/williamleif/GraphSAGE?style=flat-square)](https://github.com/williamleif/GraphSAGE) | Scalable inductive learning on large graph networks | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/williamleif/GraphSAGE) |

---

## Forecasting and Time-Series Modelling

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| Darts | Unified DL / ML | [![Stars](https://img.shields.io/github/stars/unit8co/darts?style=flat-square)](https://github.com/unit8co/darts) | TFT, N-BEATS, LSTM, XGBoost unified forecasting API | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/unit8co/darts) |
| StatsForecast | Classical + Fast | [![Stars](https://img.shields.io/github/stars/Nixtla/statsforecast?style=flat-square)](https://github.com/Nixtla/statsforecast) | ETS, ARIMA, Theta at scale with 100x speedup | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Nixtla/statsforecast) |
| TimesFM | Foundation Model | [![Stars](https://img.shields.io/github/stars/google-research/timesfm?style=flat-square)](https://github.com/google-research/timesfm) | Google time-series foundation model | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/google-research/timesfm) |
| Chronos | LLM for TS | [![Stars](https://img.shields.io/github/stars/amazon-science/chronos-forecasting?style=flat-square)](https://github.com/amazon-science/chronos-forecasting) | Amazon language model pretrained on time-series data | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/amazon-science/chronos-forecasting) |
| Moirai | Foundation Model | [![Stars](https://img.shields.io/github/stars/SalesforceAIResearch/uni2ts?style=flat-square)](https://github.com/SalesforceAIResearch/uni2ts) | Salesforce universal time-series forecasting model | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/SalesforceAIResearch/uni2ts) |
| sktime | Unified ML | [![Stars](https://img.shields.io/github/stars/sktime/sktime?style=flat-square)](https://github.com/sktime/sktime) | Sklearn-compatible time-series ML toolkit | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/sktime/sktime) |
| PyOD | Anomaly Detection | [![Stars](https://img.shields.io/github/stars/yzhao062/pyod?style=flat-square)](https://github.com/yzhao062/pyod) | 45+ anomaly detection algorithms for equipment monitoring | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/yzhao062/pyod) |
| Merlion | AutoML Anomaly | [![Stars](https://img.shields.io/github/stars/salesforce/Merlion?style=flat-square)](https://github.com/salesforce/Merlion) | Multi-algorithm anomaly detection and forecasting | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/salesforce/Merlion) |

---

## Digital Twins and Simulation

| Project | Method | Stars | Description | Code |
|---------|--------|-------|-------------|------|
| PINNs | Physics-Informed NN | [![Stars](https://img.shields.io/github/stars/maziarraissi/PINNs?style=flat-square)](https://github.com/maziarraissi/PINNs) | Physics-Informed Neural Networks for energy systems | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/maziarraissi/PINNs) |
| EnergyPlus | Building Simulation | [![Stars](https://img.shields.io/github/stars/NREL/EnergyPlus?style=flat-square)](https://github.com/NREL/EnergyPlus) | DOE whole-building energy simulation engine | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/NREL/EnergyPlus) |
| Sinergym | RL + Simulation | [![Stars](https://img.shields.io/github/stars/ugr-sail/sinergym?style=flat-square)](https://github.com/ugr-sail/sinergym) | RL environment wrapping EnergyPlus for smart buildings | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/ugr-sail/sinergym) |
| Azure Digital Twins | Cloud IoT + ML | [![Stars](https://img.shields.io/github/stars/Azure-Samples/digital-twins-samples?style=flat-square)](https://github.com/Azure-Samples/digital-twins-samples) | Azure IoT + ML pipeline for industrial digital twins | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Azure-Samples/digital-twins-samples) |
| PyBaMM | Battery Physics ML | [![Stars](https://img.shields.io/github/stars/pybamm-team/PyBaMM?style=flat-square)](https://github.com/pybamm-team/PyBaMM) | Fast battery simulation and state-of-health estimation | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/pybamm-team/PyBaMM) |

---

## ML Frameworks and Agentic Platforms

| Framework | Category | Stars | Code |
|-----------|---------|-------|------|
| PyTorch | DL Framework | [![Stars](https://img.shields.io/github/stars/pytorch/pytorch?style=flat-square)](https://github.com/pytorch/pytorch) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/pytorch/pytorch) |
| Scikit-learn | Classical ML | [![Stars](https://img.shields.io/github/stars/scikit-learn/scikit-learn?style=flat-square)](https://github.com/scikit-learn/scikit-learn) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/scikit-learn/scikit-learn) |
| HuggingFace Transformers | Foundation Models | [![Stars](https://img.shields.io/github/stars/huggingface/transformers?style=flat-square)](https://github.com/huggingface/transformers) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/huggingface/transformers) |
| LangChain | LLM Orchestration | [![Stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat-square)](https://github.com/langchain-ai/langchain) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langchain) |
| LlamaIndex | Data + LLM | [![Stars](https://img.shields.io/github/stars/run-llama/llama_index?style=flat-square)](https://github.com/run-llama/llama_index) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/run-llama/llama_index) |
| CrewAI | Multi-Agent | [![Stars](https://img.shields.io/github/stars/crewAIInc/crewAI?style=flat-square)](https://github.com/crewAIInc/crewAI) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/crewAIInc/crewAI) |
| AutoGen | Conversational Agents | [![Stars](https://img.shields.io/github/stars/microsoft/autogen?style=flat-square)](https://github.com/microsoft/autogen) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/autogen) |
| LangGraph | Graph-based Agents | [![Stars](https://img.shields.io/github/stars/langchain-ai/langgraph?style=flat-square)](https://github.com/langchain-ai/langgraph) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/langchain-ai/langgraph) |
| Agno | Agentic AI | [![Stars](https://img.shields.io/github/stars/agno-agi/agno?style=flat-square)](https://github.com/agno-agi/agno) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/agno-agi/agno) |
| XGBoost | Gradient Boosting | [![Stars](https://img.shields.io/github/stars/dmlc/xgboost?style=flat-square)](https://github.com/dmlc/xgboost) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/dmlc/xgboost) |
| LightGBM | GBDT | [![Stars](https://img.shields.io/github/stars/microsoft/LightGBM?style=flat-square)](https://github.com/microsoft/LightGBM) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/microsoft/LightGBM) |

---

## Related Repositories by the Author

| Repository | Topic | Stars | Code |
|-----------|-------|-------|------|
| Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python | Linear programming and integer optimisation | [![Stars](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python?style=flat-square)](https://github.com/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python) |
| Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX | Graph and network analysis | [![Stars](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX?style=flat-square)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX) |
| Python_MasterClass | Python programming fundamentals to advanced | [![Stars](https://img.shields.io/github/stars/Harrypatria/Python_MasterClass?style=flat-square)](https://github.com/Harrypatria/Python_MasterClass) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Harrypatria/Python_MasterClass) |
| SQLite_Advanced_Tutorial_Google_Colab | Advanced SQL and database programming | [![Stars](https://img.shields.io/github/stars/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab?style=flat-square)](https://github.com/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab) |
| ML_BERT_Prediction | BERT for classification and NLP | [![Stars](https://img.shields.io/github/stars/Harrypatria/ML_BERT_Prediction?style=flat-square)](https://github.com/Harrypatria/ML_BERT_Prediction) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Harrypatria/ML_BERT_Prediction) |
| Python-Programming-for-Everyone-From-Basics-to-Advanced | Python for all levels | [![Stars](https://img.shields.io/github/stars/Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced?style=flat-square)](https://github.com/Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced) | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Harrypatria/Python-Programming-for-Everyone-From-Basics-to-Advanced) |

---

## Contributing

1. Fork this repository.
2. Locate the correct method cluster section.
3. Add a row using this exact format:

```markdown
| Project Name | Method | [![Stars](https://img.shields.io/github/stars/owner/repo?style=flat-square)](https://github.com/owner/repo) | One-sentence description | [![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/owner/repo) |
```

Requirements: the project must have 100+ GitHub stars or forks, and a working implementation. Energy-sector projects are prioritised.

---

## Star History

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="https://api.star-history.com/svg?repos=nilmtk/nilmtk,rte-france/Grid2Op,PyPSA/PyPSA,pybamm-team/PyBaMM,unit8co/darts&type=Date&theme=dark"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="https://api.star-history.com/svg?repos=nilmtk/nilmtk,rte-france/Grid2Op,PyPSA/PyPSA,pybamm-team/PyBaMM,unit8co/darts&type=Date"
  />
  <img
    alt="Star History — Featured Energy AI Repositories"
    src="https://api.star-history.com/svg?repos=nilmtk/nilmtk,rte-france/Grid2Op,PyPSA/PyPSA,pybamm-team/PyBaMM,unit8co/darts&type=Date"
  />
</picture>

---

MIT License · [github.com/Harrypatria](https://github.com/Harrypatria) · [patriaco.co.uk](https://www.patriaco.co.uk) · [ORCID 0000-0002-7844-538X](https://orcid.org/0000-0002-7844-538X)