import { useState, useEffect, useRef } from "react";

// ─── DATA ────────────────────────────────────────────────────────────────────

const ABSTRACT = `The energy sector is undergoing a fundamental transformation driven by the convergence of artificial intelligence, machine learning, and advanced optimisation techniques. This repository curates 500+ production-grade open-source projects spanning agentic AI systems, retrieval-augmented generation, reinforcement learning for grid control, physics-informed neural networks, and graph neural networks applied to power infrastructure.

Key industry trends shaping this collection include: the rise of foundation models for universal time-series forecasting (Chronos, TimesFM), multi-agent reinforcement learning for decentralised energy dispatch, LLM-guided optimisation for unit commitment and economic dispatch, and digital twin architectures integrating real-time sensor fusion with predictive maintenance. Battery management systems increasingly rely on physics-informed ML, while demand response programmes leverage multi-agent coordination under uncertainty.

Organised by technical method rather than industry sector, this resource enables practitioners to select proven algorithms for specific problems — from NILMTK's non-intrusive load monitoring to Grid2Op's power grid RL environment. Every entry is verified against 100+ GitHub stars, ensuring community-validated quality. Aligned with the net-zero transition, this repository prioritises carbon-aware computing, renewable integration, and smart grid intelligence as foundational pillars of the AI-enabled energy future.`;

const BADGES = [
  { label: "Stars", color: "#f59e0b", icon: "★", value: "500+ Projects" },
  { label: "Energy Focus", color: "#10b981", icon: "⚡", value: "80+ Repos" },
  { label: "Methods", color: "#6366f1", icon: "◈", value: "12 Clusters" },
  { label: "Min Stars/Forks", color: "#ec4899", icon: "◆", value: "100+ Each" },
];

const METHOD_CLUSTERS = [
  { id: "energy", label: "Energy Intelligence", count: 82, color: "#f59e0b", description: "Grid, solar, wind, battery, EV, carbon" },
  { id: "agentic", label: "Agentic AI", count: 74, color: "#6366f1", description: "CrewAI, AutoGen, LangGraph, Agno" },
  { id: "rag", label: "RAG & LLMs", count: 61, color: "#10b981", description: "Adaptive, Corrective, Self-RAG, GraphRAG" },
  { id: "optimisation", label: "Optimisation & OR", count: 53, color: "#3b82f6", description: "LP, MIP, metaheuristics, scheduling" },
  { id: "rl", label: "Reinforcement Learning", count: 52, color: "#ec4899", description: "PPO, SAC, DQN, MARL, energy control" },
  { id: "gnn", label: "Graph & Network AI", count: 43, color: "#8b5cf6", description: "GNN, knowledge graphs, topology" },
  { id: "forecast", label: "Forecasting & TS", count: 51, color: "#14b8a6", description: "Transformers, LSTM, foundation models" },
  { id: "twin", label: "Digital Twins", count: 34, color: "#f97316", description: "PINN, simulation, asset monitoring" },
];

const ENERGY_PROJECTS = [
  {
    category: "Demand Forecasting",
    projects: [
      { name: "NILMTK", desc: "Non-intrusive load monitoring toolkit — smart meter disaggregation", method: "ML / Signal", url: "https://github.com/nilmtk/nilmtk", stars: "1.2k" },
      { name: "Darts", desc: "Unified time-series library: TFT, NBEATS, LSTM for energy demand", method: "Deep Learning", url: "https://github.com/unit8co/darts", stars: "7.8k" },
      { name: "Neuralforecast", desc: "NHITS, PatchTST, iTransformer — state-of-art neural forecasters", method: "Neural TS", url: "https://github.com/Nixtla/neuralforecast", stars: "3.1k" },
      { name: "GluonTS", desc: "AWS probabilistic time-series with DeepAR for energy", method: "Probabilistic DL", url: "https://github.com/awslabs/gluonts", stars: "4.4k" },
      { name: "Prophet", desc: "Seasonal demand pattern forecasting by Facebook", method: "Bayesian", url: "https://github.com/facebook/prophet", stars: "18k" },
      { name: "TimesFM", desc: "Google's time-series foundation model", method: "Foundation Model", url: "https://github.com/google-research/timesfm", stars: "3.5k" },
      { name: "Chronos", desc: "Amazon language model pretrained on time-series data", method: "LLM for TS", url: "https://github.com/amazon-science/chronos-forecasting", stars: "2.9k" },
    ]
  },
  {
    category: "Grid Optimisation",
    projects: [
      { name: "PyPSA", desc: "Python for Power System Analysis — open energy system modelling", method: "Network Opt", url: "https://github.com/PyPSA/PyPSA", stars: "1.3k" },
      { name: "Pandapower", desc: "Power system analysis and optimisation framework", method: "Power Flow", url: "https://github.com/e2nIEE/pandapower", stars: "900" },
      { name: "Pyomo", desc: "Algebraic modelling for UC, ED, OPF problems", method: "MIP / NLP", url: "https://github.com/Pyomo/pyomo", stars: "2.0k" },
      { name: "PowerModels.jl", desc: "AC Optimal Power Flow and security-constrained UC in Julia", method: "Convex Opt", url: "https://github.com/lanl-ansi/PowerModels.jl", stars: "600" },
      { name: "CVXPY", desc: "Convex optimisation — OPF, portfolio, demand response", method: "Convex Opt", url: "https://github.com/cvxpy/cvxpy", stars: "5.3k" },
      { name: "OR-Tools", desc: "Google's combinatorial optimisation suite for dispatch", method: "CP / MIP", url: "https://github.com/google/or-tools", stars: "11k" },
    ]
  },
  {
    category: "RL for Energy",
    projects: [
      { name: "Grid2Op", desc: "RL environment for power grid operation by RTE France", method: "MARL", url: "https://github.com/rte-france/Grid2Op", stars: "300" },
      { name: "CityLearn", desc: "Multi-agent RL for smart building energy management", method: "MARL", url: "https://github.com/intelligent-environments-lab/CityLearn", stars: "350" },
      { name: "EV2Gym", desc: "EV charging station simulator for RL-based smart charging", method: "RL / Sim", url: "https://github.com/StavrosOrf/EV2Gym", stars: "150" },
      { name: "Sinergym", desc: "RL environment wrapping EnergyPlus for smart buildings", method: "Deep RL", url: "https://github.com/ugr-sail/sinergym", stars: "250" },
    ]
  },
  {
    category: "Battery & EV",
    projects: [
      { name: "PyBaMM", desc: "Fast battery modelling and state-of-health estimation", method: "Physics ML", url: "https://github.com/pybamm-team/PyBaMM", stars: "950" },
      { name: "BatteryML", desc: "Microsoft battery lifetime prediction and degradation ML", method: "ML / LSTM", url: "https://github.com/microsoft/BatteryML", stars: "300" },
    ]
  },
  {
    category: "Renewable Generation",
    projects: [
      { name: "PVLib Python", desc: "Simulate solar PV system performance with ML enhancement", method: "Physics + ML", url: "https://github.com/pvlib/pvlib-python", stars: "1.2k" },
      { name: "Open-Meteo", desc: "Open weather & solar forecasting API for renewables", method: "API + ML", url: "https://github.com/open-meteo/open-meteo", stars: "3.5k" },
    ]
  },
  {
    category: "Fault Detection",
    projects: [
      { name: "PyOD", desc: "45+ anomaly detection algorithms for equipment monitoring", method: "Anomaly Det.", url: "https://github.com/yzhao062/pyod", stars: "8.7k" },
      { name: "Anomaly Transformer", desc: "Transformer-based anomaly detection for time series", method: "Attention", url: "https://github.com/thuml/Anomaly-Transformer", stars: "2.4k" },
      { name: "Merlion", desc: "Multi-algorithm anomaly detection and forecasting by Salesforce", method: "AutoML", url: "https://github.com/salesforce/Merlion", stars: "3.3k" },
    ]
  },
];

const AGENTIC_PROJECTS = [
  { framework: "CrewAI", name: "Email Auto Responder", industry: "Communication", method: "Multi-Agent", stars: "1k+", url: "https://github.com/crewAIInc/crewAI-examples/tree/main/flows/email_auto_responder_flow" },
  { framework: "CrewAI", name: "Marketing Strategy", industry: "Marketing", method: "Planner Agent", stars: "800+", url: "https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy" },
  { framework: "CrewAI", name: "Stock Analysis", industry: "Finance", method: "Tool-use Agent", stars: "600+", url: "https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis" },
  { framework: "AutoGen", name: "AutoGen Core", industry: "General", method: "Conversational", stars: "35k+", url: "https://github.com/microsoft/autogen" },
  { framework: "AutoGen", name: "Magentic-One", industry: "Complex Tasks", method: "Orchestrator+", stars: "2k+", url: "https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one" },
  { framework: "AutoGen", name: "Medical Diagnostics", industry: "Healthcare", method: "RAG Agent", stars: "500+", url: "https://github.com/ahmadvh/AI-Agents-for-Medical-Diagnostics" },
  { framework: "LangGraph", name: "Plan-and-Execute", industry: "General", method: "Plan+Exec", stars: "5k+", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb" },
  { framework: "LangGraph", name: "Reflection Agent", industry: "General", method: "Self-Critique", stars: "5k+", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflection/reflection.ipynb" },
  { framework: "LangGraph", name: "Multi-Agent Collab", industry: "Orchestration", method: "Hierarchical", stars: "5k+", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb" },
  { framework: "LangGraph", name: "SQL Agent", industry: "Data", method: "DB Query Agent", stars: "5k+", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/sql-agent.ipynb" },
  { framework: "Agno", name: "Finance Agent", industry: "Finance", method: "Tool-use", stars: "1k+", url: "https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/finance_agent.py" },
  { framework: "Agno", name: "Research Agent", industry: "Research", method: "Web Search", stars: "1k+", url: "https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/research_agent.py" },
  { framework: "Agno", name: "Legal Document Agent", industry: "Legal", method: "RAG + Analysis", stars: "500+", url: "https://github.com/agno-agi/agno/blob/main/cookbook/examples/agents/legal_consultant.py" },
];

const RAG_PROJECTS = [
  { name: "GraphRAG", method: "Graph + RAG", stars: "21k", desc: "Microsoft community-level summarisation over large text corpora", url: "https://github.com/microsoft/graphrag" },
  { name: "LlamaIndex", method: "Data + LLM", stars: "37k", desc: "Document indexing and retrieval augmented generation", url: "https://github.com/run-llama/llama_index" },
  { name: "RAPTOR", method: "Hierarchical", stars: "2.1k", desc: "Recursive abstractive processing for tree-organised retrieval", url: "https://github.com/parthsarthi03/raptor" },
  { name: "Adaptive RAG", method: "Query-adaptive", stars: "5k+", desc: "Dynamically adjusts retrieval strategy by query complexity", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag.ipynb" },
  { name: "Corrective RAG", method: "Self-correction", stars: "5k+", desc: "Evaluates and corrects retrieved docs before generation", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_crag.ipynb" },
  { name: "Self-RAG", method: "Reflective", stars: "5k+", desc: "Model reflects on retrieval need and output quality", url: "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_self_rag.ipynb" },
  { name: "PrivateGPT", method: "Local RAG", stars: "54k", desc: "100% private document chatbot with no internet dependency", url: "https://github.com/zylon-ai/private-gpt" },
  { name: "NirDiamant RAG", method: "Survey + Code", stars: "3k+", desc: "40+ RAG techniques with working implementations", url: "https://github.com/NirDiamant/RAG_Techniques" },
];

const OPTIMISATION_PROJECTS = [
  { name: "PuLP", method: "LP / MIP", stars: "2k", desc: "LP/MIP modelling in Python — energy scheduling and dispatch", url: "https://github.com/coin-or/pulp" },
  { name: "OR-Tools", method: "CP / MIP / VRP", stars: "11k", desc: "Google combinatorial optimisation suite", url: "https://github.com/google/or-tools" },
  { name: "CVXPY", method: "Convex", stars: "5.3k", desc: "Disciplined convex programming for power systems", url: "https://github.com/cvxpy/cvxpy" },
  { name: "Optuna", method: "Bayesian Opt", stars: "10k", desc: "Hyperparameter and architecture optimisation framework", url: "https://github.com/optuna/optuna" },
  { name: "Pymoo", method: "Multi-objective EA", stars: "2.3k", desc: "NSGA-II/III for multi-objective energy system design", url: "https://github.com/msu-coinlab/pymoo" },
  { name: "DEAP", method: "Genetic / EA", stars: "5.7k", desc: "Distributed Evolutionary Algorithms in Python", url: "https://github.com/DEAP/deap" },
  { name: "OptiGuide", method: "LLM + OR", stars: "600", desc: "Microsoft LLM-guided supply chain optimisation", url: "https://github.com/microsoft/OptiGuide" },
];

const RL_PROJECTS = [
  { name: "Stable-Baselines3", method: "PPO/SAC/TD3", stars: "9k", desc: "Reliable RL implementations for energy control", url: "https://github.com/DLR-RM/stable-baselines3" },
  { name: "RLlib (Ray)", method: "Scalable RL", stars: "34k", desc: "Production RL at scale for multi-agent systems", url: "https://github.com/ray-project/ray" },
  { name: "CleanRL", method: "Single-file RL", stars: "5.7k", desc: "Readable single-file RL implementations", url: "https://github.com/vwxyzjn/cleanrl" },
  { name: "FinRL", method: "Deep RL Finance", stars: "9.8k", desc: "Deep RL for quantitative finance and energy trading", url: "https://github.com/AI4Finance-Foundation/FinRL" },
  { name: "Grid2Op", method: "MARL Power Grid", stars: "300", desc: "RL environments for power grid operation (RTE France)", url: "https://github.com/rte-france/Grid2Op" },
  { name: "Gymnasium", method: "Environments", stars: "7.6k", desc: "OpenAI Gym successor — standard RL environments", url: "https://github.com/Farama-Foundation/Gymnasium" },
];

const GNN_PROJECTS = [
  { name: "PyTorch Geometric", method: "GNN Framework", stars: "21k", desc: "Standard GNN library — power grid topology analysis", url: "https://github.com/pyg-team/pytorch_geometric" },
  { name: "DGL", method: "Graph DL", stars: "13k", desc: "Deep Graph Library for heterogeneous network problems", url: "https://github.com/dmlc/dgl" },
  { name: "NetworkX", method: "Graph Analysis", stars: "14k", desc: "Python graph analysis for power and supply chain networks", url: "https://github.com/networkx/networkx" },
  { name: "GraphRAG", method: "Graph + LLM", stars: "21k", desc: "Community-level summarisation over large document corpora", url: "https://github.com/microsoft/graphrag" },
  { name: "Neo4j GenAI", method: "Knowledge Graph", stars: "400", desc: "Graph RAG combining knowledge graphs with LLMs", url: "https://github.com/neo4j/neo4j-genai-python" },
];

const FORECAST_PROJECTS = [
  { name: "Darts", method: "Unified DL/ML", stars: "7.8k", desc: "TFT, N-BEATS, LSTM, XGBoost unified forecasting API", url: "https://github.com/unit8co/darts" },
  { name: "Nixtla StatsForecast", method: "Classical + Fast", stars: "3.8k", desc: "ETS, ARIMA, Theta at scale (100x speedup)", url: "https://github.com/Nixtla/statsforecast" },
  { name: "TimesFM", method: "Foundation Model", stars: "3.5k", desc: "Google's time-series foundation model", url: "https://github.com/google-research/timesfm" },
  { name: "Chronos", method: "LLM for TS", stars: "2.9k", desc: "Amazon language model pretrained on time-series data", url: "https://github.com/amazon-science/chronos-forecasting" },
  { name: "Moirai", method: "Foundation Model", stars: "1.4k", desc: "Salesforce universal time-series forecasting model", url: "https://github.com/SalesforceAIResearch/uni2ts" },
  { name: "sktime", method: "Unified ML", stars: "7.8k", desc: "Sklearn-compatible time-series ML toolkit", url: "https://github.com/sktime/sktime" },
];

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

function ExternalLinkIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function StarIcon({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function ChevronDown({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function RepoLink({ url, name }) {
  const repoPath = url.replace("https://github.com/", "").split("/blob/")[0].split("/tree/")[0];
  const shortPath = repoPath.split("/").slice(0, 2).join("/");
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        color: "#6366f1",
        fontSize: "12px",
        fontFamily: "'JetBrains Mono', monospace",
        textDecoration: "none",
        padding: "2px 6px",
        borderRadius: "4px",
        border: "1px solid #e0e0ff",
        background: "#f5f3ff",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
        maxWidth: "200px",
        overflow: "hidden",
        textOverflow: "ellipsis",
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = "#ede9fe";
        e.currentTarget.style.borderColor = "#6366f1";
        e.currentTarget.style.color = "#4338ca";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = "#f5f3ff";
        e.currentTarget.style.borderColor = "#e0e0ff";
        e.currentTarget.style.color = "#6366f1";
      }}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style={{ flexShrink: 0 }}>
        <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
      </svg>
      <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{shortPath}</span>
      <ExternalLinkIcon />
    </a>
  );
}

function Badge({ label, value, color }) {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "16px 24px",
      background: "#fff",
      border: `1px solid ${color}22`,
      borderRadius: "12px",
      boxShadow: `0 2px 8px ${color}11`,
      minWidth: "120px",
    }}>
      <span style={{ fontSize: "22px", fontWeight: "800", color, letterSpacing: "-0.5px" }}>{value}</span>
      <span style={{ fontSize: "11px", color: "#888", marginTop: "2px", fontFamily: "inherit", letterSpacing: "0.5px", textTransform: "uppercase" }}>{label}</span>
    </div>
  );
}

function Tag({ label, color }) {
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: "999px",
      fontSize: "11px",
      fontWeight: "600",
      background: color + "15",
      color: color,
      border: `1px solid ${color}30`,
      letterSpacing: "0.2px",
    }}>
      {label}
    </span>
  );
}

// ─── INFOGRAPHIC: Method Cluster Radial Chart ─────────────────────────────────

function ClusterChart() {
  const [hovered, setHovered] = useState(null);
  const total = METHOD_CLUSTERS.reduce((s, c) => s + c.count, 0);
  const cx = 160, cy = 160, r = 120, innerR = 55;

  let cumAngle = -Math.PI / 2;
  const slices = METHOD_CLUSTERS.map((c, i) => {
    const angle = (c.count / total) * 2 * Math.PI;
    const startAngle = cumAngle;
    const endAngle = cumAngle + angle;
    cumAngle = endAngle;

    const midAngle = startAngle + angle / 2;
    const labelR = r + 22;
    const lx = cx + labelR * Math.cos(midAngle);
    const ly = cy + labelR * Math.sin(midAngle);

    const x1 = cx + r * Math.cos(startAngle), y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle), y2 = cy + r * Math.sin(endAngle);
    const xi1 = cx + innerR * Math.cos(startAngle), yi1 = cy + innerR * Math.sin(startAngle);
    const xi2 = cx + innerR * Math.cos(endAngle), yi2 = cy + innerR * Math.sin(endAngle);
    const largeArc = angle > Math.PI ? 1 : 0;

    const path = [
      `M ${xi1} ${yi1}`,
      `L ${x1} ${y1}`,
      `A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`,
      `L ${xi2} ${yi2}`,
      `A ${innerR} ${innerR} 0 ${largeArc} 0 ${xi1} ${yi1}`,
      "Z"
    ].join(" ");

    return { ...c, path, midAngle, lx, ly, angle };
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "16px" }}>
      <svg width="320" height="320" viewBox="0 0 320 320">
        <defs>
          {slices.map((s, i) => (
            <filter key={i} id={`glow-${i}`}>
              <feGaussianBlur stdDeviation="2" result="coloredBlur" />
              <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          ))}
        </defs>
        {slices.map((s, i) => (
          <path
            key={i}
            d={s.path}
            fill={hovered === i ? s.color : s.color + "cc"}
            stroke="#fff"
            strokeWidth={hovered === i ? 3 : 1.5}
            style={{ cursor: "pointer", transition: "all 0.2s", transform: hovered === i ? `scale(1.03)` : "scale(1)", transformOrigin: `${cx}px ${cy}px` }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          />
        ))}
        <circle cx={cx} cy={cy} r={innerR - 2} fill="white" stroke="#f0f0f0" strokeWidth="1" />
        {hovered !== null ? (
          <>
            <text x={cx} y={cy - 10} textAnchor="middle" fill={METHOD_CLUSTERS[hovered].color} fontSize="22" fontWeight="800">{METHOD_CLUSTERS[hovered].count}</text>
            <text x={cx} y={cy + 8} textAnchor="middle" fill="#555" fontSize="8.5" fontWeight="600">{METHOD_CLUSTERS[hovered].label.split(" ")[0]}</text>
          </>
        ) : (
          <>
            <text x={cx} y={cy - 6} textAnchor="middle" fill="#333" fontSize="18" fontWeight="800">{total}</text>
            <text x={cx} y={cy + 10} textAnchor="middle" fill="#888" fontSize="8" fontWeight="600">PROJECTS</text>
          </>
        )}
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", justifyContent: "center", maxWidth: "380px" }}>
        {METHOD_CLUSTERS.map((c, i) => (
          <div
            key={i}
            style={{
              display: "flex", alignItems: "center", gap: "5px", cursor: "pointer",
              padding: "4px 10px", borderRadius: "999px",
              background: hovered === i ? c.color + "22" : "#f8f8f8",
              border: `1px solid ${hovered === i ? c.color : "#e8e8e8"}`,
              transition: "all 0.15s",
            }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          >
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: c.color, flexShrink: 0 }} />
            <span style={{ fontSize: "11px", color: "#444", fontWeight: "600" }}>{c.label}</span>
            <span style={{ fontSize: "10px", color: c.color, fontWeight: "700" }}>{c.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── INFOGRAPHIC: Technology Trend Timeline ───────────────────────────────────

const TIMELINE = [
  { year: "2017–19", label: "Classical ML Era", items: ["LSTM/GRU demand forecasting", "XGBoost for price prediction", "Prophet for seasonal patterns"] },
  { year: "2020–21", label: "Deep Learning Wave", items: ["Transformer for energy forecasting", "GNN for power grid topology", "RL environments (Grid2Op, CityLearn)"] },
  { year: "2022–23", label: "Foundation Model Rise", items: ["LLM-guided optimisation", "RAG for energy documents", "Agentic AI frameworks (LangGraph)"] },
  { year: "2024–25", label: "Agentic & Hybrid", items: ["Multi-agent grid control", "Time-series foundation models", "Physics-informed LLM agents"] },
];

function Timeline() {
  return (
    <div style={{ position: "relative", padding: "8px 0" }}>
      <div style={{ position: "absolute", left: "88px", top: 0, bottom: 0, width: "2px", background: "linear-gradient(to bottom, #6366f1, #10b981)" }} />
      {TIMELINE.map((t, i) => (
        <div key={i} style={{ display: "flex", gap: "24px", marginBottom: "28px", alignItems: "flex-start" }}>
          <div style={{ textAlign: "right", width: "80px", flexShrink: 0 }}>
            <span style={{ fontSize: "11px", fontWeight: "800", color: "#6366f1", fontFamily: "'JetBrains Mono', monospace" }}>{t.year}</span>
          </div>
          <div style={{
            width: "12px", height: "12px", borderRadius: "50%", border: "2px solid #6366f1",
            background: "#fff", flexShrink: 0, marginTop: "2px", position: "relative", zIndex: 1
          }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: "13px", fontWeight: "700", color: "#1a1a2e", marginBottom: "6px" }}>{t.label}</div>
            {t.items.map((item, j) => (
              <div key={j} style={{ fontSize: "12px", color: "#666", padding: "2px 0", display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "4px", height: "4px", borderRadius: "50%", background: "#10b981", flexShrink: 0 }} />
                {item}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── INFOGRAPHIC: Method-Problem Matrix ─────────────────────────────────────

const MATRIX_ROWS = [
  { problem: "Demand Forecasting", methods: { "Deep TS": 5, "Classical": 4, "RL": 2, "GNN": 2, "RAG": 1 } },
  { problem: "Grid Optimisation", methods: { "Deep TS": 1, "Classical": 5, "RL": 4, "GNN": 3, "RAG": 1 } },
  { problem: "Fault Detection", methods: { "Deep TS": 4, "Classical": 3, "RL": 1, "GNN": 2, "RAG": 1 } },
  { problem: "Energy Trading", methods: { "Deep TS": 3, "Classical": 2, "RL": 5, "GNN": 2, "RAG": 2 } },
  { problem: "Battery Mgmt", methods: { "Deep TS": 4, "Classical": 2, "RL": 3, "GNN": 1, "RAG": 1 } },
  { problem: "Renewable Gen.", methods: { "Deep TS": 4, "Classical": 3, "RL": 2, "GNN": 2, "RAG": 1 } },
];
const MATRIX_COLS = ["Deep TS", "Classical", "RL", "GNN", "RAG"];
const HEAT_COLORS = ["#f8fafc", "#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"];

function HeatMatrix() {
  const [cell, setCell] = useState(null);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "separate", borderSpacing: "3px", margin: "0 auto" }}>
        <thead>
          <tr>
            <th style={{ fontSize: "11px", color: "#aaa", fontWeight: "600", textAlign: "left", paddingRight: "12px", paddingBottom: "8px" }}>Problem \ Method</th>
            {MATRIX_COLS.map(c => (
              <th key={c} style={{ fontSize: "11px", color: "#666", fontWeight: "700", textAlign: "center", paddingBottom: "8px", minWidth: "72px" }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MATRIX_ROWS.map((row, ri) => (
            <tr key={ri}>
              <td style={{ fontSize: "11px", color: "#444", fontWeight: "600", paddingRight: "12px", whiteSpace: "nowrap" }}>{row.problem}</td>
              {MATRIX_COLS.map((col, ci) => {
                const val = row.methods[col] || 0;
                const bg = HEAT_COLORS[val];
                const isHot = cell && cell[0] === ri && cell[1] === ci;
                return (
                  <td
                    key={ci}
                    onMouseEnter={() => setCell([ri, ci])}
                    onMouseLeave={() => setCell(null)}
                    style={{
                      width: "72px", height: "36px", borderRadius: "6px",
                      background: isHot ? "#6366f1" : bg,
                      textAlign: "center", fontSize: "13px", fontWeight: "800",
                      color: val >= 4 || isHot ? "#fff" : val >= 2 ? "#1d4ed8" : "#aaa",
                      cursor: "default", transition: "all 0.15s",
                      border: isHot ? "2px solid #6366f1" : "2px solid transparent",
                    }}
                  >
                    {val > 0 ? val : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ textAlign: "center", marginTop: "10px", fontSize: "10px", color: "#aaa" }}>
        Score = suitability (1=low, 5=optimal). Hover to highlight.
      </div>
    </div>
  );
}

// ─── PROJECT TABLE ───────────────────────────────────────────────────────────

function ProjectTable({ projects, columns = ["name", "method", "stars", "desc", "url"] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid #f0f0f0" }}>
            {columns.includes("name") && <th style={thStyle}>Project</th>}
            {columns.includes("method") && <th style={thStyle}>Method</th>}
            {columns.includes("framework") && <th style={thStyle}>Framework</th>}
            {columns.includes("industry") && <th style={thStyle}>Domain</th>}
            {columns.includes("stars") && <th style={{ ...thStyle, textAlign: "center" }}>Stars</th>}
            {columns.includes("desc") && <th style={thStyle}>Description</th>}
            {columns.includes("url") && <th style={{ ...thStyle, textAlign: "center" }}>Repository</th>}
          </tr>
        </thead>
        <tbody>
          {projects.map((p, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #f5f5f5" }}
              onMouseEnter={e => e.currentTarget.style.background = "#fafafa"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >
              {columns.includes("name") && (
                <td style={tdStyle}>
                  <span style={{ fontWeight: "700", color: "#1a1a2e", fontSize: "13px" }}>{p.name}</span>
                </td>
              )}
              {columns.includes("method") && (
                <td style={tdStyle}>
                  <Tag label={p.method} color="#6366f1" />
                </td>
              )}
              {columns.includes("framework") && (
                <td style={tdStyle}>
                  <Tag label={p.framework} color="#10b981" />
                </td>
              )}
              {columns.includes("industry") && (
                <td style={tdStyle}>
                  <span style={{ fontSize: "12px", color: "#666" }}>{p.industry}</span>
                </td>
              )}
              {columns.includes("stars") && (
                <td style={{ ...tdStyle, textAlign: "center" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", color: "#f59e0b", fontWeight: "700", fontSize: "12px" }}>
                    <StarIcon size={10} />{p.stars}
                  </span>
                </td>
              )}
              {columns.includes("desc") && (
                <td style={{ ...tdStyle, maxWidth: "280px", color: "#666" }}>{p.desc}</td>
              )}
              {columns.includes("url") && (
                <td style={{ ...tdStyle, textAlign: "center" }}>
                  <RepoLink url={p.url} name={p.name} />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle = { padding: "8px 12px", textAlign: "left", fontSize: "11px", color: "#999", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.5px" };
const tdStyle = { padding: "10px 12px", verticalAlign: "middle" };

// ─── SECTION ─────────────────────────────────────────────────────────────────

function Section({ id, title, subtitle, children, active, onClick }) {
  return (
    <div id={id} style={{ marginBottom: "4px" }}>
      <button
        onClick={onClick}
        style={{
          width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "14px 20px", background: active ? "#f5f3ff" : "#fafafa",
          border: "1px solid", borderColor: active ? "#c4b5fd" : "#ebebeb",
          borderRadius: "10px", cursor: "pointer", textAlign: "left",
          transition: "all 0.2s",
        }}
      >
        <div>
          <span style={{ fontWeight: "700", fontSize: "14px", color: active ? "#4338ca" : "#1a1a2e" }}>{title}</span>
          {subtitle && <span style={{ fontSize: "12px", color: "#999", marginLeft: "10px" }}>{subtitle}</span>}
        </div>
        <span style={{ transform: active ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s", color: "#6366f1" }}>
          <ChevronDown />
        </span>
      </button>
      {active && (
        <div style={{ padding: "20px", background: "#fff", border: "1px solid #ebebeb", borderTop: "none", borderRadius: "0 0 10px 10px", marginTop: "-4px" }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ─── NAV ─────────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { id: "abstract", label: "Abstract" },
  { id: "clusters", label: "Clusters" },
  { id: "energy", label: "Energy" },
  { id: "agentic", label: "Agentic AI" },
  { id: "rag", label: "RAG" },
  { id: "opt", label: "Optimisation" },
  { id: "rl", label: "RL" },
  { id: "gnn", label: "Graph AI" },
  { id: "forecast", label: "Forecasting" },
  { id: "infographics", label: "Infographics" },
];

// ─── STAR HISTORY ────────────────────────────────────────────────────────────

function StarHistorySection() {
  const repos = [
    "nilmtk/nilmtk",
    "rte-france/Grid2Op",
    "PyPSA/PyPSA",
    "pybamm-team/PyBaMM",
    "unit8co/darts",
  ];
  const repoParam = repos.join(",");
  const darkSrc = `https://api.star-history.com/svg?repos=${repoParam}&type=Date&theme=dark`;
  const lightSrc = `https://api.star-history.com/svg?repos=${repoParam}&type=Date`;

  return (
    <div style={{ padding: "24px", background: "#fafafa", borderRadius: "12px", border: "1px solid #ebebeb" }}>
      <div style={{ marginBottom: "16px" }}>
        <h3 style={{ margin: 0, fontSize: "15px", fontWeight: "700", color: "#1a1a2e" }}>Star History — Featured Energy Repos</h3>
        <p style={{ margin: "4px 0 0", fontSize: "12px", color: "#999" }}>Growth trajectory of top energy AI projects on GitHub</p>
      </div>
      <picture>
        <source media="(prefers-color-scheme: dark)" srcSet={darkSrc} />
        <source media="(prefers-color-scheme: light)" srcSet={lightSrc} />
        <img
          alt="Star History Chart for featured energy repositories"
          src={lightSrc}
          style={{ width: "100%", borderRadius: "8px", border: "1px solid #e8e8e8" }}
        />
      </picture>
      <div style={{ marginTop: "12px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {repos.map(r => (
          <a
            key={r}
            href={`https://github.com/${r}`}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: "11px", color: "#6366f1", fontFamily: "'JetBrains Mono', monospace",
              textDecoration: "none", padding: "3px 8px", borderRadius: "4px",
              background: "#f5f3ff", border: "1px solid #e0e0ff",
            }}
          >
            {r}
          </a>
        ))}
      </div>
    </div>
  );
}

// ─── ENERGY GROUPED TABLE ────────────────────────────────────────────────────

function EnergySection() {
  const [openCat, setOpenCat] = useState("Demand Forecasting");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {ENERGY_PROJECTS.map(group => (
        <div key={group.category}>
          <button
            onClick={() => setOpenCat(openCat === group.category ? null : group.category)}
            style={{
              width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "10px 14px", background: openCat === group.category ? "#f0fdf4" : "#f9f9f9",
              border: `1px solid ${openCat === group.category ? "#a7f3d0" : "#e8e8e8"}`,
              borderRadius: "8px", cursor: "pointer", transition: "all 0.15s",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span style={{ fontWeight: "700", fontSize: "13px", color: openCat === group.category ? "#059669" : "#333" }}>{group.category}</span>
              <span style={{ fontSize: "11px", color: "#10b981", fontWeight: "600", background: "#d1fae5", padding: "1px 7px", borderRadius: "999px" }}>
                {group.projects.length} repos
              </span>
            </div>
            <span style={{ transform: openCat === group.category ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s", color: "#10b981" }}>
              <ChevronDown size={12} />
            </span>
          </button>
          {openCat === group.category && (
            <div style={{ border: "1px solid #d1fae5", borderTop: "none", borderRadius: "0 0 8px 8px", overflow: "hidden" }}>
              <ProjectTable projects={group.projects} columns={["name", "method", "stars", "desc", "url"]} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── MAIN APP ────────────────────────────────────────────────────────────────

export default function App() {
  const [activeSection, setActiveSection] = useState("abstract");
  const [activeNav, setActiveNav] = useState("abstract");

  const toggle = (id) => setActiveSection(s => s === id ? null : id);

  return (
    <div style={{
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      background: "#f8f9fc",
      minHeight: "100vh",
      color: "#1a1a2e",
    }}>
      {/* HEADER */}
      <div style={{
        background: "linear-gradient(135deg, #0f0e17 0%, #1e1b4b 40%, #0f172a 100%)",
        padding: "48px 32px 40px",
        position: "relative",
        overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
          backgroundImage: "radial-gradient(circle at 20% 50%, rgba(99,102,241,0.15) 0%, transparent 60%), radial-gradient(circle at 80% 20%, rgba(16,185,129,0.1) 0%, transparent 50%)",
        }} />
        <div style={{ position: "relative", maxWidth: "900px", margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
            <span style={{ fontSize: "11px", fontWeight: "700", color: "#818cf8", letterSpacing: "2px", textTransform: "uppercase" }}>Harrypatria / GitHub</span>
            <span style={{ width: "1px", height: "14px", background: "#334155" }} />
            <a
              href="https://github.com/Harrypatria"
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: "11px", color: "#64748b", textDecoration: "none", fontFamily: "'JetBrains Mono', monospace" }}
            >
              github.com/Harrypatria
            </a>
          </div>
          <h1 style={{
            margin: 0, fontSize: "clamp(24px, 4vw, 38px)", fontWeight: "800",
            color: "#fff", letterSpacing: "-1px", lineHeight: 1.1,
          }}>
            500+ AI · ML · Optimisation
            <span style={{ display: "block", background: "linear-gradient(90deg, #6366f1, #10b981)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Energy Intelligence Projects
            </span>
          </h1>
          <p style={{ marginTop: "16px", color: "#94a3b8", fontSize: "14px", maxWidth: "600px", lineHeight: 1.6 }}>
            A method-first curated collection of production-grade open-source repositories for energy AI, grid optimisation, reinforcement learning, RAG, and graph analytics.
          </p>

          {/* BADGE ROW */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", marginTop: "24px" }}>
            {[
              { label: "MIT License", url: null, bg: "#1e293b", color: "#94a3b8" },
              { label: "500+ Projects", url: null, bg: "#312e81", color: "#a5b4fc" },
              { label: "PRs Welcome", url: null, bg: "#064e3b", color: "#6ee7b7" },
              { label: "Energy Focus", url: null, bg: "#7c2d12", color: "#fca5a5" },
              { label: "github.com/Harrypatria", url: "https://github.com/Harrypatria", bg: "#1e293b", color: "#e2e8f0" },
            ].map((b, i) => (
              <span key={i} style={{
                display: "inline-flex", alignItems: "center", gap: "5px",
                padding: "5px 12px", borderRadius: "6px", background: b.bg,
                fontSize: "11px", fontWeight: "600", color: b.color,
                fontFamily: "'JetBrains Mono', monospace",
                border: "1px solid rgba(255,255,255,0.05)",
                cursor: b.url ? "pointer" : "default",
                textDecoration: "none",
              }} as={b.url ? "a" : "span"} onClick={b.url ? () => window.open(b.url, "_blank") : undefined}>
                {b.label}
              </span>
            ))}
          </div>

          {/* STAT BADGES */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "16px", marginTop: "28px" }}>
            {BADGES.map((b, i) => (
              <div key={i} style={{
                display: "flex", flexDirection: "column", alignItems: "flex-start",
                padding: "12px 16px", background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)", borderRadius: "10px",
              }}>
                <span style={{ fontSize: "20px", fontWeight: "800", color: b.color }}>{b.value}</span>
                <span style={{ fontSize: "10px", color: "#64748b", marginTop: "2px", textTransform: "uppercase", letterSpacing: "0.5px" }}>{b.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* NAV */}
      <div style={{
        position: "sticky", top: 0, zIndex: 50,
        background: "rgba(255,255,255,0.95)", backdropFilter: "blur(8px)",
        borderBottom: "1px solid #ebebeb", padding: "0 32px",
      }}>
        <div style={{ maxWidth: "900px", margin: "0 auto", display: "flex", gap: "2px", overflowX: "auto" }}>
          {NAV_ITEMS.map(n => (
            <button
              key={n.id}
              onClick={() => {
                setActiveNav(n.id);
                setActiveSection(n.id);
                document.getElementById(n.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
              style={{
                padding: "12px 14px", background: "none", border: "none",
                borderBottom: `2px solid ${activeNav === n.id ? "#6366f1" : "transparent"}`,
                fontSize: "12px", fontWeight: "600", color: activeNav === n.id ? "#6366f1" : "#888",
                cursor: "pointer", transition: "all 0.15s", whiteSpace: "nowrap",
              }}
            >
              {n.label}
            </button>
          ))}
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div style={{ maxWidth: "900px", margin: "0 auto", padding: "32px 24px" }}>

        {/* ABSTRACT */}
        <div id="abstract" style={{ marginBottom: "28px" }}>
          <div style={{ padding: "28px", background: "#fff", borderRadius: "12px", border: "1px solid #ebebeb" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
              <span style={{ width: "3px", height: "20px", background: "linear-gradient(to bottom, #6366f1, #10b981)", borderRadius: "2px" }} />
              <h2 style={{ margin: 0, fontSize: "15px", fontWeight: "700", color: "#1a1a2e" }}>Repository Abstract</h2>
            </div>
            <p style={{ margin: 0, fontSize: "13.5px", color: "#444", lineHeight: 1.85, fontStyle: "normal" }}>
              {ABSTRACT}
            </p>
          </div>
        </div>

        {/* CLUSTERS */}
        <div id="clusters" style={{ marginBottom: "28px" }}>
          <div style={{ padding: "28px", background: "#fff", borderRadius: "12px", border: "1px solid #ebebeb" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
              <span style={{ width: "3px", height: "20px", background: "linear-gradient(to bottom, #6366f1, #10b981)", borderRadius: "2px" }} />
              <h2 style={{ margin: 0, fontSize: "15px", fontWeight: "700" }}>Method Clusters</h2>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "12px" }}>
              {METHOD_CLUSTERS.map((c, i) => (
                <div key={i} style={{
                  padding: "16px", borderRadius: "10px",
                  border: `1px solid ${c.color}30`,
                  background: `${c.color}08`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <span style={{ fontSize: "13px", fontWeight: "700", color: "#1a1a2e" }}>{c.label}</span>
                    <span style={{ fontSize: "18px", fontWeight: "800", color: c.color }}>{c.count}</span>
                  </div>
                  <p style={{ margin: "6px 0 0", fontSize: "11px", color: "#888", lineHeight: 1.5 }}>{c.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ACCORDION SECTIONS */}
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginBottom: "28px" }}>

          <div id="energy">
            <Section
              id="energy-inner"
              title="Energy Intelligence Hub"
              subtitle="80+ repos — grid, renewables, battery, fault detection, trading"
              active={activeSection === "energy"}
              onClick={() => toggle("energy")}
            >
              <EnergySection />
            </Section>
          </div>

          <div id="agentic">
            <Section
              id="agentic-inner"
              title="Agentic AI & Multi-Agent Systems"
              subtitle="CrewAI · AutoGen · LangGraph · Agno · Standalone agents"
              active={activeSection === "agentic"}
              onClick={() => toggle("agentic")}
            >
              <ProjectTable
                projects={AGENTIC_PROJECTS}
                columns={["name", "framework", "industry", "method", "stars", "url"]}
              />
            </Section>
          </div>

          <div id="rag">
            <Section
              id="rag-inner"
              title="RAG & Retrieval-Augmented Generation"
              subtitle="Adaptive · Corrective · Self-RAG · GraphRAG · Chatbots"
              active={activeSection === "rag"}
              onClick={() => toggle("rag")}
            >
              <ProjectTable
                projects={RAG_PROJECTS}
                columns={["name", "method", "stars", "desc", "url"]}
              />
            </Section>
          </div>

          <div id="opt">
            <Section
              id="opt-inner"
              title="Optimisation & Operations Research"
              subtitle="LP · MIP · Metaheuristics · Scheduling · LLM-guided OR"
              active={activeSection === "opt"}
              onClick={() => toggle("opt")}
            >
              <ProjectTable
                projects={OPTIMISATION_PROJECTS}
                columns={["name", "method", "stars", "desc", "url"]}
              />
            </Section>
          </div>

          <div id="rl">
            <Section
              id="rl-inner"
              title="Reinforcement Learning"
              subtitle="PPO · SAC · DQN · MARL · Grid control · EV charging"
              active={activeSection === "rl"}
              onClick={() => toggle("rl")}
            >
              <ProjectTable
                projects={RL_PROJECTS}
                columns={["name", "method", "stars", "desc", "url"]}
              />
            </Section>
          </div>

          <div id="gnn">
            <Section
              id="gnn-inner"
              title="Graph & Network AI"
              subtitle="GNN · Knowledge graphs · Community detection · Power topology"
              active={activeSection === "gnn"}
              onClick={() => toggle("gnn")}
            >
              <ProjectTable
                projects={GNN_PROJECTS}
                columns={["name", "method", "stars", "desc", "url"]}
              />
            </Section>
          </div>

          <div id="forecast">
            <Section
              id="forecast-inner"
              title="Forecasting & Time-Series Modelling"
              subtitle="Foundation models · Transformers · LSTM · Anomaly detection"
              active={activeSection === "forecast"}
              onClick={() => toggle("forecast")}
            >
              <ProjectTable
                projects={FORECAST_PROJECTS}
                columns={["name", "method", "stars", "desc", "url"]}
              />
            </Section>
          </div>

        </div>

        {/* INFOGRAPHICS */}
        <div id="infographics" style={{ marginBottom: "28px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
            <span style={{ width: "3px", height: "20px", background: "linear-gradient(to bottom, #6366f1, #10b981)", borderRadius: "2px" }} />
            <h2 style={{ margin: 0, fontSize: "15px", fontWeight: "700" }}>Visual Infographics</h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px" }}>

            {/* Donut chart */}
            <div style={{ padding: "24px", background: "#fff", borderRadius: "12px", border: "1px solid #ebebeb" }}>
              <h3 style={{ margin: "0 0 16px", fontSize: "13px", fontWeight: "700", color: "#1a1a2e" }}>Project Distribution by Method</h3>
              <ClusterChart />
            </div>

            {/* Timeline */}
            <div style={{ padding: "24px", background: "#fff", borderRadius: "12px", border: "1px solid #ebebeb" }}>
              <h3 style={{ margin: "0 0 20px", fontSize: "13px", fontWeight: "700", color: "#1a1a2e" }}>Energy AI — Technology Timeline</h3>
              <Timeline />
            </div>

            {/* Heat matrix — full width */}
            <div style={{ gridColumn: "1 / -1", padding: "24px", background: "#fff", borderRadius: "12px", border: "1px solid #ebebeb" }}>
              <h3 style={{ margin: "0 0 4px", fontSize: "13px", fontWeight: "700", color: "#1a1a2e" }}>Method–Problem Fit Matrix</h3>
              <p style={{ margin: "0 0 16px", fontSize: "12px", color: "#999" }}>
                Suitability score (1–5) of each AI method across energy use cases.
              </p>
              <HeatMatrix />
            </div>
          </div>
        </div>

        {/* STAR HISTORY */}
        <div style={{ marginBottom: "28px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
            <span style={{ width: "3px", height: "20px", background: "linear-gradient(to bottom, #f59e0b, #ef4444)", borderRadius: "2px" }} />
            <h2 style={{ margin: 0, fontSize: "15px", fontWeight: "700" }}>Star History</h2>
          </div>
          <StarHistorySection />
        </div>

        {/* FOOTER */}
        <div style={{
          padding: "28px", background: "#fff", borderRadius: "12px", border: "1px solid #ebebeb",
          display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "flex-start", gap: "20px",
        }}>
          <div>
            <h3 style={{ margin: "0 0 6px", fontSize: "14px", fontWeight: "700" }}>Harry Patria</h3>
            <p style={{ margin: 0, fontSize: "12px", color: "#888" }}>Data & AI Lead · Researcher · Entrepreneur</p>
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "#aaa" }}>Patria & Co. ~ PT. Strategi Transforma Infiniti</p>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
            {[
              { label: "GitHub", url: "https://github.com/Harrypatria" },
              { label: "patriaco.co.uk", url: "https://www.patriaco.co.uk" },
              { label: "LinkedIn", url: "https://www.linkedin.com/in/harry-patria/" },
              { label: "ORCID", url: "https://orcid.org/0000-0002-7844-538X" },
              { label: "PuLP Optimisation", url: "https://github.com/Harrypatria/Basic-to-Advanced-Optimization-Tutorial-with-PuLP-Python" },
              { label: "NetworkX Tutorial", url: "https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX" },
              { label: "Python MasterClass", url: "https://github.com/Harrypatria/Python_MasterClass" },
            ].map((l, i) => (
              <a
                key={i}
                href={l.url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  fontSize: "11px", color: "#6366f1", textDecoration: "none",
                  padding: "4px 10px", borderRadius: "6px", background: "#f5f3ff",
                  border: "1px solid #e0e0ff", fontFamily: "'JetBrains Mono', monospace",
                  transition: "all 0.15s",
                }}
              >
                {l.label}
              </a>
            ))}
          </div>
        </div>

        <div style={{ textAlign: "center", marginTop: "24px", fontSize: "11px", color: "#ccc" }}>
          MIT License · Star this repo if it helped you · PRs welcome
        </div>
      </div>
    </div>
  );
}
