"""Task definitions for multi-agent collaboration experiments.

Defines two task groups with correlated tasks to demonstrate:
- Group 1: Research and analysis (cross-task memory reuse)
- Group 2: Code security analysis (pattern reuse from shared memory)
"""

from typing import Any, Dict, List

# ============================================================
# Task Group 1: Renewable Energy Research
# Correlated tasks that build on each other's findings
# ============================================================

TASK_GROUP_1 = [
    {
        "task_id": "energy_task_1",
        "description": "Research and summarize solar energy technologies including photovoltaic types, efficiency metrics, costs, and environmental impact. Create a comprehensive report.",
        "tags": ["solar", "renewable", "energy", "photovoltaic", "research"],
        "expected_findings": [
            "Monocrystalline and polycrystalline silicon panels",
            "15-22% efficiency range",
            "Cost reduction: 90% since 2010",
            "Carbon footprint: 40-50 gCO2/kWh",
            "Perovskite tandem cells as emerging technology",
        ],
    },
    {
        "task_id": "energy_task_2",
        "description": "Research wind energy technologies and compare them with solar energy. Analyze which technology is better suited for different geographic regions and use cases. Include cost comparison, capacity factors, and environmental trade-offs.",
        "tags": ["wind", "solar", "renewable", "comparison", "energy", "research"],
        "expected_findings": [
            "Wind: 25-55% capacity factor vs Solar: 15-25%",
            "Wind carbon footprint: 10-12 gCO2/kWh (lower than solar)",
            "Offshore wind growing rapidly with >15 MW turbines",
            "Solar better for distributed/decentralized deployment",
            "Both needed for complementary renewable grid",
        ],
    },
]

# ============================================================
# Task Group 2: Code Analysis and Security Audit
# Correlated tasks that share security patterns
# ============================================================

TASK_GROUP_2 = [
    {
        "task_id": "code_task_1",
        "description": "Analyze Python code for common security vulnerabilities. Search for SQL injection, command injection, hardcoded credentials, path traversal, and insecure deserialization patterns. Document detection methods and recommended fixes.",
        "tags": ["security", "code", "python", "vulnerability", "analysis"],
        "expected_findings": [
            "SQL injection via string formatting",
            "Command injection via os.system with user input",
            "Hardcoded API keys in source code",
            "Path traversal from unsanitized file paths",
            "Insecure pickle deserialization",
        ],
    },
    {
        "task_id": "code_task_2",
        "description": "Review a web application codebase for security issues, applying the vulnerability patterns identified in previous analysis. Also check for additional web-specific vulnerabilities like XSS, CSRF, and authentication bypass. Generate a security audit report with severity ratings.",
        "tags": ["security", "web", "code", "audit", "python", "analysis"],
        "expected_findings": [
            "XSS vulnerabilities in template rendering",
            "CSRF protection missing on state-changing endpoints",
            "Reuse of SQL injection detection patterns from previous analysis",
            "Authentication token validation issues",
            "Severity-based vulnerability classification",
        ],
    },
]

# ============================================================
# Task Group 3: Multi-Step Reasoning (for 10+ round testing)
# ============================================================

TASK_GROUP_3_CONTINUOUS = [
    {
        "task_id": "cont_task_1",
        "description": "Analyze the efficiency of renewable energy sources for urban deployment, considering space constraints and population density.",
        "tags": ["renewable", "urban", "efficiency", "analysis"],
    },
    {
        "task_id": "cont_task_2",
        "description": "Compare battery storage technologies (lithium-ion, flow batteries, solid-state) for grid-scale renewable energy storage.",
        "tags": ["battery", "storage", "grid", "comparison"],
    },
    {
        "task_id": "cont_task_3",
        "description": "Design a hybrid renewable energy system for a medium-sized city (500k population), combining solar, wind, and storage.",
        "tags": ["hybrid", "design", "renewable", "city-planning"],
    },
    {
        "task_id": "cont_task_4",
        "description": "Perform cost-benefit analysis of the hybrid system designed in the previous task, including installation, maintenance, and ROI projections.",
        "tags": ["cost", "benefit", "roi", "analysis"],
    },
    {
        "task_id": "cont_task_5",
        "description": "Research policy incentives and regulatory frameworks that would accelerate adoption of the proposed hybrid system.",
        "tags": ["policy", "regulation", "incentives", "research"],
    },
    {
        "task_id": "cont_task_6",
        "description": "Analyze environmental impact assessment requirements for the hybrid renewable energy project.",
        "tags": ["environmental", "impact", "assessment", "analysis"],
    },
    {
        "task_id": "cont_task_7",
        "description": "Evaluate grid integration challenges and solutions for the hybrid renewable system.",
        "tags": ["grid", "integration", "challenges", "evaluation"],
    },
    {
        "task_id": "cont_task_8",
        "description": "Compare the proposed hybrid system with conventional natural gas power plant in terms of lifecycle emissions and costs.",
        "tags": ["comparison", "lifecycle", "emissions", "costs"],
    },
    {
        "task_id": "cont_task_9",
        "description": "Research community engagement strategies for renewable energy project deployment.",
        "tags": ["community", "engagement", "deployment", "strategy"],
    },
    {
        "task_id": "cont_task_10",
        "description": "Synthesize all previous findings into a comprehensive master plan for city renewable energy transition.",
        "tags": ["synthesis", "master-plan", "transition", "comprehensive"],
    },
    {
        "task_id": "cont_task_11",
        "description": "Analyze financing options for the master plan: green bonds, power purchase agreements (PPAs), and municipal green loans. Build on prior cost-benefit and policy findings.",
        "tags": ["finance", "green-bonds", "ppa", "continuous", "energy"],
        "domain": "energy",
        "suite": "continuous",
    },
    {
        "task_id": "cont_task_12",
        "description": "Produce an executive rollout roadmap for the city renewable transition: 5-year KPIs, milestones, risk register, and dependencies on all prior continuous tasks (tasks 1-11).",
        "tags": ["roadmap", "kpi", "executive", "continuous", "energy"],
        "domain": "energy",
        "suite": "continuous",
    },
]

# All task groups
ALL_TASK_GROUPS: Dict[str, List[Dict[str, Any]]] = {
    "energy_research": TASK_GROUP_1,
    "code_security": TASK_GROUP_2,
    "continuous_10": TASK_GROUP_3_CONTINUOUS[:10],
    "continuous12": TASK_GROUP_3_CONTINUOUS,
}
