flowchart TD

    subgraph Demand Side Drivers
        A1[Temporal Lockout\nWeekend & Evening Clinic Gaps]
        A2[Lack of PCP Access]
        A3[Social Resource Scarcity\nSDOH Barriers]
    end

    subgraph Core Operational Problem
        B[Avoidable Emergency Department Utilization]
    end

    subgraph Supply Side Constraints
        C1[PCP Overload]
        C2[Specialist Workforce Imbalance]
        C3[Limited Outpatient Capacity]
    end

    subgraph Operational Interventions
        D1[Dynamic Clinic Hours]
        D2[Upgraded Triage Routing]
        D3[MyChart & Resource Navigation]
        D4[Specialist Rebalancing]
    end

    A1 --> B
    A2 --> B
    A3 --> B

    C1 --> B
    C2 --> B
    C3 --> B

    B --> D1
    B --> D2
    B --> D3
    B --> D4
