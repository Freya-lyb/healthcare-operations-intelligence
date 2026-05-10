# Demand–Supply Framework

```mermaid
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
```

## Framework Summary

This project framed avoidable emergency department utilization as a systems coordination problem rather than a patient behavior problem.

The analysis identified both demand-side and supply-side structural failures contributing to emergency department overcrowding.

### Demand-Side Drivers

- Temporal healthcare access gaps
- Lack of PCP relationships
- Social resource scarcity and SDOH barriers

### Supply-Side Constraints

- PCP overload
- Specialist workload imbalance
- Limited outpatient absorption capacity

### Operational Goal

The project focused on identifying scalable operational interventions capable of reducing avoidable ED burden using existing healthcare system resources rather than major policy or staffing expansion.
