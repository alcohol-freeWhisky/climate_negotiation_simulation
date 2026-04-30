┌─────────────────────────────────────────────────────────┐
│                    Simulation Controller                 │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ Config Mgr   │ │ Phase Engine │ │ Evaluation Engine │  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Negotiation Engine                    │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐             │
│  │ Text Mgr  │ │ Turn Mgr   │ │ Amendment   │             │
│  │ (draft    │ │ (speaking  │ │ Processor   │             │
│  │  tracking)│ │  order,    │ │ (add/del/   │             │
│  │           │ │  timing)   │ │  modify)    │             │
│  └──────────┘ └───────────┘ └────────────┘             │
├─────────────────────────────────────────────────────────┤
│                    Agent Layer                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │
│  │ EU     │ │ G77    │ │ AOSIS  │ │ Umbr.  │ ...      │
│  │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │          │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘          │
│      │          │          │          │                 │
│  ┌───┴──────────┴──────────┴──────────┴──────────────┐ │
│  │              LLM Backend (GPT-4 / Claude)          │ │
│  └────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                            │
│  ┌────────────┐ ┌───────────┐ ┌─────────────────────┐  │
│  │ Agent      │ │ Document  │ │ Negotiation History │  │
│  │ Profiles   │ │ Store     │ │ & Memory            │  │
│  └────────────┘ └───────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────┘

Phase 0: Initialization
  ├── Load draft text (bracketed negotiating text)
  ├── Initialize agents with profiles
  └── Set agenda and rules

Phase 1: Opening Statements (1 round)
  ├── Each agent states general position
  ├── Identifies key priorities and red lines
  └── Output: Initial positions recorded

Phase 2: First Reading (paragraph-by-paragraph)
  ├── For each paragraph/section:
  │   ├── Chair presents text
  │   ├── Agents propose amendments (add/modify/delete)
  │   └── All proposals collected into "compiled text" with brackets
  └── Output: Expanded bracketed text

Phase 3: Informal Consultations (multiple rounds)
  ├── Focus on bracketed/disputed text
  ├── Agents argue for their positions
  ├── Chair proposes compromise language
  ├── Agents accept/reject/counter-propose
  └── Repeat until convergence or max rounds

Phase 4: Final Plenary
  ├── Chair presents "clean" text
  ├── Agents give final objections or acceptance
  ├── Consensus check
  └── Output: Final agreed text (or failure)