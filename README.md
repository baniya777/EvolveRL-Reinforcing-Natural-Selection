# EvolveRL — Reinforcing Natural Selection

A multi-agent reinforcement learning simulation where predator and prey agents independently learn survival strategies through Q-learning on a hexagonal grid. The project validates whether artificially learned population dynamics match the classical Lotka-Volterra ecological model.

Developed as part of undergraduate research in Computational Mathematics at Kathmandu University.

---

## Overview

EvolveRL simulates an ecosystem on a 79×39 hexagonal grid (3,081 cells). Two types of agents — prey and predators — learn through trial and error without being explicitly programmed with hunting or fleeing behaviors. Over hundreds of training episodes, they develop strategies that mirror natural selection: prey learning to conserve energy and avoid danger, predators learning to hunt efficiently.

### Key Results

| Metric | Random Baseline | Trained Agents | Improvement |
|--------|----------------|----------------|-------------|
| Prey Survival | 3.1 agents | 8.5 agents | **+174%** |
| Predator Survival | 2.4 agents | 6.2 agents | **+158%** |
| Episode Length | 156 steps | 387 steps | **+148%** |

- Population dynamics showed **86% correlation** with Lotka-Volterra theoretical predictions
- **Stable oscillations** emerge without extinction after training
- Agents converge by episode **300–400**

---

## Project Structure

```
EvolveRL/
├── hexagon.py              # Hexagonal grid tile system (rendering)
├── rl_environment.py       # PettingZoo multi-agent environment
├── q_learning.py           # Q-learning agent implementation
├── train_rl.py             # Training script with visualization
├── demo_agents.py          # Terminal demo + performance comparison
├── visual_demo.py          # Full pygame hexagonal visual simulation
├── test_implementation.py  # Test suite — run this first
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Installation

**Requirements:** Python 3.8+

```bash
git clone https://github.com/baniya777/EvolveRL-Reinforcing-Natural-Selection.git
cd EvolveRL-Reinforcing-Natural-Selection
pip install -r requirements.txt
```

---

## Quick Start

### 1. Verify installation

```bash
python test_implementation.py
```

This runs a suite of checks on imports, environment creation, Q-learning, and a short 10-episode training run. Fix any errors shown before proceeding.

### 2. Train the agents

```bash
python train_rl.py
```

Runs 500 training episodes (~15–30 minutes). Outputs:
- `training_results.png` — learning curves and population dynamics
- `lotka_volterra_comparison.png` — RL results vs theoretical model
- `q_tables/` — saved Q-tables for all agents

### 3. Watch the visual simulation

```bash
python visual_demo.py
```

Opens a pygame window showing the full hexagonal grid with live agents and a stats panel. Works with or without trained Q-tables (falls back to random agents if none found).

**Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `1` | 1× speed |
| `2` | 2× speed |
| `4` | 4× speed |
| `ESC` | Quit |

### 4. Terminal demo and comparison

```bash
python demo_agents.py           # Run trained agents, print stats to terminal
python demo_agents.py compare   # Compare trained vs random over 20 episodes
```

---

## How It Works

### Environment

The environment is built on the [PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv` framework. The hexagonal grid allows 6-directional movement, producing more natural and isotropic agent behavior than a square grid.

**Mechanics:**
- Agents gain and lose energy each step
- Prey reproduce when energy reaches 100, splitting energy with offspring
- Predators die if energy reaches 0
- A predator occupying the same cell as a prey eats it immediately

### State Representation

**Prey — 36 dimensions:**
```
[energy_level, vision_cell_1, ..., vision_cell_35]
```
- Energy: normalized 0–1
- Vision: 3-hexagon radius around the agent; each cell encodes `0` = empty, `1` = prey, `2` = predator

**Predator — 15 dimensions:**
```
[energy_level, vision_cone_1, ..., vision_cone_9, direction_one_hot(6)]
```
- Energy: normalized 0–1
- Vision: directional cone of 9 cells forward
- Direction: one-hot encoded across 6 hexagonal directions

### Action Space

**Prey (7 actions):**
- `0–5`: Move in 6 hexagonal directions (costs −0.1 reward)
- `6`: Stay still (+15 energy, +1 reward)

**Predator (9 actions):**
- `0–5`: Move in 6 directions
- `6`: Stay
- `7`: Rotate vision cone left
- `8`: Rotate vision cone right

### Reward Structure

| Event | Prey | Predator |
|-------|------|----------|
| Stay still | +1 | — |
| Move | −0.1 | — |
| Eat prey | — | +20 |
| Reproduce | +5 | +5 |
| Be eaten | −10 | — |
| Starve | — | −20 |
| Per timestep | — | −1 energy |

### Learning Algorithm

Each agent independently runs tabular Q-learning (Independent Q-Learning). There is no shared policy or communication between agents.

**Bellman update:**
```
Q(s,a) ← Q(s,a) + α [ R + γ · max_a' Q(s',a') − Q(s,a) ]
```

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.95 |
| Epsilon start | 1.0 |
| Epsilon end | 0.01 |
| Epsilon decay | 0.995 per episode |

States are discretized into energy bins and vision counts to keep Q-tables tractable (~1,000–5,000 states per agent).

---

## Results

### Emergent Behaviors

After training, agents develop biologically plausible strategies without being explicitly programmed:

**Prey learned to:**
- Stay in safe zones and accumulate energy when no predators are visible
- Move away when predators enter the vision radius
- Reproduce at near-optimal energy levels

**Predators learned to:**
- Rotate their vision cone toward areas with prey
- Chase prey rather than wandering randomly
- Balance the trade-off between hunting and energy conservation

### Population Dynamics

The trained system produces cyclical oscillations characteristic of real predator-prey ecosystems — prey populations peak when predators are scarce, predator populations lag prey peaks by approximately 100 timesteps, and neither population goes extinct. These dynamics show 86% correlation with the Lotka-Volterra theoretical model.

---

## Technical Details

- **Framework:** PettingZoo `ParallelEnv`
- **Grid:** 79×39 hexagonal cells with toroidal wrapping
- **Max episode length:** 1,000 steps
- **Visualization:** pygame 2.5+ with `HexagonTile` rendering from `hexagon.py`
- **Convergence:** Typically episode 300–400

---

## Challenges

**State space design** — Balancing observation detail against learnability required iteration. Too many dimensions slowed convergence; too few left agents effectively blind.

**Reward engineering** — Early reward structures produced degenerate behaviors (prey never moving, predators ignoring prey entirely). Multiple redesigns were needed to produce realistic emergent behavior.

**Population collapse** — Early versions suffered frequent extinctions. Tuning the energy parameters `ALPHA` (prey gain), `DELTA` (predation gain), and `GAMMA` (predator decay) was necessary to maintain stable long-run populations.

---

## Future Improvements

- Deep Q-Networks (DQN) to handle richer continuous state spaces
- Agent communication for coordinated swarm behavior
- Environmental heterogeneity (obstacles, resource patches, terrain)
- Validation against real ecological time-series data (e.g., lynx–hare populations)

---

## Dependencies

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pygame>=2.5.0
pettingzoo>=1.24.0
gymnasium>=0.29.0
tqdm>=4.65.0
```

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Lotka, A. J. (1925). *Elements of Physical Biology*. Williams & Wilkins.
- Volterra, V. (1926). Fluctuations in the abundance of a species considered mathematically. *Nature*, 118, 558–560.
- PettingZoo documentation: https://pettingzoo.farama.org/

---

## Acknowledgments

Developed as part of a B.Sc. in Computational Mathematics at Kathmandu University under the supervision of Mr. Harish Chandra Bhandari. Thanks to project team members for collaboration on the initial environment design.

---

## Author

**Supriya Baniya**  
