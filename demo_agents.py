"""
EvolveRL - Visual Hexagonal Grid Simulation
Full pygame rendering with hexagonal grid, colored agents, and live stats panel
"""

import sys
import os
import numpy as np
import math
import pygame

sys.path.append('/mnt/user-data/uploads')

from rl_environment import PredatorPreyEnv, list_to_axial, axial_to_list
from q_learning import MultiAgentQLearning

# ── Display config ────────────────────────────────────────────────────────────
SCREEN_W      = 1280
SCREEN_H      = 800
PANEL_W       = 260          # right-side stats panel
HEX_COLS      = 79
HEX_ROWS      = 39
HEX_RADIUS    = 10           # pixel radius of each hexagon

# ── Colours ───────────────────────────────────────────────────────────────────
BG            = (15,  20,  30)
GRID_BG       = (25,  32,  45)
HEX_EMPTY     = (35,  45,  60)
HEX_BORDER    = (50,  65,  85)
PREY_COL      = (60,  200,  80)
PREY_GLOW     = (120, 255, 140)
PRED_COL      = (220,  50,  50)
PRED_GLOW     = (255, 120, 100)
PANEL_BG      = (18,  24,  36)
PANEL_BORDER  = (60,  80, 110)
TEXT_PRIMARY  = (220, 230, 250)
TEXT_DIM      = (120, 140, 170)
PREY_CHART    = (60,  200,  80)
PRED_CHART    = (220,  50,  50)
WHITE         = (255, 255, 255)
GOLD          = (255, 200,  60)


def hex_pixel(col: int, row: int, radius: float) -> tuple:
    """
    Convert hex grid (col, row) → pixel (x, y).
    Uses offset layout matching rl_environment's axial system.
    """
    w = radius * math.sqrt(3)
    h = radius * 2
    x = col * w + (row % 2) * (w / 2)
    y = row * h * 0.75
    return (x, y)


def axial_to_grid(ax, ay):
    """Convert axial coords used by rl_environment → (col, row) for rendering."""
    col = ax
    row = ay + (ax // 2)
    return col % HEX_COLS, row % HEX_ROWS


def pointy_hex_vertices(cx: float, cy: float, r: float):
    """Return 6 vertices for a pointy-top hexagon centred at (cx, cy)."""
    pts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def draw_hex(surface, cx, cy, r, fill, border=None, border_w=1):
    verts = pointy_hex_vertices(cx, cy, r)
    pygame.draw.polygon(surface, fill, verts)
    if border:
        pygame.draw.polygon(surface, border, verts, border_w)


def draw_agent_dot(surface, cx, cy, colour, glow_colour, radius=4):
    """Draw a glowing circle for an agent."""
    pygame.draw.circle(surface, glow_colour, (int(cx), int(cy)), radius + 2)
    pygame.draw.circle(surface, colour,      (int(cx), int(cy)), radius)


def draw_text(surface, text, x, y, font, colour=TEXT_PRIMARY, align='left'):
    surf = font.render(text, True, colour)
    if align == 'right':
        x -= surf.get_width()
    elif align == 'center':
        x -= surf.get_width() // 2
    surface.blit(surf, (x, y))
    return surf.get_height()


def build_grid_surface(radius):
    """Pre-render all empty hexagons onto a surface (drawn once)."""
    w = radius * math.sqrt(3)
    h = radius * 2
    grid_w = int(HEX_COLS * w + w / 2) + 4
    grid_h = int(HEX_ROWS * h * 0.75 + h * 0.25) + 4
    surf = pygame.Surface((grid_w, grid_h))
    surf.fill(GRID_BG)
    for row in range(HEX_ROWS):
        for col in range(HEX_COLS):
            px, py = hex_pixel(col, row, radius)
            cx = px + radius * math.sqrt(3) / 2
            cy = py + radius
            draw_hex(surf, cx, cy, radius - 1, HEX_EMPTY, HEX_BORDER, 1)
    return surf, grid_w, grid_h


class PopulationChart:
    """Scrolling line chart for prey/predator counts."""
    def __init__(self, maxlen=300):
        self.maxlen  = maxlen
        self.prey    = []
        self.pred    = []

    def push(self, p, q):
        self.prey.append(p)
        self.pred.append(q)
        if len(self.prey) > self.maxlen:
            self.prey.pop(0)
            self.pred.pop(0)

    def draw(self, surface, rect, font_sm):
        x, y, w, h = rect
        pygame.draw.rect(surface, (22, 30, 45), (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

        if len(self.prey) < 2:
            return

        all_vals = self.prey + self.pred
        vmax = max(max(all_vals), 1)

        def to_pt(i, val):
            px = x + int(i / (len(self.prey) - 1) * (w - 4)) + 2
            py = y + h - 4 - int(val / vmax * (h - 8))
            return (px, py)

        for series, col in [(self.prey, PREY_CHART), (self.pred, PRED_CHART)]:
            pts = [to_pt(i, v) for i, v in enumerate(series)]
            if len(pts) >= 2:
                pygame.draw.lines(surface, col, False, pts, 2)


def run_visual_demo(q_tables_dir='/mnt/user-data/outputs/q_tables', num_steps=2000, fps=30):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("EvolveRL — Predator-Prey Hexagonal Simulation")
    clock = pygame.time.Clock()

    font_lg  = pygame.font.SysFont('consolas', 22, bold=True)
    font_md  = pygame.font.SysFont('consolas', 15)
    font_sm  = pygame.font.SysFont('consolas', 12)

    # ── Build static hex grid ─────────────────────────────────────────────────
    r = HEX_RADIUS
    grid_surf, grid_w, grid_h = build_grid_surface(r)

    # Scale to fit available space
    avail_w = SCREEN_W - PANEL_W - 10
    avail_h = SCREEN_H - 10
    scale   = min(avail_w / grid_w, avail_h / grid_h)
    disp_w  = int(grid_w * scale)
    disp_h  = int(grid_h * scale)
    grid_x  = 5
    grid_y  = (SCREEN_H - disp_h) // 2

    # ── Environment & agents ──────────────────────────────────────────────────
    env = PredatorPreyEnv(render_mode=None, num_prey=5, num_predator=5)
    prey_agents = [f"prey_{i}" for i in range(5)]
    pred_agents = [f"predator_{i}" for i in range(5)]
    ql_manager  = MultiAgentQLearning(prey_agents=prey_agents, predator_agents=pred_agents)

    using_trained = False
    if os.path.exists(q_tables_dir):
        try:
            ql_manager.load_all(q_tables_dir)
            using_trained = True
        except Exception:
            pass

    observations, _ = env.reset()
    chart = PopulationChart(maxlen=300)

    step       = 0
    paused     = False
    speed_mult = 1          # 1, 2, or 4 steps per frame

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running and step < num_steps:

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_1:
                    speed_mult = 1
                elif event.key == pygame.K_2:
                    speed_mult = 2
                elif event.key == pygame.K_4:
                    speed_mult = 4

        if not paused and len(observations) > 0:
            for _ in range(speed_mult):
                if using_trained:
                    actions = ql_manager.get_actions(observations, training=False)
                else:
                    actions = {}
                    for agent in observations:
                        actions[agent] = (np.random.randint(0, 7)
                                         if 'prey' in agent
                                         else np.random.randint(0, 9))

                observations, rewards, terminations, truncations, _ = env.step(actions)
                step += 1
                if len(observations) == 0:
                    break

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Scaled hex grid background
        scaled_grid = pygame.transform.scale(grid_surf, (disp_w, disp_h))
        screen.blit(scaled_grid, (grid_x, grid_y))

        # Draw agents
        hw = r * math.sqrt(3) / 2  # pointy hex half-width

        def draw_agents_of(data_dict, colour, glow):
            for agent, data in data_dict.items():
                if agent not in env.agents:
                    continue
                col, row = axial_to_grid(data['x'], data['y'])
                px, py   = hex_pixel(col, row, r)
                cx = (px + hw) * scale + grid_x
                cy = (py + r)  * scale + grid_y
                draw_agent_dot(screen, cx, cy, colour, glow, radius=max(3, int(4 * scale)))

        draw_agents_of(env.prey_data,     PREY_COL, PREY_GLOW)
        draw_agents_of(env.predator_data, PRED_COL, PRED_GLOW)

        # ── Stats panel ───────────────────────────────────────────────────────
        px0 = SCREEN_W - PANEL_W
        pygame.draw.rect(screen, PANEL_BG, (px0, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, PANEL_BORDER, (px0, 0), (px0, SCREEN_H), 2)

        py = 18
        draw_text(screen, "EvolveRL", px0 + PANEL_W // 2, py, font_lg,
                  GOLD, align='center')
        py += 28
        mode_str = "TRAINED" if using_trained else "RANDOM"
        draw_text(screen, f"Mode: {mode_str}", px0 + PANEL_W // 2, py,
                  font_sm, TEXT_DIM, align='center')
        py += 22

        # Divider
        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 12

        # Step counter
        draw_text(screen, f"Step:  {step:>5}", px0 + 14, py, font_md, TEXT_PRIMARY)
        py += 22

        # Counts
        prey_count = sum(1 for a in env.agents if 'prey' in a)
        pred_count = sum(1 for a in env.agents if 'predator' in a)

        chart.push(prey_count, pred_count)

        draw_text(screen, "▶ Prey",      px0 + 14, py, font_md, PREY_COL)
        draw_text(screen, str(prey_count), SCREEN_W - 14, py, font_md,
                  PREY_GLOW, align='right')
        py += 22
        draw_text(screen, "▶ Predators", px0 + 14, py, font_md, PRED_COL)
        draw_text(screen, str(pred_count), SCREEN_W - 14, py, font_md,
                  PRED_GLOW, align='right')
        py += 28

        # Population chart
        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 8
        draw_text(screen, "Population History", px0 + PANEL_W // 2, py,
                  font_sm, TEXT_DIM, align='center')
        py += 16
        chart_h = 110
        chart.draw(screen, (px0 + 8, py, PANEL_W - 16, chart_h), font_sm)
        py += chart_h + 14

        # Legend
        pygame.draw.rect(screen, PREY_COL,  (px0 + 14, py + 4, 10, 10))
        draw_text(screen, "Prey",      px0 + 30, py, font_sm, PREY_COL)
        pygame.draw.rect(screen, PRED_COL,  (px0 + 90, py + 4, 10, 10))
        draw_text(screen, "Predator",  px0 + 106, py, font_sm, PRED_COL)
        py += 28

        # Divider
        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 12

        # Energy stats
        if env.prey_data:
            alive_prey = [d for a, d in env.prey_data.items() if a in env.agents]
            if alive_prey:
                avg_e = sum(d['energy'] for d in alive_prey) / len(alive_prey)
                draw_text(screen, f"Prey avg energy:  {avg_e:5.1f}",
                          px0 + 14, py, font_sm, TEXT_DIM)
                py += 18
        if env.predator_data:
            alive_pred = [d for a, d in env.predator_data.items() if a in env.agents]
            if alive_pred:
                avg_e = sum(d['energy'] for d in alive_pred) / len(alive_pred)
                draw_text(screen, f"Pred avg energy:  {avg_e:5.1f}",
                          px0 + 14, py, font_sm, TEXT_DIM)
                py += 18

        py += 8
        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 12

        # Controls
        draw_text(screen, "Controls", px0 + PANEL_W // 2, py,
                  font_sm, TEXT_DIM, align='center')
        py += 18
        for line in ["SPACE  Pause / Resume",
                     "1 / 2 / 4  Speed",
                     "ESC  Quit"]:
            draw_text(screen, line, px0 + 14, py, font_sm, TEXT_DIM)
            py += 16

        # Speed / pause indicator
        py += 6
        status = "⏸ PAUSED" if paused else f"▶ {speed_mult}×"
        draw_text(screen, status, px0 + PANEL_W // 2, py, font_md,
                  GOLD if paused else PREY_GLOW, align='center')

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    q_dir = '/mnt/user-data/outputs/q_tables'
    if not os.path.exists(q_dir):
        q_dir = 'q_tables'
    run_visual_demo(q_tables_dir=q_dir, num_steps=5000, fps=30)