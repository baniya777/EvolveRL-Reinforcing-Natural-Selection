"""
EvolveRL - Visual Hexagonal Grid Simulation
Uses original hexagon.py HexagonTile for all hex rendering
"""

import sys
import os
import math
import numpy as np
import pygame

sys.path.append('/mnt/user-data/uploads')

from rl_environment import PredatorPreyEnv
from q_learning import MultiAgentQLearning
from hexagon import HexagonTile

# ── Display config ────────────────────────────────────────────────────────────
SCREEN_W   = 1280
SCREEN_H   = 800
PANEL_W    = 260
HEX_COLS   = 79
HEX_ROWS   = 39
HEX_RADIUS = 10

# ── Colours ───────────────────────────────────────────────────────────────────
BG           = (15,  20,  30)
HEX_EMPTY    = (255, 220,  30)
HEX_BORDER   = (200, 170,  10)
PREY_COL     = (60,  200,  80)
PREY_GLOW    = (120, 255, 140)
PRED_COL     = (220,  50,  50)
PRED_GLOW    = (255, 120, 100)
PANEL_BG     = (18,  24,  36)
PANEL_BORDER = (60,  80, 110)
TEXT_PRIMARY = (220, 230, 250)
TEXT_DIM     = (120, 140, 170)
GOLD         = (255, 200,  60)


def hex_position(col, row, radius):
    """Top-left position for a pointy-top HexagonTile at (col, row)."""
    minimal = radius * math.cos(math.radians(30))
    x = col * 2 * minimal + (row % 2) * minimal
    y = row * (radius + radius / 2)
    return (x, y)


def axial_to_colrow(ax, ay):
    col = ax % HEX_COLS
    row = (ay + ax // 2) % HEX_ROWS
    return col, row


def build_grid(radius):
    minimal = radius * math.cos(math.radians(30))
    surf_w  = int(HEX_COLS * 2 * minimal + minimal + 4)
    surf_h  = int(HEX_ROWS * 1.5 * radius + radius * 2 + 4)

    tiles = []
    for row in range(HEX_ROWS):
        row_tiles = []
        for col in range(HEX_COLS):
            pos  = hex_position(col, row, radius)
            tile = HexagonTile(radius=radius, position=pos, colour=HEX_EMPTY)
            row_tiles.append(tile)
        tiles.append(row_tiles)

    surf = pygame.Surface((surf_w, surf_h))
    surf.fill(BG)
    for row in tiles:
        for tile in row:
            tile.render(surf)
            tile.render_highlight(surf, HEX_BORDER)

    return tiles, surf, surf_w, surf_h


class PopulationChart:
    def __init__(self, maxlen=300):
        self.maxlen = maxlen
        self.prey   = []
        self.pred   = []

    def push(self, p, q):
        self.prey.append(p)
        self.pred.append(q)
        if len(self.prey) > self.maxlen:
            self.prey.pop(0)
            self.pred.pop(0)

    def draw(self, surface, rect):
        x, y, w, h = rect
        pygame.draw.rect(surface, (22, 30, 45), (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)
        if len(self.prey) < 2:
            return
        vmax = max(max(self.prey + self.pred), 1)

        def pt(i, val):
            px = x + int(i / (len(self.prey) - 1) * (w - 4)) + 2
            py = y + h - 4 - int(val / vmax * (h - 8))
            return (px, py)

        for series, col in [(self.prey, PREY_COL), (self.pred, PRED_COL)]:
            pts = [pt(i, v) for i, v in enumerate(series)]
            if len(pts) >= 2:
                pygame.draw.lines(surface, col, False, pts, 2)


def draw_text(surface, text, x, y, font, colour=TEXT_PRIMARY, align='left'):
    surf = font.render(text, True, colour)
    if align == 'right':
        x -= surf.get_width()
    elif align == 'center':
        x -= surf.get_width() // 2
    surface.blit(surf, (x, y))
    return surf.get_height()


def run_visual_demo(q_tables_dir='q_tables', num_steps=5000, fps=30):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("EvolveRL — Predator-Prey Hexagonal Simulation")
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont('consolas', 22, bold=True)
    font_md = pygame.font.SysFont('consolas', 15)
    font_sm = pygame.font.SysFont('consolas', 12)

    tiles, grid_surf, grid_w, grid_h = build_grid(HEX_RADIUS)

    avail_w = SCREEN_W - PANEL_W - 10
    avail_h = SCREEN_H - 10
    scale   = min(avail_w / grid_w, avail_h / grid_h)
    disp_w  = int(grid_w * scale)
    disp_h  = int(grid_h * scale)
    grid_x  = 5
    grid_y  = (SCREEN_H - disp_h) // 2

    env = PredatorPreyEnv(render_mode=None, num_prey=5, num_predator=5)
    prey_agents = [f"prey_{i}" for i in range(5)]
    pred_agents = [f"predator_{i}" for i in range(5)]
    ql_manager  = MultiAgentQLearning(
        prey_agents=prey_agents, predator_agents=pred_agents)

    using_trained = False
    if os.path.exists(q_tables_dir):
        try:
            ql_manager.load_all(q_tables_dir)
            using_trained = True
        except Exception:
            pass

    observations, _ = env.reset()
    chart      = PopulationChart(maxlen=300)
    step       = 0
    paused     = False
    speed_mult = 1

    running = True
    while running and step < num_steps:

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
                    actions = {
                        a: (np.random.randint(0, 7) if 'prey' in a
                            else np.random.randint(0, 9))
                        for a in observations
                    }
                observations, _, _, _, _ = env.step(actions)
                step += 1
                if len(observations) == 0:
                    break

        screen.fill(BG)

        scaled = pygame.transform.scale(grid_surf, (disp_w, disp_h))
        screen.blit(scaled, (grid_x, grid_y))

        def draw_agents(data_dict, fill_col, glow_col):
            for agent, data in data_dict.items():
                if agent not in env.agents:
                    continue
                col, row = axial_to_colrow(data['x'], data['y'])
                tile = tiles[row][col]
                cx = tile.centre[0] * scale + grid_x
                cy = tile.centre[1] * scale + grid_y
                r  = max(3, int(4 * scale))
                pygame.draw.circle(screen, glow_col, (int(cx), int(cy)), r + 2)
                pygame.draw.circle(screen, fill_col, (int(cx), int(cy)), r)

        draw_agents(env.prey_data,     PREY_COL, PREY_GLOW)
        draw_agents(env.predator_data, PRED_COL, PRED_GLOW)

        # Stats panel
        px0 = SCREEN_W - PANEL_W
        pygame.draw.rect(screen, PANEL_BG, (px0, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, PANEL_BORDER, (px0, 0), (px0, SCREEN_H), 2)

        py = 18
        draw_text(screen, "EvolveRL", px0 + PANEL_W // 2, py, font_lg, GOLD, align='center')
        py += 28
        draw_text(screen, f"Mode: {'TRAINED' if using_trained else 'RANDOM'}",
                  px0 + PANEL_W // 2, py, font_sm, TEXT_DIM, align='center')
        py += 22
        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 12

        draw_text(screen, f"Step:  {step:>5}", px0 + 14, py, font_md)
        py += 22

        prey_count = sum(1 for a in env.agents if 'prey' in a)
        pred_count = sum(1 for a in env.agents if 'predator' in a)
        chart.push(prey_count, pred_count)

        draw_text(screen, "Prey",       px0 + 14, py, font_md, PREY_COL)
        draw_text(screen, str(prey_count), SCREEN_W - 14, py, font_md, PREY_GLOW, align='right')
        py += 22
        draw_text(screen, "Predators",  px0 + 14, py, font_md, PRED_COL)
        draw_text(screen, str(pred_count), SCREEN_W - 14, py, font_md, PRED_GLOW, align='right')
        py += 28

        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 8
        draw_text(screen, "Population History", px0 + PANEL_W // 2, py,
                  font_sm, TEXT_DIM, align='center')
        py += 16
        chart_h = 110
        chart.draw(screen, (px0 + 8, py, PANEL_W - 16, chart_h))
        py += chart_h + 10

        pygame.draw.rect(screen, PREY_COL, (px0 + 14, py + 4, 10, 10))
        draw_text(screen, "Prey",     px0 + 30,  py, font_sm, PREY_COL)
        pygame.draw.rect(screen, PRED_COL, (px0 + 90, py + 4, 10, 10))
        draw_text(screen, "Predator", px0 + 106, py, font_sm, PRED_COL)
        py += 28

        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 12

        alive_prey = [d for a, d in env.prey_data.items() if a in env.agents]
        if alive_prey:
            avg_e = sum(d['energy'] for d in alive_prey) / len(alive_prey)
            draw_text(screen, f"Prey avg energy: {avg_e:5.1f}", px0 + 14, py, font_sm, TEXT_DIM)
            py += 18
        alive_pred = [d for a, d in env.predator_data.items() if a in env.agents]
        if alive_pred:
            avg_e = sum(d['energy'] for d in alive_pred) / len(alive_pred)
            draw_text(screen, f"Pred avg energy: {avg_e:5.1f}", px0 + 14, py, font_sm, TEXT_DIM)
            py += 18

        py += 8
        pygame.draw.line(screen, PANEL_BORDER, (px0 + 10, py), (SCREEN_W - 10, py))
        py += 12

        for line in ["SPACE  Pause/Resume", "1/2/4  Speed", "ESC    Quit"]:
            draw_text(screen, line, px0 + 14, py, font_sm, TEXT_DIM)
            py += 16

        py += 6
        status = "PAUSED" if paused else f"Speed: {speed_mult}x"
        draw_text(screen, status, px0 + PANEL_W // 2, py, font_md,
                  GOLD if paused else PREY_GLOW, align='center')

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    q_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q_tables')
    if not os.path.exists(q_dir):
        q_dir = '/mnt/user-data/outputs/q_tables'
    run_visual_demo(q_tables_dir=q_dir, num_steps=5000, fps=30)
