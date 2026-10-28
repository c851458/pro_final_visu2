"""Visualizacion grafica de Warehouse Rush con pygame."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

BG = (24, 26, 32)
WALL = (55, 58, 66)
WALL_EDGE = (35, 37, 43)
FLOOR = (214, 210, 200)
FLOOR_LINE = (190, 186, 176)
SHELF = (120, 78, 40)
SHELF_EDGE = (80, 50, 25)
BOX = (176, 128, 74)
BOX_EDGE = (120, 82, 44)
DELIVERY = (88, 176, 104)
DELIVERY_EDGE = (50, 120, 64)
ITEM = (245, 196, 30)
ITEM_EDGE = (150, 110, 0)
AGENT = (40, 110, 230)
AGENT_LOADED = (220, 60, 60)
TEXT = (235, 235, 235)
TEXT_DIM = (150, 155, 165)
OK = (110, 210, 120)
BAD = (235, 110, 100)

ACTION_NAMES = ["arriba", "abajo", "izquierda", "derecha", "recoger", "entregar"]
PANEL_HEIGHT = 150


class PygameRenderer:
    """Dibuja la bodega, el agente y un panel de informacion.

    Controles en la ventana: ESPACIO pausa/continua, ESC o cerrar la ventana la cierra.
    """

    def __init__(self, env, cell_size, fps, human):
        self.env = env
        self.cs = cell_size
        self.fps = fps
        self.human = human
        self.size = (env.width * cell_size, env.height * cell_size + PANEL_HEIGHT)
        self.closed_by_user = False
        self.paused = False
        self.allow_pause = True
        self.hint = "ESPACIO pausa  |  ESC cerrar"
        self.pending_keys = []  # teclas no usadas por la ventana, para control manual

        pygame.init()
        if human:
            pygame.display.set_caption("Warehouse Rush")
            self.screen = pygame.display.set_mode(self.size)
        else:
            self.screen = pygame.Surface(self.size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.font_small = pygame.font.SysFont("dejavusans", 11, bold=True)
        self.font_big = pygame.font.SysFont("dejavusans", 17, bold=True)

    # ------------------------------------------------------------------ dibujo
    def draw(self):
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_labels()
        self._draw_item()
        self._draw_agent()
        self._draw_panel()

    def _rect(self, r, c, margin=0):
        cs = self.cs
        return pygame.Rect(c * cs + margin, r * cs + margin, cs - 2 * margin, cs - 2 * margin)

    def _draw_grid(self):
        for (r, c), cell in np.ndenumerate(self.env.grid):
            rect = self._rect(r, c)
            if cell == "#":
                pygame.draw.rect(self.screen, WALL, rect)
                pygame.draw.rect(self.screen, WALL_EDGE, rect, 2)
            elif cell == "S":
                pygame.draw.rect(self.screen, SHELF, rect)
                pygame.draw.rect(self.screen, SHELF_EDGE, rect, 2)
                # dos cajas por casilla de estanteria
                h = (self.cs - 12) // 2
                for i in range(2):
                    box = pygame.Rect(rect.x + 6, rect.y + 4 + i * (h + 4), rect.w - 12, h)
                    pygame.draw.rect(self.screen, BOX, box, border_radius=2)
                    pygame.draw.rect(self.screen, BOX_EDGE, box, 1, border_radius=2)
            elif cell == "D":
                pygame.draw.rect(self.screen, DELIVERY, rect)
                self.screen.set_clip(rect)
                for k in range(0, self.cs * 2, 10):  # rayas de zona de carga
                    pygame.draw.line(self.screen, DELIVERY_EDGE,
                                     (rect.x + k, rect.y), (rect.x + k - self.cs, rect.bottom), 1)
                self.screen.set_clip(None)
                pygame.draw.rect(self.screen, DELIVERY_EDGE, rect, 2)
            else:
                pygame.draw.rect(self.screen, FLOOR, rect)
                pygame.draw.rect(self.screen, FLOOR_LINE, rect, 1)

    def _draw_labels(self):
        top = 1
        # pasillos: columnas libres entre estanterias
        shelf_cols = sorted({c for _, c in self.env.shelf_cells})
        aisles = [c for c in range(shelf_cols[0] + 1, shelf_cols[-1]) if c not in shelf_cols]
        for i, c in enumerate(aisles, 1):
            self._label(f"P{i}", top + 0.5, c + 0.5, TEXT_DIM)
        # estanterias: grupos de columnas contiguas
        groups, current = [], [shelf_cols[0]]
        for c in shelf_cols[1:]:
            if c == current[-1] + 1:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)
        for i, cols in enumerate(groups, 1):
            self._label(f"E{i}", top + 0.5, (cols[0] + cols[-1] + 1) / 2, TEXT_DIM)
        dr = [r for (r, _) in self.env.delivery_cells]
        dc = [c for (_, c) in self.env.delivery_cells]
        self._label("ENTREGAS", max(dr) + 0.5, (min(dc) + max(dc) + 1) / 2, (255, 255, 255))

    def _label(self, text, r, c, color):
        surf = self.font_small.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=(c * self.cs, r * self.cs)))

    def _draw_item(self):
        if self.env.carrying or self.env.delivered:
            return
        r, c = self.env.shelf_cells[self.env.item_idx]
        rect = self._rect(r, c, margin=self.cs // 5)
        pygame.draw.rect(self.screen, ITEM, rect, border_radius=4)
        pygame.draw.rect(self.screen, ITEM_EDGE, rect, 2, border_radius=4)
        pygame.draw.line(self.screen, ITEM_EDGE, rect.midtop, rect.midbottom, 2)

    def _draw_agent(self):
        r, c = self.env.agent_pos
        rect = self._rect(r, c)
        color = AGENT_LOADED if self.env.carrying else AGENT
        pygame.draw.circle(self.screen, (0, 0, 0), (rect.centerx + 2, rect.centery + 3), self.cs // 2 - 6)
        pygame.draw.circle(self.screen, color, rect.center, self.cs // 2 - 6)
        pygame.draw.circle(self.screen, (255, 255, 255), rect.center, self.cs // 2 - 6, 2)
        if self.env.carrying:
            box = self._rect(r, c, margin=self.cs // 3)
            pygame.draw.rect(self.screen, ITEM, box, border_radius=3)
            pygame.draw.rect(self.screen, ITEM_EDGE, box, 2, border_radius=3)

    def _draw_panel(self):
        env = self.env
        y0 = env.height * self.cs + 10
        action = ACTION_NAMES[env.last_action] if env.last_action is not None else "-"
        if env.delivered:
            status, color = "ENTREGADO", OK
        elif env.steps >= env.max_steps:
            status, color = "TIEMPO AGOTADO", BAD
        elif env.carrying:
            status, color = "Llevando paquete a ENTREGAS", AGENT_LOADED
        else:
            status, color = f"Buscando paquete en {env.shelf_cells[env.item_idx]}", ITEM
        self.screen.blit(self.font_big.render(status, True, color), (12, y0))
        stats = (f"Paso {env.steps}/{env.max_steps}   Accion: {action}   "
                 f"Recompensa: {env.last_reward:+.1f}   Acumulada: {env.total_reward:+.1f}")
        self.screen.blit(self.font.render(stats, True, TEXT), (12, y0 + 26))
        for i, line in enumerate(env.hud_extra[:4]):
            self.screen.blit(self.font.render(line, True, TEXT_DIM), (12, y0 + 50 + i * 19))
        hint = self.hint + ("   [PAUSA]" if self.paused else "")
        surf = self.font.render(hint, True, TEXT_DIM)
        self.screen.blit(surf, (self.size[0] - surf.get_width() - 12, self.size[1] - 22))

    # ------------------------------------------------------------ salida
    def show(self):
        """Dibuja y presenta un fotograma en la ventana respetando los FPS."""
        self.draw()
        pygame.display.flip()
        self._handle_events()
        while self.paused and not self.closed_by_user:
            self.draw()
            pygame.display.flip()
            self._handle_events()
            self.clock.tick(20)
        self.clock.tick(self.fps)

    def hold(self, seconds=None):
        """Mantiene la ventana abierta `seconds` segundos, o hasta que el usuario la cierre."""
        start = pygame.time.get_ticks()
        while not self.closed_by_user:
            if seconds is not None and pygame.time.get_ticks() - start >= seconds * 1000:
                break
            self.draw()
            pygame.display.flip()
            self._handle_events()
            self.clock.tick(20)

    def rgb_array(self):
        self.draw()
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.closed_by_user = True
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and self.allow_pause:
                self.paused = not self.paused
            elif event.type == pygame.KEYDOWN:
                self.pending_keys.append(event.key)

    def pop_keys(self):
        keys, self.pending_keys = self.pending_keys, []
        return keys

    def close(self):
        if self.human:
            pygame.display.quit()
        pygame.quit()
