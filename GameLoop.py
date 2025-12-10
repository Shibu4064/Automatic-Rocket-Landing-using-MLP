import sys
import os
import pygame

from EventHandler import EventHandler
from Lander import Lander
from Controller import Controller
from Vector import Vector
from GameLogic import GameLogic
from Surface import Surface
from MainMenu import MainMenu
from ResultMenu import ResultMenu
from DataCollection import DataCollection
from NeuralNetHolder import NeuralNetHolder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(p):
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


class GameLoop:
    def __init__(self):
        self.controller = Controller()
        self.Handler = EventHandler(self.controller)
        self.object_list = []
        self.game_logic = GameLogic()
        self.fps_clock = pygame.time.Clock()
        self.fps = 60

        # Neural net with trained weights
        self.neuralnet = NeuralNetHolder()

        self.version = "v1.01"
        self.prediction_cycle = 0

        # Autopilot state tracking
        self.auto_phase = "APPROACH"
        self.hover_start_time = 0

    # ---------------- Window setup ----------------
    def init(self, config_data):
        pygame.init()

        if "SCREEN_WIDTH" in config_data:
            try:
                config_data["SCREEN_WIDTH"] = int(config_data["SCREEN_WIDTH"])
                config_data["SCREEN_HEIGHT"] = int(config_data["SCREEN_HEIGHT"])
            except Exception:
                pass

        flags = pygame.SCALED
        if str(config_data.get("FULLSCREEN", "FALSE")).upper() == "TRUE":
            info = pygame.display.Info()
            config_data["SCREEN_WIDTH"] = info.current_w
            config_data["SCREEN_HEIGHT"] = info.current_h
            flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode(
            (config_data["SCREEN_WIDTH"], config_data["SCREEN_HEIGHT"]), flags
        )
        pygame.display.set_caption("CE889 Assignment - NN Autopilot")

        try:
            icon_path = _resolve_path(config_data["LANDER_IMG_PATH"])
            icon_img = pygame.image.load(icon_path)
            pygame.display.set_icon(icon_img)
        except Exception as e:
            print(f"[WARN] Could not set window icon: {e}", file=sys.stderr)

    # ---------------- Score ----------------
    def score_calculation(self):
        score = 1000.0 - abs(self.surface.centre_landing_pad[0] - self.lander.position.x)
        angle = self.lander.current_angle or 1
        if self.lander.current_angle > 180:
            angle = abs(self.lander.current_angle - 360)
        score /= angle
        velocity = 500 - (abs(self.lander.velocity.x) + abs(self.lander.velocity.y))
        score += velocity
        print(f"SCORE {score:.2f}")
        return score

    # ---------------- Main loop ----------------
    def main_loop(self, config_data):
        pygame.font.init()
        myfont = pygame.font.SysFont("Comic Sans MS", 30)
        sprites = pygame.sprite.Group()

        on_menus = [True, False, False]  # main, won, lost
        game_start = False
        # [manual, data collection, NN autopilot, quit]
        game_modes = [False, False, False, False]

        bg_path = _resolve_path(config_data["BACKGROUND_IMG_PATH"])
        background_image = pygame.image.load(bg_path).convert_alpha()
        background_image = pygame.transform.scale(
            background_image,
            (config_data["SCREEN_WIDTH"], config_data["SCREEN_HEIGHT"]),
        )

        data_collector = DataCollection(config_data["ALL_DATA"])
        main_menu = MainMenu(
            (config_data["SCREEN_WIDTH"], config_data["SCREEN_HEIGHT"])
        )
        result_menu = ResultMenu(
            (config_data["SCREEN_WIDTH"], config_data["SCREEN_HEIGHT"])
        )
        score = 0

        while True:
            if game_modes[-1]:
                pygame.quit()
                sys.exit(0)

            if game_start:
                self.controller = Controller()
                self.Handler = EventHandler(self.controller)
                sprites = pygame.sprite.Group()
                self.auto_phase = "APPROACH"
                self.hover_start_time = 0
                self.game_start(config_data, sprites)
                game_start = False

            # ---------- MENUS ----------
            if on_menus[0] or on_menus[1] or on_menus[2]:
                if on_menus[1] or on_menus[2]:
                    result_menu.draw_result_objects(self.screen, on_menus[1], score)
                else:
                    main_menu.draw_buttons(self.screen)
                    textsurface = myfont.render(self.version, False, (0, 0, 0))
                    self.screen.blit(textsurface, (0, 0))

                for event in pygame.event.get():
                    if on_menus[0]:
                        main_menu.check_hover(event)
                        button_clicked = main_menu.check_button_click(event)
                        main_menu.draw_buttons(self.screen)
                        if button_clicked > -1:
                            game_modes[button_clicked] = True
                            on_menus[0] = False
                            game_start = True

                    elif on_menus[1] or on_menus[2]:
                        result_menu.check_hover(event)
                        on_menus[0] = result_menu.check_back_main_menu(event)
                        result_menu.draw_result_objects(self.screen, on_menus[1], score)
                        if on_menus[0]:
                            on_menus[1] = False
                            on_menus[2] = False

            # ---------- IN GAME ----------
            else:
                self.Handler.handle(pygame.event.get())
                self.screen.blit(background_image, (0, 0))

                if (not self.Handler.first_key_press) and not game_start:
                    self.update_objects()

                # ----- DATA COLLECTION -----
                if game_modes[1] and self.Handler.first_key_press:
                    data_input_row = data_collector.get_input_row(
                        self.lander, self.surface, self.controller
                    )
                    self.update_objects()
                    data_collector.save_current_status(
                        data_input_row, self.lander, self.surface, self.controller
                    )

                # ----- NN AUTOPILOT MODE -----
                elif game_modes[2]:
                    self.Handler.first_key_press = True
                    x_to_target = self.surface.centre_landing_pad[0] - self.lander.position.x
                    y_to_target = self.surface.centre_landing_pad[1] - self.lander.position.y
                    vx = self.lander.velocity.x
                    vy = self.lander.velocity.y

                    try:
                        input_row = f"{x_to_target},{y_to_target}"
                        nn_vx, nn_vy = self.neuralnet.predict(input_row)
                    except Exception as e:
                        print(f"NN Error: {e}")
                        nn_vx, nn_vy = 0.0, 0.5

                    self.controller.set_up(False)
                    self.controller.set_left(False)
                    self.controller.set_right(False)
                    horizontal_dist = abs(x_to_target)
                    vertical_dist = y_to_target
                    if vertical_dist > 200:
                        if horizontal_dist > 50:
                            desired_vx = 1.2 if x_to_target > 0 else -1.2
                        else:
                            desired_vx = 0.6 if x_to_target > 0 else -0.6
                        desired_vy = 1.2
                    elif vertical_dist > 100:
                        if horizontal_dist > 30:
                            desired_vx = 0.8 if x_to_target > 0 else -0.8
                        else:
                            desired_vx = 0.3 if x_to_target > 0 else -0.3
                        desired_vy = 0.8
                    elif vertical_dist > 50:
                        if horizontal_dist > 20:
                            desired_vx = 0.5 if x_to_target > 0 else -0.5
                        else:
                            desired_vx = 0.2 if x_to_target > 0 else -0.2
                        desired_vy = 0.6
                    else:
                        if horizontal_dist > 15:
                            desired_vx = 0.3 if x_to_target > 0 else -0.3
                        elif horizontal_dist > 5:
                            desired_vx = 0.15 if x_to_target > 0 else -0.15
                        else:
                            desired_vx = 0.0
                        desired_vy = 0.35
                    if vertical_dist < 30:
                        desired_vy = 0.25
                        if horizontal_dist < 5:
                            desired_vx = 0.0
                    if vertical_dist < 150:
                        target_vx = desired_vx
                        target_vy = desired_vy
                    elif abs(nn_vy) < 0.1 and vertical_dist > 50:
                        target_vx = desired_vx
                        target_vy = desired_vy
                    elif abs(nn_vx) > 5.0:
                        target_vx = desired_vx
                        target_vy = max(0.1, min(nn_vy, 2.0))
                    else:
                        target_vx = nn_vx if horizontal_dist < 100 else desired_vx
                        target_vy = max(0.1, min(nn_vy, 1.5))
                    if pygame.time.get_ticks() % 1000 < 20:
                        print(f"\n=== LANDING STATUS ===")
                        print(f"Distance: x={x_to_target:.1f}, y={y_to_target:.1f}")
                        print(f"Current vel: vx={vx:.2f}, vy={vy:.2f}")
                        print(f"NN suggests: vx={nn_vx:.2f}, vy={nn_vy:.2f}")
                        print(f"Target vel: vx={target_vx:.2f}, vy={target_vy:.2f}")
                    
                    vx_error = target_vx - vx
                    if vertical_dist > 100:
                        vx_threshold = 0.2
                    elif vertical_dist > 50:
                        vx_threshold = 0.15
                    else:
                        vx_threshold = 0.1
                    if abs(vx_error) > vx_threshold:
                        if vx_error > 0:
                            self.controller.set_right(True)
                        else:
                            self.controller.set_left(True)

                    vy_error = target_vy - vy
                    if vertical_dist > 100:
                        vy_threshold = 0.2
                    else:
                        vy_threshold = 0.15
                    if vy > target_vy + vy_threshold:
                        self.controller.set_up(True)
                    if vertical_dist < 150 and vy > 1.2:
                        self.controller.set_up(True)
                    if vertical_dist < 80 and vy > 0.9:
                        self.controller.set_up(True)
                    if vertical_dist < 30 and vy > 0.6:
                        self.controller.set_up(True)
                    if vertical_dist < 40:
                        if horizontal_dist > 25:
                            if x_to_target > 0:
                                self.controller.set_right(True)
                            else:
                                self.controller.set_left(True)
                            if vy > 0.4:
                                self.controller.set_up(True)
                    if vertical_dist < 60:
                        if abs(vx) > 0.8:
                            if vx > 0:
                                self.controller.set_left(True)
                            else:
                                self.controller.set_right(True)
                            if vy > 0.3:
                                self.controller.set_up(True)
                        if vy > 0.7:
                            self.controller.set_up(True)
                    if vertical_dist < 20:
                        if abs(vx) > 0.3:
                            if vx > 0:
                                self.controller.set_left(True)
                            else:
                                self.controller.set_right(True)
                        if vy > 0.4:
                            self.controller.set_up(True)
                    if pygame.time.get_ticks() % 1000 < 20:
                        print(f"Controls: UP={self.controller.up}, L={self.controller.left}, R={self.controller.right}\n")
                    if 30 < self.lander.current_angle < 330:
                        ang_val = round((self.lander.current_angle - 30) / 300.0)
                        self.lander.current_angle = 30 if ang_val == 0 else 330

                    self.update_objects()
                else:
                    if self.Handler.first_key_press:
                        self.update_objects()
                sprites.draw(self.screen)

                if self.lander.landing_pad_collision(self.surface):
                    score = self.score_calculation()
                    on_menus[1] = True
                    if game_modes[1]:
                        data_collector.write_to_file()
                        data_collector.reset()
                    print("✓ SUCCESSFUL LANDING!")
                elif (
                    self.lander.surface_collision(self.surface)
                    or self.lander.window_collision(
                        (
                            config_data["SCREEN_WIDTH"],
                            config_data["SCREEN_HEIGHT"],
                        )
                    )
                ):
                    on_menus[2] = True
                    data_collector.reset()
                    print("✗ CRASH!")
                if on_menus[1] or on_menus[2]:
                    game_start = False
                    game_modes = [False] * 4
            pygame.display.flip()
            self.fps_clock.tick(self.fps)

    # ---------------- Helpers ----------------
    def update_objects(self):
        self.game_logic.update(0.2)

    def setup_lander(self, config_data):
        lander = Lander(
            _resolve_path(config_data["LANDER_IMG_PATH"]),
            [
                config_data["SCREEN_WIDTH"] / 2,
                config_data["SCREEN_HEIGHT"] / 2,
            ],
            Vector(0, 0),
            self.controller,
        )
        self.game_logic.add_lander(lander)
        return lander

    def game_start(self, config_data, sprites):
        self.lander = self.setup_lander(config_data)
        self.surface = Surface(
            (config_data["SCREEN_WIDTH"], config_data["SCREEN_HEIGHT"])
        )
        sprites.add(self.lander)
        sprites.add(self.surface)



