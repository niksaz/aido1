import abc
import copy
import multiprocessing as mp

import imageio
import numpy as np
import pygame

import utils.math_utils as mu


class GraphicsBase:
    @abc.abstractclassmethod
    def refresh_frame(self, observation, score_update, step):
        pass


class VirtualGraphics(GraphicsBase):
    def __init__(self, config):
        self.queue = mp.Queue()
        self.process = mp.Process(target=_graphics_worker, args=(config, self.queue))
        self.process.start()

    def refresh_frame(self, observation, score_update, step):
        self.queue.put({'type': 'update', 'args': {'observation': copy.deepcopy(observation),
                                                   'score_update': score_update, 'step': step}})

    def __del__(self):
        self.queue.put({'type': 'end'})
        self.process.join()
        self.queue.close()


def _graphics_worker(config, query_queue):
    graphics = Graphics(config)
    while True:
        query = query_queue.get()
        if query['type'] == 'end':
            break
        elif query['type'] == 'update':
            graphics.refresh_frame(**query['args'])


class Graphics(GraphicsBase):
    def __init__(self, config: dict):
        self.frames = []
        self.fps = config.get('fps', 50)
        self.show_time = config.get('show_time', True)
        self.show_step = config.get('show_step', True)
        self.show_score = config.get('show_score', True)
        self.show_grid = config.get('show_grid', True)

        self.grid_step = config.get('grid_step', 0.5)
        self.grid_border = config.get('grid_step', 30.)
        self.borders = config.get('vertical_borders', .6)
        self.save_file = config.get('save_file', None)

        self.camera_rotation = np.array(config.get('camera_rotation', [-0.3, -0.5, 0.]))
        self.camera_perspective_deep = config.get('camera_perspective_distance', 1.)
        self.camera_position = np.array(config.get('camera_position', (0., 0.5, -3.)))

        self.colors = config.get('colors', {})
        self.size = config.get('size', [800, 600])

        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode(self.size)
        self._font = pygame.font.SysFont('Consolas', 22)
        self._video_writer = None
        if self.save_file is not None:
            self._video_writer = imageio.get_writer(self.save_file + ".mp4", fps=self.fps, quality=10)

    def __del__(self):
        self.save()
        pygame.quit()

    def refresh_frame(self, observation, score_update, step):
        self._current_position = np.array((observation['body_pos']['pelvis'][0], 0., observation['body_pos']['pelvis'][2]))
        self._current_angles = self.camera_rotation

        pygame.event.get()
        self._screen.fill(self.colors.get('background', [0, 0, 0]))

        if self.show_grid:
            self._draw_grid()
        self._draw_observation(observation['body_pos'])
        self._draw_velocities(observation)
        self._draw_info(score_update, step)

        pygame.display.flip()
        self._save_frame()

    def save(self):
        if self._video_writer is not None:
            self._video_writer.close()

    def _draw_observation(self, observation):
        body_parts = []
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['head']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['torso']) - self._current_position, self._current_angles),
            self.colors.get('body', (100, 230, 0))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['torso']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['pelvis']) - self._current_position, self._current_angles),
            self.colors.get('body', (100, 230, 0))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['pelvis']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['femur_l']) - self._current_position, self._current_angles),
            self.colors.get('body', (100, 230, 0))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['femur_l']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['femur_r']) - self._current_position, self._current_angles),
            self.colors.get('body', (100, 230, 0))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['pelvis']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['femur_r']) - self._current_position, self._current_angles),
            self.colors.get('body', (100, 230, 0))
        ))

        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['femur_l']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['tibia_l']) - self._current_position, self._current_angles),
            self.colors.get('left', (30, 90, 230))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['tibia_l']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['talus_l']) - self._current_position, self._current_angles),
            self.colors.get('left', (30, 90, 230))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['talus_l']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['toes_l']) - self._current_position, self._current_angles),
            self.colors.get('left', (30, 90, 230))
        ))

        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['femur_r']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['pros_tibia_r']) - self._current_position, self._current_angles),
            self.colors.get('right', (230, 90, 30))
        ))
        body_parts.append((
            mu.rotate_3d(self._symm_bodypart(observation['pros_tibia_r']) - self._current_position, self._current_angles),
            mu.rotate_3d(self._symm_bodypart(observation['pros_foot_r']) - self._current_position, self._current_angles),
            self.colors.get('right', (230, 90, 30))
        ))

        body_parts.sort(
            key=lambda body_part: (body_part[0][2] + body_part[1][2], min(body_part[0][2], body_part[1][2])),
            reverse=True
        )
        for start, end, color in body_parts:
            self._perspective_line(color, start, end, 5)

    def _draw_grid(self):
        x_margin = self._current_position[0] / self.grid_step
        x_margin = -self.grid_step * (x_margin - int(x_margin))
        z_margin = self._current_position[2] / self.grid_step
        z_margin = self.grid_step * (z_margin - int(z_margin))
        for x_coord in np.arange(-self.grid_border, self.grid_border, self.grid_step):
            start = mu.rotate_3d((x_coord + x_margin, 0., -self.grid_border), self._current_angles, self.camera_position)
            end = mu.rotate_3d((x_coord + x_margin, 0., self.grid_border), self._current_angles, self.camera_position)
            self._perspective_line(self.colors.get('grid', (50, 50, 50)), start, end, 2)

        for z_coord in np.arange(-self.grid_border, self.grid_border, self.grid_step):
            start = mu.rotate_3d((-self.grid_border, 0., z_coord + z_margin), self._current_angles, self.camera_position)
            end = mu.rotate_3d((self.grid_border, 0., z_coord + z_margin), self._current_angles)
            self._perspective_line(self.colors.get('grid', (50, 50, 50)), start, end, 2)

    def _draw_velocities(self, observation):
        current_velocity = copy.deepcopy(observation['body_vel']['pelvis'])
        current_velocity[1] = 0.
        current_velocity[2] *= -1.
        current_velocity = mu.rotate_3d(current_velocity, self._current_angles)

        target_velocity = copy.deepcopy(observation['target_vel'])
        target_velocity[2] *= -1.
        target_velocity = mu.rotate_3d(target_velocity, self._current_angles)
        zero = [0., 0., 0.]
        self._perspective_line((230, 180, 0, 200), zero, target_velocity, 4)
        self._perspective_line((0, 230, 180, 200), zero, current_velocity, 3)

    def _draw_info(self, score_update, step):
        y = 0
        color = self.colors.get("font", (240, 140, 0))
        if self.show_time:
            rendered_text = self._font.render('%f s.' % (step / 100), False, color)
            self._screen.blit(rendered_text, (0, y))
            y += rendered_text.get_height()
        if self.show_step:
            rendered_text = self._font.render('Step: %d' % step, False, color)
            self._screen.blit(rendered_text, (0, y))
            y += rendered_text.get_height()
        if self.show_score:
            rendered_text = self._font.render('Reward: %f' % score_update, False, color)
            self._screen.blit(rendered_text, (0, y))
            y += rendered_text.get_height()

    def _perspective_line(self, color, start, end, width):
        start, end = mu.cut_line(start, end,
                                 left_border=self.camera_position + np.array((-self.grid_border, -self.grid_border,
                                                                              self._current_position[2] - self.camera_perspective_deep - 1e-2)),
                                 right_border=(self.grid_border, self.grid_border, self.grid_border))
        start = mu.perspective(start, self.camera_position, self.camera_perspective_deep)
        end = mu.perspective(end, self.camera_position, self.camera_perspective_deep)
        self._line(color, start, end, width)

    def _line(self, color, start, end, width):
        start = self._cam_view_res_to_px(start)
        end = self._cam_view_res_to_px(end)

        start, end = mu.cut_line(start, end, (1e-2, 1e-2), (self.size[0] - 1e-2, self.size[1] - 1e-2))

        if (start[0] < 0 or end[0] < 0) or \
           (start[0] > self.size[0] or end[0] > self.size[0]) or \
           (start[1] < 0 or end[1] < 0) or \
           (start[1] > self.size[1] or end[1] > self.size[1]):
            return

        pygame.draw.line(self._screen, color, start, end, width)

    def _symm_bodypart(self, body_part):
        return np.array((body_part[0], body_part[1], 2 * self._current_position[2] - body_part[2]))

    def _save_frame(self):
        if self._video_writer is not None:
            self._video_writer.append_data(np.swapaxes(np.array(pygame.surfarray.array3d(self._screen)), 0, 1))
        pass

    def _cam_view_res_to_px(self, point2d):
        px_pt = self.size[1] * (point2d - self.camera_position[:2]) / (2 * self.borders)
        return np.array([px_pt[0] + self.size[0] // 2, self.size[1] // 2 - px_pt[1]])
