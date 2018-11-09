import math
import numpy as np


def rotate_coors(observation):
    if np.linalg.norm(observation['target_vel']) < 1e-2:
        return observation

    target_vels = np.array(observation['target_vel'])
    angle = -math.atan2(target_vels[2], target_vels[0])

    for feature in ('pos', 'vel', 'acc'):
        idx = 'body_{}_rot'.format(feature)
        for body_part_name in observation[idx]:
            observation[idx][body_part_name][1] += angle

        idx = 'body_{}'.format(feature)
        for body_part_name in observation[idx]:
            cur_body_part = observation[idx][body_part_name]
            cur_body_part[0], cur_body_part[2] = rotate_by((cur_body_part[0], cur_body_part[2]), angle)
            observation[idx][body_part_name] = cur_body_part

        idx = 'mass_center_{}'.format(feature)
        cur_mass_center = observation['misc'][idx]
        cur_mass_center[0], cur_mass_center[2] = rotate_by((cur_mass_center[0], cur_mass_center[2]), angle)
        observation['misc'][idx] = cur_mass_center

    target_vel = observation['target_vel']
    target_vel[0], target_vel[2] = rotate_by((target_vel[0], target_vel[2]), angle)
    observation['target_vel'] = target_vel

    return observation


def rotate_3d(point3d, angles, center=(0, 0, 0), order=(1, 0, 2)):
    point3d = np.array(point3d) - np.array(center)

    def rot_Ox():
        point3d[1], point3d[2] = rotate_by((point3d[1], point3d[2]), angles[0])

    def rot_Oy():
        point3d[0], point3d[2] = rotate_by((point3d[0], point3d[2]), angles[1])

    def rot_Oz():
        point3d[0], point3d[1] = rotate_by((point3d[0], point3d[1]), angles[2])

    rot_funs = [rot_Ox, rot_Oy, rot_Oz]
    for f in order:
        rot_funs[f]()
    return point3d + np.array(center)


def rotate_by(point2d, angle):
    x, y = point2d
    x_rot = math.cos(angle) * x - math.sin(angle) * y
    y_rot = math.sin(angle) * x + math.cos(angle) * y
    return x_rot, y_rot


def perspective(point3d, camera_position=(0., 0., -1.), camera_deep=1.):
    point3d = np.array(point3d) - np.array(camera_position)
    scale_coeff = camera_deep / max(point3d[2], 1e-6)
    return point3d[:2]*scale_coeff + np.array(camera_position[:2])


def cut_line(start, end, left_border, right_border):
    start = np.array(start)
    end = np.array(end)
    for i in range(len(start)):
        if start[i] < end[i]:
            start, end = end, start
        if start[i] < left_border[i] and start[i] - end[i] > 1e-2:
            start = (start - end) * (left_border[i] - end[i]) / (start[i] - end[i]) + end
        if start[i] > right_border[i] and start[i] - end[i] > 1e-2:
            start = (start - end) * (right_border[i] - end[i]) / (start[i] - end[i]) + end
        if end[i] < left_border[i] and end[i] - start[i] < -1e-2:
            end = (end - start) * (left_border[i] - start[i]) / (end[i] - start[i]) + start
        if end[i] > right_border[i] and end[i] - start[i] < -1e-2:
            end = (end - start) * (right_border[i] - start[i]) / (end[i] - start[i]) + start
    return start, end
