import numpy as np
import copy

from utils.bresenham import bresenham


def get_pos_with_anchor(body_pos, anchor='pelvis'):
    body_pos = copy.deepcopy(body_pos)

    for body_part in body_pos:
        body_pos[body_part] = np.array(body_pos[body_part])

    body_ground = np.array(body_pos[anchor])

    for body_part in body_pos:
        body_pos[body_part] -= body_ground

    return body_pos


def get_image_from_pos(body_pos, projection='side'):
    body_pos_grounded = get_pos_with_anchor(body_pos)

    body_pos_depth = {}

    for body_part in body_pos_grounded:
        if projection == 'side':
            body_pos_depth[body_part] = body_pos_grounded[body_part][2]
            body_pos_grounded[body_part] = body_pos_grounded[body_part][:2]
        if projection == 'front':
            body_pos_depth[body_part] = body_pos_grounded[body_part][1]
            body_pos_grounded[body_part] = body_pos_grounded[body_part][1:]

    for body_part in body_pos_grounded:
        body_pos_grounded[body_part] += np.ones(2, dtype=np.float)

    picture_size = 32

    for body_part in body_pos_grounded:
        body_pos_grounded[body_part] *= picture_size // 2

    for body_part in body_pos_grounded:
        body_pos_grounded[body_part] = np.rint(body_pos_grounded[body_part]).astype(np.int)

    # spine = ('head', 'torso', 'pelvis')
    # left_hip = ('femur_l', 'tibia_l')
    # left_ankle = ('tibia_l', 'talus_l', 'calcn_l', 'toes_l')
    # right_hip = ('femur_r', 'pros_tibia_r')
    # right_ankle = ('pros_tibia_r', 'pros_foot_r')
    #
    # layers_lines = (spine, left_hip, left_ankle, right_hip, right_ankle)

    spine = ('head', 'torso', 'pelvis')
    left_leg = ('femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l')
    right_leg = ('femur_r', 'pros_tibia_r', 'pros_foot_r')

    layers_lines = (spine, left_leg, right_leg)

    layers = []
    depth_image = np.zeros(shape=(picture_size, picture_size), dtype=np.float32)

    for layer in layers_lines:
        image = np.zeros(shape=(picture_size, picture_size), dtype=np.float32)
        depth_image = np.zeros(shape=(picture_size, picture_size), dtype=np.float32)

        for i in range(1, len(layer)):
            start_vertex = body_pos_grounded[layer[i - 1]].tolist()
            end_vertex = body_pos_grounded[layer[i]].tolist()

            xys = list(bresenham(*start_vertex, *end_vertex))
            xys = list(filter(lambda xy: xy[0] < picture_size and xy[1] < picture_size, xys))

            depths = np.linspace(body_pos_depth[layer[i - 1]], body_pos_depth[layer[i]], len(xys)).tolist()

            for (x, y), depth in zip(xys, depths):
                image[x, y] = 1.0
                depth_image[x, y] = depth
                # image[x, y] = depth

        layers.append(image)
        # layers.append(depth_image)

    layers.append(depth_image)

    for i in range(len(layers)):
        layers[i] = np.reshape(layers[i], newshape=(1, layers[i].shape[0], layers[i].shape[1]))

    layers = np.concatenate(layers).astype(dtype=np.float16)
    return layers
