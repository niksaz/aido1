import random

from osim.env.osim import rect

from utils.features import Relatives, Muscles, Plainer, Forces, Heights, MassCenter, TargetVelocity, TimeFromChange, \
    Activations
from utils.reward_shaping.additive_functions import *
from utils.reward_shaping.aggregation_functions import *
from utils.util import set_seeds

DEFAULT_SEED_OFFSET = 2237
DEFAULT_SEED_MODULO = 1000000007


class Rewarder:
    def __init__(self, additive_functions_config, aggregation_functions_config, observation):
        self.additive_functions = []
        for cls in additive_functions_config:
            self.additive_functions.append(globals()[cls](additive_functions_config[cls]))
        self.aggregation_functions = []
        for f_config in aggregation_functions_config:
            self.aggregation_functions.append(globals()[f_config['class']](f_config['config']))
        self.previous_observation = None
        self.reset(observation)

    def reset(self, observation):
        self.previous_observation = copy.deepcopy(observation)
        for pf in self.additive_functions:
            pf.reset()
        for af in self.aggregation_functions:
            af.reset()

    def reward(self, observation, transformed_observation, current_reward, action):
        for aggregation_function in self.aggregation_functions:
            current_reward = aggregation_function(self.previous_observation, observation, current_reward)
        for additive_function in self.additive_functions:
            current_reward += additive_function(observation, transformed_observation, action)
        self.previous_observation = copy.deepcopy(observation)
        return current_reward


class TargetTransformer:
    def __init__(self, target_type, config={}):
        self.target_type = target_type
        self.config = config
        self.seed_offset = config.get("seed_offset", DEFAULT_SEED_OFFSET)
        self.seed_modulo = config.get("seed_modulo", DEFAULT_SEED_MODULO)
        self.use_bad_seeds = config.get("enable_bad_seeds", False) and self.target_type == "random"
        if self.use_bad_seeds:
            self.bad_seeds_set = set()
            self.max_bad_seeds = config.get("max_bad_seeds_count", 256)
            self.bad_steps_count = config.get("bad_seed_steps_border", 700)
            self.bad_reward = config.get("bad_seed_reward_per_step_border", 9.5)
            self._bad_seed = False
            self.bad_targets = {}
        self.target_vel = None
        self.targets = None
        self.step = self.bad_steps_count if self.use_bad_seeds else None
        self.seed = 0

    def transform(self, observation, reward):
        if self.target_type == "random":
            self.target_vel = self.targets[self.step % len(self.targets)]

        if self.target_type == "normal":
            self.target_vel = observation["target_vel"]
        else:
            observation = copy.deepcopy(observation)
            reward += (observation["body_vel"]["pelvis"][0] - observation["target_vel"][0])**2
            reward += (observation["body_vel"]["pelvis"][2] - observation["target_vel"][2])**2
            observation["target_vel"] = self.target_vel
            reward -= (observation["body_vel"]["pelvis"][0] - observation["target_vel"][0])**2
            reward -= (observation["body_vel"]["pelvis"][2] - observation["target_vel"][2])**2
        self.step += 1
        return observation, reward

    def seed_step(self):
        self.set_seed((self.seed + self.seed_offset) % self.seed_modulo)

    def set_seed(self, seed):
        self.seed = seed
        set_seeds(seed)

    def _generate_start_speed(self, speed_range=(.25, 2.), rotation_range=(-math.pi/8, math.pi/8)):
        return random.uniform(*speed_range), random.uniform(*rotation_range)

    def _generate_new_targets(self, poisson_lambda=300, start_speed=1.25,
                              start_rotation=0, change_coef=1,
                              speed_borders=(-.25, 2.25)):
        change_coef *= poisson_lambda / 300

        nsteps = self.config.get("steps", 1000) + 1
        velocity = np.zeros(nsteps)
        heading = np.zeros(nsteps)

        velocity[0] = start_speed
        heading[0] = start_rotation

        change = np.cumsum(np.random.poisson(poisson_lambda, max(2, poisson_lambda // 30)))

        for i in range(1, nsteps):
            velocity[i] = velocity[i-1]
            heading[i] = heading[i-1]

            if i in change:
                velocity[i] += change_coef * random.choice([-1, 1]) * random.uniform(-0.5, 0.5)
                heading[i] += change_coef * random.choice([-1, 1]) * random.uniform(-math.pi/8, math.pi/8)
                velocity[i] = min(max(velocity[i], speed_borders[0]), speed_borders[1])

        trajectory_polar = np.vstack((velocity,heading)).transpose()
        self.targets = np.apply_along_axis(rect, 1, trajectory_polar)

    def reset(self, reward=None):
        if self.use_bad_seeds:
            if (self.step < self.bad_steps_count or (reward is not None and reward / self.step < self.bad_reward)) \
                    and len(self.bad_seeds_set) < self.max_bad_seeds:
                if self._bad_seed is not None:
                    self.bad_seeds_set.add(self._bad_seed)
                    self.bad_targets[self._bad_seed] = self.targets
                else:
                    self.bad_seeds_set.add(self.seed)
                    self.bad_targets[self.seed] = self.targets

            if self._bad_seed is not None:
                self._bad_seed = None
            elif len(self.bad_seeds_set) > 4:
                self._bad_seed = random.sample(self.bad_seeds_set, 1)[0]
                self.targets = self.bad_targets.pop(self._bad_seed)
                self.bad_seeds_set.remove(self._bad_seed)
                return

        self.seed_step()

        self.step = 0
        if self.target_type == "static":
            if self.config.get("constant", True):
                self.target_vel = self.config.get("target_vel", [1.25, 0., 0.])
            else:
                self.target_vel = rect(self._generate_start_speed(
                    self.config.get("speed_range", (.25, 2.)),
                    self.config.get("rotation_range", (-math.pi/8, math.pi/8))
                ))
        elif self.target_type == "random":
            if self.config.get("constant_start", True):
                speed = self.config.get("start_speed", 1.25)
                rotation = self.config.get("start_rotation", 0.)
            else:
                speed, rotation = self._generate_start_speed(
                    self.config.get("start_speed_range", (-.25, 2.25)),
                    self.config.get("start_rotation_range", (-math.pi/8, math.pi/8))
                )
            self._generate_new_targets(self.config.get("poisson_lambda", 300), speed, rotation,
                                       self.config.get("change_coef", 1),
                                       self.config.get("speed_borders", [-.25, 2.25]))


class Transformer:
    def __init__(self, features, observation, transfer=False, vel_acc_coef=100/3, target_noise=0.02):
        self.features = features
        self.transfer = transfer
        self.target_noise = target_noise
        self.vel_acc_coef = vel_acc_coef

        self.pos_relative_features = Relatives('body_{}',
                                               self.features.get("body_pos_relative", []),
                                               transfer=transfer, vel_acc_coef=vel_acc_coef)

        self.rot_relative_features = Relatives('body_{}_rot',
                                               self.features.get("body_rot_relative", []),
                                               transfer=transfer, vel_acc_coef=vel_acc_coef)

        self.mass_center_relative = MassCenter(self.features.get("mass_center_relative", []), vel_acc_coef=vel_acc_coef)
        self.time_from_change = TimeFromChange()

        self.reset(observation)
        self.left_indices = None
        self.right_indices = None
        self.mirror_indices = None
        self._init_complement_indices(observation)

    def reset(self, observation):
        self.pos_relative_features.reset(observation)
        self.rot_relative_features.reset(observation)
        self.mass_center_relative.reset(observation)
        self.time_from_change.reset(observation)

    def transform(self, observation):
        self.pos_relative_features.update(observation)
        self.rot_relative_features.update(observation)
        self.mass_center_relative.update(observation)
        self.time_from_change.update(observation)
        pos_relatives = self.pos_relative_features.to_numpy()
        rot_relatives = self.rot_relative_features.to_numpy()
        mass_center_relatives = self.mass_center_relative.to_numpy()

        raw_features = []

        for raw_feature in sorted(self.features.get('raw_features', [])):
            if raw_feature == 'forces':
                forces = Forces(observation[raw_feature])
                raw_features.append(forces.to_numpy())
            elif raw_feature == 'muscles':
                muscles = Muscles(observation[raw_feature])
                raw_features.append(muscles.to_numpy())
            elif raw_feature == 'heights':
                heights = Heights(observation)
                raw_features.append(heights.to_numpy())
            elif raw_feature == 'activations':
                activations = Activations(observation)
                raw_features.append(activations.to_numpy())
            else:
                raw_feature = Plainer(observation[raw_feature])
                raw_features.append(raw_feature.to_numpy())

        if self.features.get('target_vel', False):
            raw_features.append(TargetVelocity(observation, self.target_noise).to_numpy())

        if self.features.get('time_from_change', False):
            raw_features.append(self.time_from_change.to_numpy())

        for feature in self.features.get("raw_height"):
            raw_features.append(np.array([observation["body_pos"][feature][1]]))

        for feature in self.features.get("raw_vel"):
            raw_features.append(np.array(observation["body_vel"][feature]) / self.vel_acc_coef)
            raw_features.append(np.array(observation["body_acc"][feature]) / self.vel_acc_coef)

        for feature in self.features.get("raw_rot"):
            raw_features.append(np.array(observation["body_pos_rot"][feature]))
            raw_features.append(np.array(observation["body_vel_rot"][feature]) / self.vel_acc_coef)
            raw_features.append(np.array(observation["body_acc_rot"][feature]) / self.vel_acc_coef)

        cur_observation = np.concatenate([pos_relatives,
                                          rot_relatives,
                                          mass_center_relatives,
                                          *raw_features])

        return cur_observation.astype(np.float32)

    def _init_complement_indices(self, observation):
        left_indices = []
        right_indices = []
        names_list = []

        for field in (self.pos_relative_features, self.rot_relative_features, self.mass_center_relative):
            left_indices.append(field.get_left_indices())
            right_indices.append(field.get_right_indices())
            names_list.append(field.get_names())

        left_indices = list(filter(lambda lst: len(lst) > 0, left_indices))
        right_indices = list(filter(lambda lst: len(lst) > 0, right_indices))

        cum_index = len(names_list[-1])

        def append_indices(feature):
            nonlocal cum_index
            left_indices.append(feature.get_left_indices() + cum_index)
            right_indices.append(feature.get_right_indices() + cum_index)
            names_list.append(feature.get_names())

            cum_index += len(names_list[-1])

        for raw_feature in sorted(self.features.get('raw_features', [])):
            if raw_feature == 'forces':
                forces = Forces(observation[raw_feature])
                append_indices(forces)
            elif raw_feature == 'muscles':
                muscles = Muscles(observation[raw_feature])
                append_indices(muscles)
            elif raw_feature == 'heights':
                heights = Heights(observation)
                append_indices(heights)
            elif raw_feature == 'activations':
                activations = Activations(observation)
                append_indices(activations)
            else:
                raw_feature = Plainer(observation[raw_feature], raw_feature)
                append_indices(raw_feature)

        if self.features.get('target_vel', False):
            append_indices(TargetVelocity(observation))

        self.left_indices = np.concatenate(left_indices)
        self.right_indices = np.concatenate(right_indices)

        flatten_names = []
        for names in names_list:
            flatten_names.extend(names)

        self.mirror_indices = []

        for i, name in enumerate(flatten_names):
            if 'z' in name:
                self.mirror_indices.append(i)

        self.mirror_indices = np.array(self.mirror_indices)

    def mirror_legs(self, observation):
        mirrored_observation = np.copy(observation)
        left = mirrored_observation[self.left_indices]
        right = mirrored_observation[self.right_indices]
        mirrored_observation[self.left_indices] = right
        mirrored_observation[self.right_indices] = left
        mirrored_observation[self.mirror_indices] = -mirrored_observation[self.mirror_indices]
        return mirrored_observation



