import abc
import copy
from collections import OrderedDict
from itertools import cycle

import numpy as np

MAX_ITER = 300

FORCE_DIVIDER = 1024.0


class FeaturesBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.left_indices = None
        self.right_indices = None
        self.names = None

    def get_left_indices(self):
        self.get_names()
        if not self.left_indices:
            self._init_complement_indices()
        return self.left_indices

    def get_right_indices(self):
        self.get_names()
        if self.left_indices is None:
            self._init_complement_indices()
        return self.right_indices

    def _init_complement_indices(self):
        self.left_indices = []
        self.right_indices = []

        for index, name in enumerate(self.names):
            if name[-1] == 'l':
                try:
                    right_index = self.names.index(name[:-1] + 'r')
                    self.left_indices.append(index)
                    self.right_indices.append(right_index)
                except ValueError:
                    continue
        self.left_indices = np.array(self.left_indices)
        self.right_indices = np.array(self.right_indices)

    def get_names(self):
        if not self.names:
            self._create_names()
        return self.names

    @abc.abstractmethod
    def _create_names(self):
        pass

    @abc.abstractmethod
    def to_numpy(self):
        pass


class Plainer(FeaturesBase):
    def __init__(self, raw_data, prefix=''):
        super(Plainer, self).__init__()

        self.raw_data = copy.deepcopy(raw_data)
        self.prefix = prefix
        self._create_names()

        for raw_part in self.raw_data:
            self.raw_data[raw_part] = np.array(self.raw_data[raw_part])

    def to_numpy(self):
        arrays = [self.raw_data[raw_part] for raw_part in sorted(self.raw_data.keys())]
        return np.concatenate(arrays)

    def _create_names(self):
        self.names = []
        for raw_part in sorted(self.raw_data.keys()):
            for _, coor_name in zip(range(len(self.raw_data[raw_part])), cycle(('x', 'y', 'z'))):
                self.names.append('{}_{}_{}'.format(self.prefix, coor_name, raw_part))


class Forces(Plainer):
    def __init__(self, raw_data, prefix=''):
        super(Plainer, self).__init__()

        self.raw_data = copy.deepcopy(raw_data)
        self.prefix = prefix
        self._create_names()

        for raw_part in self.raw_data:
            self.raw_data[raw_part] = np.array(self.raw_data[raw_part]) / FORCE_DIVIDER

    def _create_names(self):
        self.names = []
        for raw_part in sorted(self.raw_data.keys()):
            for i in range(len(self.raw_data[raw_part])):
                self.names.append('{}_{}'.format(i, raw_part)) 


class RelativePlainer(FeaturesBase):
    def __init__(self, raw_body_pos, basis, transfer=False):
        super(RelativePlainer, self).__init__()

        self.body_pos = copy.deepcopy(raw_body_pos)

        if transfer:
            self.body_pos['tibia_r'] = list(self.body_pos['pros_tibia_r'])
            self.body_pos['talus_r'] = list(self.body_pos['pros_foot_r'])
            self.body_pos['calcn_r'] = list(self.body_pos['pros_foot_r'])
            self.body_pos['toes_r'] = list(self.body_pos['pros_foot_r'])
            del self.body_pos['pros_tibia_r']
            del self.body_pos['pros_foot_r']

        self.basis = basis
        self.body_parts = sorted(self.body_pos.keys())

        basis_pos = np.array(self.body_pos[basis])

        for body_part in self.body_parts:
            self.body_pos[body_part] = np.array(self.body_pos[body_part])
            self.body_pos[body_part] -= basis_pos

    def _create_names(self):
        self.names = []
        for body_part in self.body_parts:
            for _, coor_name in zip(range(len(self.body_pos[body_part])), cycle(('x', 'y', 'z'))):
                self.names.append('{}_rel_to_{}_{}'.format(coor_name, self.basis, body_part))

    def to_numpy(self):
        arrays = [self.body_pos[body_part] for body_part in self.body_parts]
        return np.concatenate(arrays)


class Activations(FeaturesBase):
    def __init__(self, observation):
        super(Activations, self).__init__()
        self.raw_muscles = copy.deepcopy(observation['muscles'])

        muscle_names = sorted(self.raw_muscles.keys())
        self.muscles = []
        self._create_names()

        for muscle_name in muscle_names:
            muscle_features = self.raw_muscles[muscle_name]
            # feature_names = sorted(muscle_features.keys())
            self.muscles.append(muscle_features['activation'])
            self.names.append('activation_{}'.format(muscle_name))

    def to_numpy(self):
        return np.array(self.muscles)

    def _create_names(self):
        self.names = []


class Muscles(FeaturesBase):
    def __init__(self, raw_muscles):
        super(Muscles, self).__init__()
        self.raw_muscles = copy.deepcopy(raw_muscles)

        muscle_names = sorted(raw_muscles.keys())
        self.muscles = []
        self._create_names()

        for muscle_name in muscle_names:
            muscle_features = raw_muscles[muscle_name]
            # feature_names = sorted(muscle_features.keys())
            feature_names = ('activation', 'fiber_length', 'fiber_force')
            for feature_name in feature_names:
                if feature_name == 'fiber_force':
                    self.muscles.append(muscle_features[feature_name] / FORCE_DIVIDER)
                else:
                    self.muscles.append(muscle_features[feature_name])
                self.names.append('{}_{}'.format(feature_name, muscle_name))

    def to_numpy(self):
        return np.array(self.muscles)

    def _create_names(self):
        self.names = []


class Heights(FeaturesBase):
    def __init__(self, observation):
        super(Heights, self).__init__()
        self.heights = []
        self._create_names()

        feature_names = ('body_pos', 'body_vel', 'body_acc')

        body_parts = sorted(observation[feature_names[0]].keys())

        for feature_name in feature_names:
            feature = observation[feature_name]
            for body_part in body_parts:
                self.heights.append(feature[body_part][1])
                self.names.append('height_{}_{}'.format(feature_name, body_part))

    def _create_names(self):
        self.names = []

    def to_numpy(self):
        return np.array(self.heights)


class MassCenter(FeaturesBase):
    def __init__(self, relative_against, vel_acc_coef=100/3):
        super(MassCenter, self).__init__()

        self.vel_acc_coef = vel_acc_coef
        self.relative_against = relative_against
        self.relatives = {
            "vel": OrderedDict(),
            "pos": OrderedDict(),
            "acc": OrderedDict()
        }

        self._create_names()

    def reset(self, observation):
        self.update(observation)

    def update(self, observation):
        for feature in ("vel", "pos", "acc"):
            for body_part in self.relative_against:
                limb = np.array(observation['body_{}'.format(feature)][body_part])
                mass_center = np.array(observation['misc']['mass_center_{}'.format(feature)])
                self.relatives[feature][body_part] = limb - mass_center
        for feature in ("vel", "acc"):
            for body_part in self.relative_against:
                self.relatives[feature][body_part] /= self.vel_acc_coef

    def _create_names(self):
        self.names = []

        coors = ('x', 'y', 'z')
        features = ('pos', 'vel', 'acc')

        for body_part in self.relative_against:
            for feature in features:
                for coor in coors:
                    self.names.append('mass_center_{}_{}_{}'.format(body_part, feature, coor))

    def to_numpy(self):
        if len(self.relative_against) == 0:
            return np.empty((0,))

        features = []

        for body_part in self.relative_against:
            for feature in ("pos", "vel", "acc"):
                features.append(self.relatives[feature][body_part])

        return np.concatenate(features)


class TimeFromChange(FeaturesBase):
    def __init__(self, mean=300, epsilon=0.001):
        super().__init__()
        self.mean = mean
        self.epsilon = epsilon
        self.target_vel = None
        self.steps = None
        self._create_names()

    def _create_names(self):
        self.names = ['time_from_last_change']

    def update(self, observation):
        self.steps += 1
        if abs(self.target_vel[0] - observation["target_vel"][0]) < self.epsilon or \
           abs(self.target_vel[0] - observation["target_vel"][0]) < self.epsilon:
            self.reset(observation)

    def reset(self, observation):
        self.steps = 0
        self.target_vel = copy.deepcopy(observation["target_vel"])

    def to_numpy(self):
        return np.array([self.steps / self.mean])


class TargetVelocity(FeaturesBase):
    def __init__(self, observation, noise_strength=0., vel_acc_coef=100./3):
        super(TargetVelocity, self).__init__()

        self._create_names()
        self.target_vels = np.array(observation['target_vel'])
        # replacing y part (zero all the time) with length of the velocity vector
        noise = np.random.uniform(-1., 1., 2)
        noise = np.array([noise[0], 0., noise[1]])
        noise *= np.linalg.norm(observation["target_vel"]) * noise_strength / np.linalg.norm(noise)
        self.target_vels += noise
        self.target_vels /= vel_acc_coef

        self.target_vels[1] = np.sqrt(self.target_vels[0]**2 + self.target_vels[2]**2)

    def _initialize(self, observation):
        pass

    def _create_names(self):
        self.names = ['target_velocity_{}'.format(coor) for coor in ('x', 'y', 'z')]

    def to_numpy(self):
        return self.target_vels


class Relatives(FeaturesBase):
    def __init__(self, relatives_type, relative_against, features=('pos', 'vel', 'acc'),
                 transfer=False, vel_acc_coef=100./3):
        super(Relatives, self).__init__()
        self.relatives_type = relatives_type
        self.vel_acc_coef = vel_acc_coef
        self.features = features
        self.transfer = transfer
        self.relative_against = sorted(relative_against)
        self.per_type_names = []
        self.relatives = []

    def reset(self, observation):
        self.update(observation)

    def to_numpy(self):
        if len(self.relatives) == 0:
            return np.empty((0,))
        return np.concatenate(self.relatives)

    def update(self, observation):
        relatives = self._compute_relatives(observation)
        relatives = self._process_relatives(relatives)
        self.relatives = relatives

    def _create_names(self):
        self.names = []
        for relative_type in self.features:
            relative_type_feature_names = ['{}_{}_{}'.format(self.relatives_type, relative_type, name)
                                           for name in self.per_type_names]
            self.names += relative_type_feature_names

    def _compute_relatives(self, observation):
        result = {
            'pos': OrderedDict(),
            'vel': OrderedDict(),
            'acc': OrderedDict()
        }

        for feature in ('pos', 'vel', 'acc'):
            for body_part in self.relative_against:
                result[feature][body_part] = RelativePlainer(observation[self.relatives_type.format(feature)],
                                                             basis=body_part, transfer=self.transfer).to_numpy()
        for feature in ('vel', 'acc'):
            for body_part in self.relative_against:
                result[feature][body_part] /= self.vel_acc_coef
        result = [result[relative_type] for relative_type in self.features]

        return result

    @staticmethod
    def _process_relatives(relatives):

        def plain_feature_dict(feature_dict):
            features = [feature_dict[feature] for feature in feature_dict]
            return features

        processed_relatives = [plain_feature_dict(relative) for relative in relatives]
        flatten_relatives = []

        for processed_relative in processed_relatives:
            flatten_relatives += processed_relative

        return flatten_relatives
