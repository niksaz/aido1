import gym
from gym import spaces
import numpy as np
from collections import deque


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)


class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(GrayscaleWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            0.0,
            1.0,
            (obs_shape[0], obs_shape[1], 1),
            dtype=np.float32)

    def observation(self, observation):
        # print('GrayscaleWrapper', observation[:,:,0])
        from skimage import color
        return color.rgb2gray(observation).reshape(self.observation_space.shape)


class ImgTransposer(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgTransposer, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        # print('ImgTransposer', observation[:,:,0])
        return observation.transpose(2, 0, 1)


class ImgStacker(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgStacker, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)
        self.deque = deque(maxlen=3)

    def observation(self, observation):
        dq = self.deque
        if dq:
            dq.pop()
        while len(dq) < dq.maxlen:
            dq.appendleft(observation)
        res = np.concatenate((dq[0], dq[1], dq[2]), axis=0)
        return res

    def reset(self):
        self.deque.clear()
        return super(ImgStacker, self).reset()


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self,
                 env,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102
                 ):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist

    def action(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels
