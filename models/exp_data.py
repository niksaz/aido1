import pandas as pd
import numpy as np


class GateAgent:
    muscles = ['Rectus_Femoris',
               'Medial_Hamstrings',
               'Lateral_Hamstrings',
               'Anterior_Tibialis',
               'Medial_Gastrocnemius',
               'Iliopsoas'
               ]

    muscle_to_index = {
        'Rectus_Femoris': 6,
        'Medial_Hamstrings': 2,
        'Lateral_Hamstrings': 3,
        'Anterior_Tibialis': 10,
        'Medial_Gastrocnemius': 8,
        'Iliopsoas': 5
    }

    multiplier = 1.0

    def __init__(self, gate_name='VerySlowUB', repeats=1, miss_rate=3, legs_offset=25):
        self.gate_name = gate_name
        data = pd.read_table('emg_data.txt')
        self.emg_data = data[['Muscles', 'PercentGaitCycle', self.gate_name]]

        self.muscle_to_emg = {}

        for muscle_name in GateAgent.muscles:
            muscle_emg_data = self.emg_data[self.emg_data['Muscles'] == muscle_name]
            self.muscle_to_emg[muscle_name] = muscle_emg_data[self.gate_name].values.astype('float32')

        self.counter = 0
        self.repeats = repeats - 1
        self.repeat_counter = 0
        self.miss_rate = miss_rate
        self.legs_offset = legs_offset

    def act(self, observation, step):
        second_leg = step > 135
        cur_action = [0] * 22
        for muscle_name in GateAgent.muscles:
            emg = self.muscle_to_emg[muscle_name]
            index = GateAgent.muscle_to_index[muscle_name]
            cur_action[index] = max(0, min(1, emg[self.counter] * GateAgent.multiplier))
            if second_leg:
                cur_action[11 + index] = max(0, min(1, emg[(self.counter + self.legs_offset) % 51] * GateAgent.multiplier))

        self.repeat_counter += 1
        if self.repeat_counter > self.repeats:
            self.repeat_counter = 1
            self.counter += self.miss_rate

        if self.counter >= 51:
            self.counter = 0
        # hardcoded

        cur_action[11 + 0] = 0.5
        cur_action[11 + 1] = 0.3

        if step < 100:
            action = [0] * 22
        else:
            action = cur_action

        action[10] = 1.0
        action[11 + 10] = action[10]

        action[6] = 0.2
        action[11 + 6] = action[6]

        if step > 80:
            action[5] = 0.01 * (step - 80)
            action[11 + 5] = action[5]
        if step > 120:
            action[5] = 0.5 + 0.01 * (step - 120)
            action[11 + 5] = action[5]
            action[11 + 0] = 0.7

            if step > 140:
                action[5] = 0.1
                action[11 + 5] += 0.5
                action[7] = 0.5 + 0.02 * (step - 140)
                action[6] = 0.5 + 0.02 * (step - 140)
                action[10] = 1.0 - action[9] * 0.5

                if step < 150:

                    action[11 + 7] = 0.5 + 0.01 * (step - 135)
                    action[11 + 6] = 0.5 + 0.01 * (step - 135)

                    action[11 + 8] = 0.5 + 0.01 * (step - 135)
                    action[11 + 9] = 0.5 + 0.01 * (step - 135)

                    action[11 + 10] = 1.0 - action[11 + 9] * 0.5
                else:
                    action[4] = 0.7 + 0.02 * (step - 150)
                    action[1] = 0.3

                if step > 170:
                    action[5] = 0.3
                    action[4] = 0.3

                if step > 180:
                    action[6] = 0.3
                    action[7] = 0.3

                action[0] = 0.3

        return np.array(action, dtype=np.float32)