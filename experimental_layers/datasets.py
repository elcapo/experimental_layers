import numpy as np

class RandomSinusoid():
    def __init__(self):
        self.angular_velocity = np.random.rand(1)[0] * np.random.randint(1, 5)
        self.phase = np.random.rand(1)[0] * 10
    
    def get_half_period(self):
        return np.pi / self.angular_velocity
    
    def get(self, from_period_point = 0, to_period_point = 0.5, step = 1./25):
        x = np.arange(
            from_period_point * self.get_half_period(),
            to_period_point * self.get_half_period(),
            step, dtype=float)
        y = np.cos(self.angular_velocity * x + self.phase)
        return x, y