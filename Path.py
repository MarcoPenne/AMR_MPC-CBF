import math
import matplotlib.pyplot as plt
import numpy as np

class Path:

    def __init__(self, l1, l2, r):
        self.l1 = l1
        self.l2 = l2
        self.r = r


    def __call__(self, s):
        s = float(s)
        completed_lap = 0
        while s >= 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            completed_lap += 1
            s = s - (2*self.l1 + 2*self.r*math.pi + 2*self.l2)

        # first edge
        if s>=0 and s<self.l1:
            return (s, 0)
        # first arc
        elif s>=self.l1 and s< self.l1 + 2*self.r*math.pi/4:
            arc = s - self.l1
            alpha = arc/self.r
            return (self.l1 + self.r*math.sin(alpha), self.r - self.r*math.cos(alpha))
        # second edge
        elif s>= self.l1 + 2*self.r*math.pi/4 and s< self.l1 + 2*self.r*math.pi/4 + self.l2:
            return (self.l1 + self.r , s - self.l1 - 2*self.r*math.pi/4 + self.r)
        # second arc
        elif s>= self.l1 + 2*self.r*math.pi/4 + self.l2 and s < self.l1 + self.r*math.pi + self.l2:
            arc = s - (self.l1 + 2*self.r*math.pi/4 + self.l2)
            alpha = arc/self.r
            return (self.l1 + math.cos(alpha)*self.r, self.r + self.l2 + math.sin(alpha)*self.r)
        # third edge
        elif s>= self.l1 + self.r*math.pi + self.l2 and s < 2*self.l1 + self.r*math.pi + self.l2:
            return (self.l1 - (s - self.l1 - self.r*math.pi - self.l2), 2*self.r+self.l2)
        # third arc
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 and s< 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4:
            arc = s - (2*self.l1 + self.r*math.pi + self.l2)
            alpha = arc/self.r
            return (-math.sin(alpha)*self.r, self.r+ self.l2+self.r*math.cos(alpha))
        # fourth edge
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4:
            return (-self.r, self.r+ self.l2 - (s-(2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4)))
        # fourth arc
        elif s>= 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            arc = s - (2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4)
            alpha = arc/self.r
            return (-self.r*math.cos(alpha), self.r- self.r*math.sin(alpha))
        else:
            Exception("Sbagliato controlla!")

    def get_k(self, s):
        s = float(s)
        completed_lap = 0
        while s >= 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            completed_lap += 1
            s = s - (2*self.l1 + 2*self.r*math.pi + 2*self.l2)

        # first edge
        if s>=0 and s<self.l1:
            return 0.
        # first arc
        elif s>=self.l1 and s< self.l1 + 2*self.r*math.pi/4:
            return 1./self.r
        # second edge
        elif s>= self.l1 + 2*self.r*math.pi/4 and s< self.l1 + 2*self.r*math.pi/4 + self.l2:
            return 0.
        # second arc
        elif s>= self.l1 + 2*self.r*math.pi/4 + self.l2 and s < self.l1 + self.r*math.pi + self.l2:
            return 1./self.r
        # third edge
        elif s>= self.l1 + self.r*math.pi + self.l2 and s < 2*self.l1 + self.r*math.pi + self.l2:
            return 0.
        # third arc
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 and s< 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4:
            return 1./self.r
        # fourth edge
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4:
            return 0.
        # fourth arc
        elif s>= 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            return 1./self.r
        else:
            Exception("Sbagliato controlla!")

    def get_theta_r(self, s):
        s = float(s)
        completed_lap = 0
        while s >= 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            completed_lap += 1
            s = s - (2*self.l1 + 2*self.r*math.pi + 2*self.l2)

        # first edge
        if s>=0 and s<self.l1:
            return 0.
        # first arc
        elif s>=self.l1 and s< self.l1 + 2*self.r*math.pi/4:
            arc = s - self.l1
            alpha = arc/self.r
            return alpha
        # second edge
        elif s>= self.l1 + 2*self.r*math.pi/4 and s< self.l1 + 2*self.r*math.pi/4 + self.l2:
            return math.pi/2
        # second arc
        elif s>= self.l1 + 2*self.r*math.pi/4 + self.l2 and s < self.l1 + self.r*math.pi + self.l2:
            arc = s - (self.l1 + 2*self.r*math.pi/4 + self.l2)
            alpha = arc/self.r
            return math.pi+alpha
        # third edge
        elif s>= self.l1 + self.r*math.pi + self.l2 and s < 2*self.l1 + self.r*math.pi + self.l2:
            return math.pi
        # third arc
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 and s< 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4:
            arc = s - (2*self.l1 + self.r*math.pi + self.l2)
            alpha = arc/self.r
            return math.pi+alpha
        # fourth edge
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4:
            return (3/2)*math.pi
        # fourth arc
        elif s>= 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            arc = s - (2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4)
            alpha = arc/self.r
            return (3/2)*math.pi + alpha
        else:
            Exception("Sbagliato controlla!")

    def get_len(self):
        return 2*self.l1 + 2*self.r*math.pi + 2*self.l2


#path = Path(10, 5, 2)
#print(path.get_theta_r(10+2*2*math.pi/4))
# samples = np.arange(0., 70., 0.1)
# print(samples)
# coord = []
# for s in samples:
#     coord += [path(s)]

# x = [c[0] for c in coord]
# y = [c[1] for c in coord]

# plt.xlim((-3, 13))
# plt.ylim((-3, 13))
# plt.plot(x, y, 'bo')
# plt.show()