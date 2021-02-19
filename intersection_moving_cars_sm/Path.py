import math
import matplotlib.pyplot as plt
import numpy as np

class Path:

    def __init__(self, l1, l2, r, traslx=0, trasly=0):
        self.l1 = l1
        self.l2 = l2
        self.r = r
        self.traslx = traslx
        self.trasly = trasly


    def __call__(self, s):
        s = float(s)
        completed_lap = 0
        #print(s)
        while s >= 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            completed_lap += 1
            s = s - (2*self.l1 + 2*self.r*math.pi + 2*self.l2)
        while s<0:
            s = s + (2*self.l1 + 2*self.r*math.pi + 2*self.l2)
        # first edge
        if s>=0 and s<self.l1:
            return (s + self.traslx , 0. + self.trasly)
        # first arc
        elif s>=self.l1 and s< self.l1 + 2*self.r*math.pi/4:
            arc = s - self.l1
            alpha = arc/self.r
            return (self.l1 + self.r*math.sin(alpha) + self.traslx , self.r - self.r*math.cos(alpha) + self.trasly)
        # second edge
        elif s>= self.l1 + 2*self.r*math.pi/4 and s< self.l1 + 2*self.r*math.pi/4 + self.l2:
            return (self.l1 + self.r  + self.traslx , s - self.l1 - 2*self.r*math.pi/4 + self.r + self.trasly)
        # second arc
        elif s>= self.l1 + 2*self.r*math.pi/4 + self.l2 and s < self.l1 + self.r*math.pi + self.l2:
            arc = s - (self.l1 + 2*self.r*math.pi/4 + self.l2)
            alpha = arc/self.r
            return (self.l1 + math.cos(alpha)*self.r + self.traslx , self.r + self.l2 + math.sin(alpha)*self.r + self.trasly)
        # third edge
        elif s>= self.l1 + self.r*math.pi + self.l2 and s < 2*self.l1 + self.r*math.pi + self.l2:
            return (self.l1 - (s - self.l1 - self.r*math.pi - self.l2) + self.traslx , 2*self.r+self.l2 + self.trasly)
        # third arc
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 and s< 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4:
            arc = s - (2*self.l1 + self.r*math.pi + self.l2)
            alpha = arc/self.r
            return (-math.sin(alpha)*self.r + self.traslx , self.r+ self.l2+self.r*math.cos(alpha) + self.trasly)
        # fourth edge
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4:
            return (-self.r + self.traslx , self.r+ self.l2 - (s-(2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4)) + self.trasly)
        # fourth arc
        elif s>= 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            arc = s - (2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4)
            alpha = arc/self.r
            return (-self.r*math.cos(alpha) + self.traslx , self.r- self.r*math.sin(alpha) + self.trasly)
        else:
            Exception("Sbagliato controlla!")

    def get_cartesian_coords(self, s, l):

        res = self(s)
        (x1, y1) = res
        theta_r = self.get_theta_r(s)
        x = x1 - np.sin(theta_r)*l
        y = y1 + np.cos(theta_r)*l
        return (x, y)

    def get_k(self, s):
        s = float(s)
        completed_lap = 0
        while s >= 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            completed_lap += 1
            s = s - (2*self.l1 + 2*self.r*math.pi + 2*self.l2)
        while s<0:
            s = s + (2*self.l1 + 2*self.r*math.pi + 2*self.l2)

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
        while s<0:
            completed_lap -= 1
            s = s + (2*self.l1 + 2*self.r*math.pi + 2*self.l2)

        # first edge
        if s>=0 and s<self.l1:
            return 0. + completed_lap*2*np.pi
        # first arc
        elif s>=self.l1 and s< self.l1 + 2*self.r*math.pi/4:
            arc = s - self.l1
            alpha = arc/self.r
            return alpha + completed_lap*2*np.pi
        # second edge
        elif s>= self.l1 + 2*self.r*math.pi/4 and s< self.l1 + 2*self.r*math.pi/4 + self.l2:
            return math.pi/2 + completed_lap*2*np.pi
        # second arc
        elif s>= self.l1 + 2*self.r*math.pi/4 + self.l2 and s < self.l1 + self.r*math.pi + self.l2:
            arc = s - (self.l1 + 2*self.r*math.pi/4 + self.l2)
            alpha = arc/self.r
            return math.pi/2+alpha + completed_lap*2*np.pi
        # third edge
        elif s>= self.l1 + self.r*math.pi + self.l2 and s < 2*self.l1 + self.r*math.pi + self.l2:
            return math.pi + completed_lap*2*np.pi
        # third arc
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 and s< 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4:
            arc = s - (2*self.l1 + self.r*math.pi + self.l2)
            alpha = arc/self.r
            return math.pi+alpha + completed_lap*2*np.pi
        # fourth edge
        elif s>= 2*self.l1 + self.r*math.pi + self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4:
            return (3/2)*math.pi + completed_lap*2*np.pi
        # fourth arc
        elif s>= 2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4 and s< 2*self.l1 + 2*self.r*math.pi + 2*self.l2:
            arc = s - (2*self.l1 + self.r*math.pi + 2*self.l2 + 2*self.r*math.pi/4)
            alpha = arc/self.r
            return (3/2)*math.pi + alpha + completed_lap*2*np.pi
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