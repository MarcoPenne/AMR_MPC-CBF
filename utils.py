import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm

def transformProj2Orig(s, l, theta_tilde, path):
    
    X = np.zeros_like(s)
    Y = np.zeros_like(s)
    THETA = np.zeros_like(s)
    for i in range(len(s)):
        (x1, y1) = path(s[i])
        theta_r = path.get_theta_r(s[i])
        
        x = x1 - np.sin(theta_r)*l[i]
        y = y1 + np.cos(theta_r)*l[i]
        
        theta = theta_tilde[i] + theta_r
        #print(x, y, theta)
        X[i] = x
        Y[i] = y
        THETA[i] = theta
    return (X, Y, THETA)

def savePlot(x, y, theta, v, w, X_horizon, folder, i, car_model, path):
    plt.figure()
    plt.xlim((-4, 14))
    plt.ylim((-2, 12))
    a = plt.gca()
    a.set_aspect('equal')
    #plt.plot(x, y, 'bo')

    drawPath(path)
    right_side = a.spines["right"]
    right_side.set_visible(False)
    #print(plt.axes().spines)
    top_side = a.spines["top"]
    top_side.set_visible(False)
    h = car_model.l1
    h2 = car_model.l2
    half_edge = car_model.l2/2
    t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta), y+ (1/2)*h*np.sin(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='blue')
    plt.gca().add_patch(t1)

    obs = [6., 0.1, 0.]
    obs = transformProj2Orig([obs[0]], [obs[1]], [obs[2]], path)
    x = obs[0][0]
    y = obs[1][0]
    theta = obs[2][0]
    t2 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t2)
    t2 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t2)

    #for i in range(X_horizon.shape[0]):
    (_x, _y, _theta) = transformProj2Orig(X_horizon[:,0], X_horizon[:,1], X_horizon[:,2], path)
    plt.plot(_x, _y, '-r', linewidth=0.5)
    
    # this is an inset axes over the main axes
    a = plt.axes([.48, .78, .2, .2], facecolor='y')
    a.set_ylabel('v')
    plt.plot(v)
    plt.ylim((0, 4))
    plt.xticks([])
    #plt.yticks([])

    a = plt.axes([.78, .78, .2, .2], facecolor='y')
    a.set_ylabel('omega')
    plt.plot(w)
    plt.ylim((-2, 2))
    plt.xticks([])
    
    plt.savefig('results/' + folder + "/%04d" % i +".png")

    #plt.show()
    plt.close()
    
def drawPath(path):
    samples = np.arange(0., path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [path(s)]

    x = [c[0] for c in coord]
    y = [c[1] for c in coord]

    plt.plot(x, y, '-y', linewidth=0.5)


def renderVideo(simX, simU, simX_horizon, t, car_model, path, folder):
    # load track
    s=simX[:,0]
    l=simX[:,1]
    theta_tilde=simX[:,2]

    period = np.mean(np.diff(t))
    fr = int(np.around(1/period, decimals=0))
    
    # transform data
    (x, y, theta) = transformProj2Orig(s, l, theta_tilde, path)
    
    # plot racetrack map
    for i in tqdm(range(len(x)), desc="Creating frames of video"):
        v = simU[:i, 0]
        w = simU[:i, 1]
        input_len = 40
        if len(v)>input_len:
            v = v[-input_len:]
            w = w[-input_len:]

        savePlot(x[i], y[i], theta[i], v, w, simX_horizon[i, :, :],folder, i, car_model, path)
        #plt.show()
    os.chdir('results/' + folder)
    os.system(f"ffmpeg -framerate {fr}"+" -i %04d.png -r 30 -pix_fmt yuv420p video.mp4")
    for i in tqdm(range(len(x)), desc="Removing temp files"):
        os.system('rm %04d.png' %i)
    os.chdir('../..')