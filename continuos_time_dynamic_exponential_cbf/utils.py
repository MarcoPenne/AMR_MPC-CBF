import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm
from Path import Path

def transformProj2Orig(s, l, theta_tilde, path):
    
    X = np.zeros_like(s)
    Y = np.zeros_like(s)
    THETA = np.zeros_like(s)
    for i in range(len(s)):
        #print(s[i])
        res = path(s[i])
        #print(res)
        (x1, y1) = res
        theta_r = path.get_theta_r(s[i])
        
        x = x1 - np.sin(theta_r)*l[i]
        y = y1 + np.cos(theta_r)*l[i]
        
        theta = theta_tilde[i] + theta_r
        #print(x, y, theta)
        X[i] = x
        Y[i] = y
        THETA[i] = theta
    return (X, Y, THETA)

def savePlot(x, y, theta, v, w, X_horizon, folder, i, car_model, fixed_obstacles, moving_obstacles, path, h_cbf):
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
    t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='blue')
    plt.gca().add_patch(t1)
    t1 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='blue')
    plt.gca().add_patch(t1)
    t1 = t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta), y+ (1/2)*h*np.sin(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='red')
    plt.gca().add_patch(t1)
    
    if fixed_obstacles is not None:
        for o in range(fixed_obstacles.shape[0]):
            obs = fixed_obstacles[o, :]
            drawObstacles(obs, path, car_model, h_cbf)

    for o in range(moving_obstacles.shape[0]):
        obs = moving_obstacles[o, :3]
        drawObstacles(obs, path, car_model, h_cbf)

    #for i in range(X_horizon.shape[0]):
    (_x, _y, _theta) = transformProj2Orig(X_horizon[:,0], X_horizon[:,1], X_horizon[:,2], path)
    plt.plot(_x, _y, '-r', linewidth=0.5)
    
    # this is an inset axes over the main axes
    a = plt.axes([.48, .78, .2, .2], facecolor='y')
    a.set_ylabel(r'$a$')
    plt.plot(v)
    plt.ylim((-4, 4))
    plt.xticks([])
    #plt.yticks([])

    a = plt.axes([.78, .78, .2, .2], facecolor='y')
    a.set_ylabel(r'$a_{\omega}$')
    plt.plot(w)
    plt.ylim((-2, 2))
    plt.xticks([])
    
    plt.savefig('results/' + folder + "/%04d" % i +".png")
    if i==0:
        plt.savefig('results/' + folder + "/%04d" % i +".eps")
    #plt.show()
    plt.close()

def drawObstacles(obs, path, car_model, h_cbf):
    s_obs = obs[0]
    l_obs = obs[1]
    theta_obs = obs[2]
    obs = transformProj2Orig([obs[0]], [obs[1]], [obs[2]], path)
    h = car_model.l1
    h2 = car_model.l2
    half_edge = car_model.l2/2

    x = obs[0][0]
    y = obs[1][0]
    theta = obs[2][0]
    t2 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t2)
    t2 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t2)

    delta = 0.1
    srange = np.arange(0., path.get_len()*3, delta)
    lrange = np.arange(-2, 2, delta)
    S, L = np.meshgrid(srange,lrange)

    F = (S-s_obs)**4/h**4 + (L-l_obs)**4/h2**4
    X1, Y1 = np.vectorize(path)(S)
    THETA_R = np.vectorize(path.get_theta_r)(S)
    X = X1 - np.sin(THETA_R)*L
    Y = Y1 + np.cos(THETA_R)*L
    #F = ((np.cos(-theta)*(X-x)-np.sin(-theta)*(Y-y))**4 / h**4 ) + ((np.sin(-theta)*(X-x)+np.cos(-theta)*(Y-y))**4 / h2**4)
    plt.contour(X, Y, (F), [h_cbf], linestyles='dashed', linewidths=0.5, colors='blue')
    
def drawPath(path):
    samples = np.arange(0., path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [path(s)]

    x = [c[0] for c in coord]
    y = [c[1] for c in coord]

    plt.plot(x, y, '-y', linewidth=1.)

    inner_path = Path(path.l1, path.l2, path.r - 2)
    samples = np.arange(0., inner_path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [inner_path(s)]

    x = [c[0] for c in coord]
    y = [c[1]+2 for c in coord]

    plt.plot(x, y, '-k', linewidth=0.5)

    ext_path = Path(path.l1, path.l2, path.r + 2)
    samples = np.arange(0., ext_path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [ext_path(s)]

    x = [c[0] for c in coord]
    y = [c[1]-2 for c in coord]

    plt.plot(x, y, '-k', linewidth=0.5)

def renderVideo(simX, simU, simX_horizon, t, car_model, fixed_obstacles, simObs_position, path, folder, h_cbf):
    # load track
    s=simX[:,0]
    l=simX[:,1]
    theta_tilde=simX[:,2]

    period = np.mean(np.diff(t))
    fr = int(np.around(1/period, decimals=0))
    
    #print(s, l, theta_tilde)
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

        moving_obstacles = simObs_position[i, 0, :]
        moving_obstacles = simObs_position[i]
        moving_obstacles = moving_obstacles.reshape((2, 4))
        savePlot(x[i], y[i], theta[i], v, w, simX_horizon[i, :, :],folder, i, car_model, fixed_obstacles, moving_obstacles, path, h_cbf)
        #plt.show()
    os.chdir('results/' + folder)
    os.system(f"ffmpeg -framerate {fr}"+" -i %04d.png -r 30 -pix_fmt yuv420p video.mp4")
    for i in tqdm(range(1, len(x)), desc="Removing temp files"):
        os.system('rm %04d.png' %i)
    os.chdir('../..')


def plotRes2(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t, simU[:,0], color='r')
    plt.step(t, simU[:,1], color='g')
    plt.title('closed-loop simulation')
    plt.legend([r'$a$',r'$a_w$'])
    plt.ylabel(r'$u$')
    #plt.xlabel(r'$t$')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:,1:])
    plt.ylabel(r'$x$')
    plt.xlabel(r'$t$')
    plt.legend([r'$l$',r'$\tilde{\theta}$', r'$v$', r'$\omega$'])
    plt.grid(True)

def plotResS(simX,simU,t):
    # plot results
    plt.figure()
    plt.title('s coordinate')
    plt.plot(t, simX[:,0], color='c')
    plt.ylabel(r'$x$')
    #plt.xlabel(r'$t$')
    plt.legend([r'$s$'])
    plt.grid(True)


def plotRes3(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.step(t, simU[:,0], color='r')
    plt.step(t, simU[:,1], color='g')
    plt.title('closed-loop simulation')
    plt.legend([r'$a$',r'$a_w$'])
    plt.ylabel(r'$u$')
    #plt.xlabel(r'$t$')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(t, simX[:,0], color='c')
    plt.ylabel(r'$x$')
    #plt.xlabel(r'$t$')
    plt.legend([r'$s$'])
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(t, simX[:,1:])
    plt.ylabel(r'$x$')
    plt.xlabel(r'$t$')
    plt.legend([r'$l$',r'$\tilde{\theta}$', r'$v$', r'$\omega$'])
    plt.grid(True)
