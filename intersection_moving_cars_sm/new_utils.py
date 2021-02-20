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


def drawCar(x, y, theta, X_horizon, car_model, path, color):

    h = car_model.l1
    h2 = car_model.l2
    half_edge = car_model.l2/2
    t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color=color)
    plt.gca().add_patch(t1)
    t1 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color=color)
    plt.gca().add_patch(t1)
    t1 = t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta), y+ (1/2)*h*np.sin(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='red')
    plt.gca().add_patch(t1)

    (_x, _y, _theta) = transformProj2Orig(X_horizon[:,0], X_horizon[:,1], X_horizon[:,2], path)
    plt.plot(_x, _y, '-r', linewidth=0.5)

def drawObstacles(obs, path, car_model, h_cbf):
    obs = transformProj2Orig([obs[0]], [obs[1]], [obs[2]], path)
    h = car_model.l1
    h2 = car_model.l2
    half_edge = car_model.l2/2

    x = obs[0][0]
    y = obs[1][0]
    theta = obs[2][0]
    #print(x, y, theta)
    t2 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t2)
    t2 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t2)
    
    delta = 0.025
    xrange = np.arange(-4, 14, delta)
    yrange = np.arange(-5.5, 12.5, delta)
    X, Y = np.meshgrid(xrange,yrange)
    
    F = ((X-x)**4 / h**4 ) + ((Y-y)**4 / h**4)
    plt.contour(X, Y, (F), [h_cbf], linestyles='dashed', linewidths=0.5, colors='blue')

def savePlot(folder, i):
    plt.savefig('results/' + folder + "/%04d" % i +".png")

def drawPath(path, width):
    samples = np.arange(0., path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [path(s)]

    x = [c[0] for c in coord]
    y = [c[1] for c in coord]

    plt.plot(x, y, '-y', linewidth=0.5)

    inner_path = Path(path.l1, path.l2, path.r - width, traslx=path.traslx, trasly=path.trasly + width)
    samples = np.arange(0., inner_path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [inner_path(s)]

    x = [c[0] for c in coord]
    y = [c[1] for c in coord]

    plt.plot(x, y, '--k', linewidth=0.5)

    ext_path = Path(path.l1, path.l2, path.r + width, traslx=path.traslx, trasly=path.trasly - width)
    samples = np.arange(0., ext_path.get_len(), 0.1)

    coord = []
    for s in samples:
        coord += [ext_path(s)]

    x = [c[0] for c in coord]
    y = [c[1] for c in coord]

    plt.plot(x, y, '--k', linewidth=0.5)

def renderVideo(simX1, simU1, simX_horizon1, fixed_obstacles1, simObs_position1, path1,
                simX2, simU2, simX_horizon2, fixed_obstacles2, simObs_position2, path2,
                car_model, h_cbf, t, folder):

    # frame rate
    period = np.mean(np.diff(t))
    fr = int(np.around(1/period, decimals=0))

    # load track
    if simX1 is not None:
        s1=simX1[:,0]
        l1=simX1[:,1]
        theta_tilde1=simX1[:,2]
        # transform data
        (x1, y1, theta1) = transformProj2Orig(s1, l1, theta_tilde1, path1)

    if simX2 is not None:
        s2=simX2[:,0]
        l2=simX2[:,1]
        theta_tilde2=simX2[:,2]
        # transform data
        (x2, y2, theta2) = transformProj2Orig(s2, l2, theta_tilde2, path2)
    
    # plot racetrack map
    for i in tqdm(range(len(t)), desc="Creating frames of video"):

        plt.figure()
        drawPath(path1, 2)
        drawPath(path2, 2)
        if simX1 is not None:
            drawCar(x1[i], y1[i], theta1[i], simX_horizon1[i], car_model, path1, 'blue')

        if simX2 is not None:
            drawCar(x2[i], y2[i], theta2[i], simX_horizon2[i], car_model, path2, 'yellow')

        if fixed_obstacles1 is not None:
            for j in range(fixed_obstacles1.shape[0]):
                drawObstacles(fixed_obstacles1[j], path1, car_model, h_cbf)
        if fixed_obstacles2 is not None:
            for j in range(fixed_obstacles2.shape[0]):
                drawObstacles(fixed_obstacles2[j], path2, car_model, h_cbf)
        if simObs_position1 is not None:
            moving_obstacles = simObs_position1[i, 0, :]
            num_of_cars = int(moving_obstacles.shape[0]/4)
            moving_obstacles = moving_obstacles.reshape((num_of_cars, 4))
            for j in range(num_of_cars):
                drawObstacles(moving_obstacles[j], path1, car_model, h_cbf)
        if simObs_position2 is not None:
            moving_obstacles = simObs_position2[i, 0, :]
            num_of_cars = int(moving_obstacles.shape[0]/4)
            moving_obstacles = moving_obstacles.reshape((num_of_cars, 4))
            for j in range(num_of_cars):
                drawObstacles(moving_obstacles[j], path2, car_model, h_cbf)
        savePlot(folder, i)
        plt.close()

    os.chdir('results/' + folder)
    os.system(f"ffmpeg -framerate {fr}"+" -i %04d.png -r 30 -pix_fmt yuv420p video.mp4")
    for i in tqdm(range(len(t)), desc="Removing temp files"):
        os.system('rm %04d.png' %i)
    os.chdir('../..')

def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t, simU[:,0], color='r')
    plt.step(t, simU[:,1], color='g')
    plt.step(t, simU[:,2], color='b')
    plt.step(t, simU[:,3], color='c')
    plt.title('closed-loop simulation')
    plt.legend(['v1','w1', 'v2', 'w2'])
    plt.ylabel('u')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:,:])
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend(['s1','l1','theta_tilde1', 's2','l2','theta_tilde2'])
    plt.grid(True)