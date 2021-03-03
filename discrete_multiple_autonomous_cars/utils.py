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

def savePlot(x1, y1, theta1,x2, y2, theta2, x3, y3, theta3, v, w, X_horizon, folder, i, car_model, fixed_obstacles, moving_obstacles, path, h_cbf, frenet_coords):
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
    x = x1
    y = y1
    theta = theta1
    t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='blue')
    plt.gca().add_patch(t1)
    t1 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='blue')
    plt.gca().add_patch(t1)
    t1 = t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta), y+ (1/2)*h*np.sin(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='red')
    plt.gca().add_patch(t1)

    s_obs = frenet_coords[0, 0]
    l_obs = frenet_coords[0, 1]
    theta_obs = frenet_coords[0, 2]
    delta = 0.1
    srange = np.arange(-2, path.get_len()*3, delta)
    lrange = np.arange(-2, 2, delta)
    S, L = np.meshgrid(srange,lrange)

    F = (S-s_obs)**4/h**4 + (L-l_obs)**4/h2**4
    X1, Y1 = np.vectorize(path)(S)
    THETA_R = np.vectorize(path.get_theta_r)(S)
    X = X1 - np.sin(THETA_R)*L
    Y = Y1 + np.cos(THETA_R)*L
    #F = ((np.cos(-theta)*(X-x)-np.sin(-theta)*(Y-y))**4 / h**4 ) + ((np.sin(-theta)*(X-x)+np.cos(-theta)*(Y-y))**4 / h2**4)
    plt.contour(X, Y, (F), [h_cbf], linestyles='dashed', linewidths=0.5, colors='blue')

    x = x2
    y = y2
    theta = theta2
    t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='yellow')
    plt.gca().add_patch(t1)
    t1 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='yellow')
    plt.gca().add_patch(t1)
    t1 = t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta), y+ (1/2)*h*np.sin(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='red')
    plt.gca().add_patch(t1)

    s_obs = frenet_coords[1, 0]
    l_obs = frenet_coords[1, 1]
    theta_obs = frenet_coords[1, 2]
    delta = 0.1
    srange = np.arange(-2, path.get_len()*3, delta)
    lrange = np.arange(-2, 2, delta)
    S, L = np.meshgrid(srange,lrange)

    F = (S-s_obs)**4/h**4 + (L-l_obs)**4/h2**4
    X1, Y1 = np.vectorize(path)(S)
    THETA_R = np.vectorize(path.get_theta_r)(S)
    X = X1 - np.sin(THETA_R)*L
    Y = Y1 + np.cos(THETA_R)*L
    #F = ((np.cos(-theta)*(X-x)-np.sin(-theta)*(Y-y))**4 / h**4 ) + ((np.sin(-theta)*(X-x)+np.cos(-theta)*(Y-y))**4 / h2**4)
    plt.contour(X, Y, (F), [h_cbf], linestyles='dashed', linewidths=0.5, colors='blue')

    x = x3
    y = y3
    theta = theta3
    t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)],  [x + half_edge*np.sin(theta)+(1/2)*h*np.cos(theta), y- half_edge*np.cos(theta)+(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t1)
    t1 = plt.Polygon([[x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x+ (1/2)*h*np.cos(theta)- half_edge*np.sin(theta), y+(1/2)*h*np.sin(theta)+ half_edge*np.cos(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='green')
    plt.gca().add_patch(t1)
    t1 = t1 = plt.Polygon([[x+ (1/2)*h*np.cos(theta), y+ (1/2)*h*np.sin(theta)], [x - half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y+ half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)], [x + half_edge*np.sin(theta)-(1/2)*h*np.cos(theta), y - half_edge*np.cos(theta)-(1/2)*h*np.sin(theta)]], color='red')
    plt.gca().add_patch(t1)

    s_obs = frenet_coords[2, 0]
    l_obs = frenet_coords[2, 1]
    theta_obs = frenet_coords[2, 2]
    delta = 0.1
    srange = np.arange(-2, path.get_len()*3, delta)
    lrange = np.arange(-2, 2, delta)
    S, L = np.meshgrid(srange,lrange)

    F = (S-s_obs)**4/h**4 + (L-l_obs)**4/h2**4
    X1, Y1 = np.vectorize(path)(S)
    THETA_R = np.vectorize(path.get_theta_r)(S)
    X = X1 - np.sin(THETA_R)*L
    Y = Y1 + np.cos(THETA_R)*L
    #F = ((np.cos(-theta)*(X-x)-np.sin(-theta)*(Y-y))**4 / h**4 ) + ((np.sin(-theta)*(X-x)+np.cos(-theta)*(Y-y))**4 / h2**4)
    plt.contour(X, Y, (F), [h_cbf], linestyles='dashed', linewidths=0.5, colors='blue')
    
    if fixed_obstacles is not None:
        for o in range(fixed_obstacles.shape[0]):
            obs = fixed_obstacles[o, :]
            drawObstacles(obs, path, car_model, h_cbf)

    #for o in range(moving_obstacles.shape[0]):
    #    obs = moving_obstacles[o, :3]
    #    drawObstacles(obs, path, car_model)

    #for i in range(X_horizon.shape[0]):
    (_x, _y, _theta) = transformProj2Orig(X_horizon[:,0], X_horizon[:,1], X_horizon[:,2], path)
    plt.plot(_x, _y, '-r', linewidth=0.5)
    (_x, _y, _theta) = transformProj2Orig(X_horizon[:,3], X_horizon[:,4], X_horizon[:,5], path)
    plt.plot(_x, _y, '-r', linewidth=0.5)
    (_x, _y, _theta) = transformProj2Orig(X_horizon[:,6], X_horizon[:,7], X_horizon[:,8], path)
    plt.plot(_x, _y, '-r', linewidth=0.5)
    
    
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
    srange = np.arange(-2, path.get_len()*3, delta)
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
    s1=simX[:,0]
    l1=simX[:,1]
    theta_tilde1=simX[:,2]
    s2=simX[:,3]
    l2=simX[:,4]
    theta_tilde2=simX[:,5]
    s3=simX[:,6]
    l3=simX[:,7]
    theta_tilde3=simX[:,8]

    period = np.mean(np.diff(t))
    fr = int(np.around(1/period, decimals=0))
    
    #print(s, l, theta_tilde)
    # transform data
    (x1, y1, theta1) = transformProj2Orig(s1, l1, theta_tilde1, path)
    (x2, y2, theta2) = transformProj2Orig(s2, l2, theta_tilde2, path)
    (x3, y3, theta3) = transformProj2Orig(s3, l3, theta_tilde3, path)
    
    # plot racetrack map
    for i in tqdm(range(len(x1)), desc="Creating frames of video"):
        v = simU[:i, 0]
        w = simU[:i, 1]
        input_len = 40
        if len(v)>input_len:
            v = v[-input_len:]
            w = w[-input_len:]

        frenet_coords = np.array([[s1[i], l1[i], theta_tilde1[i]],
                            [s2[i], l2[i], theta_tilde2[i]],
                            [s3[i], l3[i], theta_tilde3[i]]])
        savePlot(x1[i], y1[i], theta1[i],x2[i], y2[i], theta2[i],x3[i], y3[i], theta3[i], v, w, simX_horizon[i, :, :],folder, i, car_model, fixed_obstacles, None, path, h_cbf, frenet_coords)
        #plt.show()
    os.chdir('results/' + folder)
    os.system(f"ffmpeg -framerate {fr}"+" -i %04d.png -r 30 -pix_fmt yuv420p video.mp4")
    for i in tqdm(range(1, len(x1)), desc="Removing temp files"):
        os.system('rm %04d.png' %i)
    os.chdir('../..')


def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.title('closed-loop simulation')
    
    plt.subplot(2, 1, 1)
    plt.plot(t, simU[:,:])
    plt.ylabel('u')

    plt.legend([r'$v_1$',r'$w_1$', r'$v_2$',r'$w_2$', r'$v_3$',r'$w_3$'])

    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:,:])
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend([r'$s_1$',r'$l_1$',r'$\tilde{\theta}_1$', r'$s_2$',r'$l_2$',r'$\tilde{\theta}_2$', r'$s_3$',r'$l_3$',r'$\tilde{\theta}_3$'])
    plt.grid(True)