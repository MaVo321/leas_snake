"""
Right now: Only for 4 links!!!!
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# x = [theta, p, theta_dot, p_dot] \in \R^(2 * N + 4)

import numpy as np
import snake_robot as sr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from scipy.linalg import pinv
from scipy.integrate import solve_ivp
from random import uniform, seed
from matplotlib.animation import FuncAnimation
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()  # for colorful prints


#------------------------------------------------------------------------------
# global variables
#------------------------------------------------------------------------------

# time step
delt = 0.01                     
# final time
T = 0.5

# Dimensions of x and u
n_x = 2 * sr.N + 4 # dimension of state vector
n_u = sr.N - 1 # dimension of control input

print('Number of snake links = ', sr.N)
                  
#--------------------------bounds for control and state------------------------         

p_min = [0.5, 0.5]
p_max = [1.5, 1.5]

p_dot_min = [-0.05, -0.05] # in m/s
p_dot_max = [0.05, 0.05]

theta_min = [-np.pi for _ in range(sr.N)]
theta_max = [np.pi for _ in range(sr.N)]

# theta_min = [-0 for _ in range(sr.N)]
# theta_max = [2 * np.pi for _ in range(sr.N)]

theta_dot_min = [-2. for i in range(sr.N)] # in rad/s
theta_dot_max = [2. for i in range(sr.N)]

# theta_dot_min = [-np.pi for _ in range(sr.N)]
# theta_dot_max = [np.pi for _ in range(sr.N)]

x_min = np.hstack((np.hstack((theta_min, p_min)), np.hstack((theta_dot_min, p_dot_min))))
x_max = np.hstack((np.hstack((theta_max, p_max)), np.hstack((theta_dot_max, p_dot_max))))

seed(25)
# initial condition
x0 = np.array([uniform(x_min[i], x_max[i]) for i in range(n_x)])
x0[sr.N] = 1
x0[sr.N + 1] = 1
seed(25)


#------------------------------------------------------------------------------
 # number of time steps 
nt = round(T / delt) + 1         
print('Number of time steps = ', nt)

 # array of time steps 
t = np.linspace(0.0, T, nt)   

# number of random initial conditions
nData = 100
#------------------------------------------------------------------------------

def main():
    #--------------------------------------------------------------------------
    # Sampling
    #--------------------------------------------------------------------------
    # ODE is a function: RKV 
    ODE = packageRKV  # RKV
    
    # Create nData random initial conditions
    # XData = np.ones((n_x, nData), dtype=float)
    # # UData = np.ones((n_u, nuData), dtype=float)
    # # seed(1)
    # for i in range(n_x):
    #     XData[i, :] *= [uniform(x_min[i], x_max[i]) for _ in range(nData)] # XData = [x_1, x_2, ..., x_d]
    #                                                                        #         [u_1, u_2, ..., u_d]
    #                                                                        # Y = [x_{u_1}(delta t; x_1), ..., x_{u_d}(delta t; x_d)]
    #                                                                        # PsiOmega = [psi(x_1), psi(x_2), ..., psi(x_d)]
    #                                                                        #            [u_1     , u_2     , ..., u_d     ]


    # # perform one Runge Kutta step for all initial conditions in X and all 
    # # controls in U
    # Y = np.zeros((n_x, nData), dtype=float)
    # for i in range(nData):
    #     # Y[:, i] = ODE(np.zeros(sr.N-1), XData[: n_x, i], sr.snake_robot_rhs, delt) 
    #     Y[:, i] = ODE(np.ones(sr.N-1), XData[: n_x, i], sr.snake_robot_rhs, delt) 

    #Create nData random initial conditions
    XData = np.ones((n_x, nData), dtype=float)
    # UData = np.ones((n_u, nuData), dtype=float)
    # seed(1)
    for i in range(n_x):
        XData[i, :] *= [uniform(x_min[i], x_max[i]) for _ in range(nData)] # XData = [x_1, x_2, ..., x_d]
                                                                           #         [u_1, u_2, ..., u_d]
                                                                           # Y = [x_{u_1}(delta t; x_1), ..., x_{u_d}(delta t; x_d)]
                                                                           # PsiOmega = [psi(x_1), psi(x_2), ..., psi(x_d)]
                                                                           #            [u_1     , u_2     , ..., u_d     ]


    # perform one Runge Kutta step for all initial conditions in X and all 
    # controls in U
    Y = np.zeros((n_x, nData), dtype=float)
    for i in range(nData):
        # Y[:, i] = ODE(np.zeros(sr.N-1), XData[: n_x, i], sr.snake_robot_rhs, delt) 
        Y[:, i] = ODE(np.zeros(sr.N-1), XData[: n_x, i], sr.snake_robot_rhs, delt)

    
    K, iy = EDMD(XData, Y) 
    print(K.shape, x0, psi(x0))
    # input()

    #------------------------choose control sequence---------------------------
    # in this case, u is constant
    u_test =  1. * np.array([[1. for _ in range(n_u)] for _ in range(nt)], dtype=float)

    # random control 
    # u_test = np.array([[uniform(u_min, u_max) for _ in range(n_u)] for i in range(nt)])# np.array([[np.sin(i) for _ in range(n_u)] for i in range(nt)]) # 1 * np.array([[1. for _ in range(n_u)] for _ in range(nt)], dtype=float)

    # RKV
    x_test = ODE(u_test, x0, sr.snake_robot_rhs, delt).T
    print('x_test', x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3], x_test[:, 10]) # , x_test.T)

    # SUR1
    z_test = SUR(x0, u_test, K, iy).T
    print('z_test', z_test[:, 0], z_test[:, 1], z_test[:, 2], z_test[:, 3], z_test[:, 10])

    # Values for RKV
    theta_RKV = x_test[:sr.N]
    p_movement_RKV = x_test[sr.N : sr.N + 2] 
    centres_of_masses_x_RKV = - sr.l * sr.K(sr.N).T @ sr.cos(theta_RKV) + p_movement_RKV[0] * sr.e(sr.N).reshape([sr.N, 1])
    centres_of_masses_y_RKV = - sr.l * sr.K(sr.N).T @ sr.sin(theta_RKV) + p_movement_RKV[1] * sr.e(sr.N).reshape([sr.N, 1])
    joints_x_RKV = np.concatenate((centres_of_masses_x_RKV - sr.l * np.cos(theta_RKV),
                           np.array([centres_of_masses_x_RKV[sr.N-1] + sr.l * np.cos(theta_RKV[sr.N-1])])))
    joints_y_RKV = np.concatenate((centres_of_masses_y_RKV - sr.l * np.sin(theta_RKV),
                           np.array([centres_of_masses_y_RKV[sr.N-1] + sr.l * np.sin(theta_RKV[sr.N-1])])))
    
    #---------------------------------------------------------------------------
    # plot
    #---------------------------------------------------------------------------

    fig, ax = plt.subplots()
    centre_of_mass_RKV = ax.scatter(p_movement_RKV[0][0], p_movement_RKV[1][0], c="black", s=5)
    cm_before_RKV = ax.scatter(p_movement_RKV[0][0], p_movement_RKV[1][0], c='gray', s=5)
    cm_of_all_links_RKV = ax.scatter(centres_of_masses_x_RKV[0], centres_of_masses_y_RKV[0], c='blue', s=5)
    snake_RKV = ax.plot(joints_x_RKV[:, 0], joints_y_RKV[:, 0], c="gray", label='nominal')[0]

    min_x = min([min(joints_x_RKV[ii]) for ii in range(sr.N)])
    max_x = max([max(joints_x_RKV[ii]) for ii in range(sr.N)])
    # delta_x = 1/2 * abs(max_x-min_x)
    min_y = min([min(joints_y_RKV[ii]) for ii in range(sr.N)])
    max_y = max([max(joints_y_RKV[ii]) for ii in range(sr.N)])
    # delta_y = 1/2 * abs(max_y-min_y)


    # Values for SUR

    theta = z_test[:sr.N]
    p_movement = z_test[sr.N : sr.N + 2] 
    centres_of_masses_x = - sr.l * sr.K(sr.N).T @ sr.cos(theta) + p_movement[0] * sr.e(sr.N).reshape([sr.N, 1])
    centres_of_masses_y = - sr.l * sr.K(sr.N).T @ sr.sin(theta) + p_movement[1] * sr.e(sr.N).reshape([sr.N, 1])

    joints_x = np.concatenate((centres_of_masses_x - sr.l * np.cos(theta),
                           np.array([centres_of_masses_x[sr.N-1] + sr.l * np.cos(theta[sr.N-1])])))
    joints_y = np.concatenate((centres_of_masses_y - sr.l * np.sin(theta),
                           np.array([centres_of_masses_y[sr.N-1] + sr.l * np.sin(theta[sr.N-1])])))
    
    #---------------------------------------------------------------------------
    # plot
    #---------------------------------------------------------------------------

    # fig, ax = plt.subplots()
    centre_of_mass = ax.scatter(p_movement[0][0], p_movement[1][0], c="red", s=5)
    cm_before = ax.scatter(p_movement[0][0], p_movement[1][0], c='pink', s=5)
    cm_of_all_links = ax.scatter(centres_of_masses_x[0], centres_of_masses_y[0], c='blue', s=5)
    snake = ax.plot(joints_x[:, 0], joints_y[:, 0], c="green", label='EDMD')[0]


    def init():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        return centre_of_mass_RKV, cm_before_RKV, snake_RKV, cm_of_all_links_RKV, centre_of_mass, cm_before, snake, cm_of_all_links


    def update(frame):
        # RKV
        #------------------------------------------------------------------------
        # plot centre of mass
        # for each frame, update the data stored on each artist.
        xdata_cm_RKV = p_movement_RKV[0][frame]
        ydata_cm_RKV = p_movement_RKV[1][frame]
        # update the scatter plot:
        data_cm_RKV = np.stack([xdata_cm_RKV, ydata_cm_RKV]).T
        centre_of_mass_RKV.set_offsets(data_cm_RKV)
        # for each frame, update the data stored on each artist.
        xdata_before_RKV = p_movement_RKV[0][:frame]
        ydata_before_RKV = p_movement_RKV[1][:frame]
        # update the scatter plot:
        data_b_RKV = np.stack([xdata_before_RKV, ydata_before_RKV]).T
        cm_before_RKV.set_offsets(data_b_RKV)
        # ------------------------
        # plot snake
        snake_RKV.set_xdata(joints_x_RKV[:, frame])
        snake_RKV.set_ydata(joints_y_RKV[:, frame])
        # for each frame, update the data stored on each artist.
        xdata_cms_RKV = centres_of_masses_x_RKV[:, frame]
        ydata_cms_RKV = centres_of_masses_y_RKV[:, frame]
        # update the scatter plot:
        data_cms_RKV = np.stack([xdata_cms_RKV, ydata_cms_RKV]).T
        cm_of_all_links_RKV.set_offsets(data_cms_RKV)

        # SUR
        xdata_cm = p_movement[0][frame]
        ydata_cm = p_movement[1][frame]
        # update the scatter plot:
        data_cm = np.stack([xdata_cm, ydata_cm]).T
        centre_of_mass.set_offsets(data_cm)
        # for each frame, update the data stored on each artist.
        xdata_before = p_movement[0][:frame]
        ydata_before = p_movement[1][:frame]
        # update the scatter plot:
        data_b = np.stack([xdata_before, ydata_before]).T
        cm_before.set_offsets(data_b)
        # ------------------------
        # plot snake
        snake.set_xdata(joints_x[:, frame])
        snake.set_ydata(joints_y[:, frame])
        # for each frame, update the data stored on each artist.
        xdata_cms = centres_of_masses_x[:, frame]
        ydata_cms = centres_of_masses_y[:, frame]
        # update the scatter plot:
        data_cms = np.stack([xdata_cms, ydata_cms]).T
        cm_of_all_links.set_offsets(data_cms)
         
        return cm_before_RKV, centre_of_mass_RKV, snake_RKV, cm_of_all_links_RKV, cm_before, centre_of_mass, snake, cm_of_all_links


    ani = FuncAnimation(fig, update, frames=np.arange(np.shape(p_movement_RKV)[1]),
                        init_func=init, blit=True, interval=50)
    plt.legend()
    # writervideo = FFMpegWriter(fps=60)
    # ani.save('robot_snakemovement_no_control_double_mass_deg3.mp4', writer=writervideo)
    # plt.close()

    plt.show()
    
   
    
    #------------------------------plot trajectories---------------------------
    
    # plt.plot(x_test[: , sr.N], x_test[: , sr.N + 1], color='blue', linestyle='solid', label='ODE')
    plt.figure()
    plt.plot(z_test.T[: , sr.N], z_test.T[: , sr.N + 1], color='orange', linestyle='dashed', label='SUR')
    plt.legend()
    # plt.xlim([0, 2])
    # plt.ylim([0, 2])
    plt.xlabel('$p_x$')
    plt.ylabel('$p_y$')

    print(z_test.shape, x_test.shape)
    plt.figure()
    plt.title('Deviation angles')
    plt.plot([np.linalg.norm(z_test.T[i, :sr.N] - x_test.T[i, :sr.N]) for i in range(nt)])
    plt.semilogy()

    plt.show()    
    
    return

###############################################################################
# Functions
###############################################################################

def RKV(u_: np.ndarray, 
        x0_: np.ndarray, 
        rhs: callable, 
        h: float):
    """
    Runge-Kutta method of forth order for numerically solving differential 
    equations

    Parameters
    ----------
    u_ : numpy.ndarray
          control
    x0_ : numpy.ndarray
        initial condition
    rhs : function
        function of righthand side
    h : float
        step size

    Returns
    -------
    y_ : numpy.ndarray
        next iteration value(s).

    """
    # try:
    #     if not u_.shape[0] == 1:
    # # if not u_.shape[0] == n_u:
    #         x_ = np.zeros((u_.shape[0], n_x), dtype=float)
    #         x_[0, :] = x0_
    #         for ii in range(u_.shape[0] - 1):
    #             k1 = rhs(x_[ii, :], u_[ii])
    #             k2 = rhs(x_[ii, :] + h / 2 * k1, u_[ii])
    #             k3 = rhs(x_[ii, :] + h / 2 * k2, u_[ii])
    #             k4 = rhs(x_[ii, :] + h * k3, u_[ii])
    #             x_[ii + 1, :] = x_[ii, :] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #     else:
    #         k1 = rhs(x0_, u_)
    #         k2 = rhs(x0_ + h / 2 * k1, u_)
    #         k3 = rhs(x0_ + h / 2 * k2, u_)
    #         k4 = rhs(x0_ + h * k3, u_)
    #         x_ = x0_ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
      
    # except:
    #     # print('except')
    #     k1 = rhs(x0_, u_)
    #     k2 = rhs(x0_ + h / 2 * k1, u_)
    #     k3 = rhs(x0_ + h / 2 * k2, u_)
    #     k4 = rhs(x0_ + h * k3, u_)
    #     x_ = x0_ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  
    # return x_

    if u_.shape[0] <= n_u: # and u_.shape[1] == 1:
        rhs_new = lambda t, x: rhs(x, u_) 
        sol = solve_ivp(
                lambda t, x: rhs_new(t, x),
                [0, delt],
                x0_,
                t_eval=[delt]  
            )
        x_ = sol.y.T
        x_ = x_[-1]
    else: 
        # print('2', u_.shape)  
        x_ = np.zeros((u_.shape[0], n_x), dtype=float)
        # print(x_.shape)
        x_[0, :] = x0_
        t_ = 0
        for ii in range(u_.shape[0] - 1):
            rhs_new = lambda t, x: rhs(x, u_[ii]) 
            sol = solve_ivp(
                lambda t, x: rhs_new(t, x),
                [t_, t_ + delt],
                x_[ii, :],
                t_eval=[delt]  
            )
            # print(x_.shape)
            x = sol.y.T
            x_[ii + 1, :] = x[-1]

    return x_


def packageRKV(u_: np.ndarray,  # todo:delete and change back to RKV()
               x0_: np.ndarray,
               rhs: callable,
               h: float):
    """
    Runge-Kutta method of forth order for numerically solving differential
    equations

    Parameters
    ----------
    u_ : numpy.ndarray
          control
    x0_ : numpy.ndarray
        initial condition
    rhs : function
        function of righthand side
    h : float
        step size

    Returns
    -------
    y_ : numpy.ndarray
        next iteration value(s).

    """
    # try:
    #     if not u_.shape[0] == 1:
    # # if not u_.shape[0] == n_u:
    #         x_ = np.zeros((u_.shape[0], n_x), dtype=float)
    #         x_[0, :] = x0_
    #         for ii in range(u_.shape[0] - 1):
    #             k1 = rhs(x_[ii, :], u_[ii])
    #             k2 = rhs(x_[ii, :] + h / 2 * k1, u_[ii])
    #             k3 = rhs(x_[ii, :] + h / 2 * k2, u_[ii])
    #             k4 = rhs(x_[ii, :] + h * k3, u_[ii])
    #             x_[ii + 1, :] = x_[ii, :] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #     else:
    #         k1 = rhs(x0_, u_)
    #         k2 = rhs(x0_ + h / 2 * k1, u_)
    #         k3 = rhs(x0_ + h / 2 * k2, u_)
    #         k4 = rhs(x0_ + h * k3, u_)
    #         x_ = x0_ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # except:
    #     # print('except')
    #     k1 = rhs(x0_, u_)
    #     k2 = rhs(x0_ + h / 2 * k1, u_)
    #     k3 = rhs(x0_ + h / 2 * k2, u_)
    #     k4 = rhs(x0_ + h * k3, u_)
    #     x_ = x0_ + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # return x_

    print(f"{Fore.GREEN}RKV without the else{Style.RESET_ALL}")
    rhs_new = lambda t, x: rhs(x, u_)
    sol = solve_ivp(
            rhs_new,  # lambda t, x: rhs_new(t, x),  #
            [0, delt],
            x0_,
            t_eval=[delt]
        )
    x_ = sol.y.T
    x_ = x_[-1]

    return x_


def monomials(n: int, 
              m: int):
    """
    Creates exponents of n-variate monomials with degree <= m

    Parameters
    ----------
    n : int
        number of variables 
    m : int
        max degree

    Returns
    -------
    exp_new : list
        list of lists of exponents

    """
    from itertools import combinations_with_replacement, permutations 
    
    list_to_iter = range(m + 1)
    exp = list(combinations_with_replacement(list_to_iter, n))
    exp_new = []
    for e in exp:
        if sum(e) <= m:
            perms = list(set(permutations(e)))
            # prevent overfitting
            for perm in perms: 
                if perm[2 * sr.N] <= 1 and perm[2 * sr.N + 1] <= 1: # and perm[3 * sr.N + 2] <= 1 and perm[3 * sr.N + 3] <= 1: # and not (sum(perm) > 1 and not perm[sr.N] == 0 and not perm[sr.N + 1] == 0): 
                    if perm[sr.N] == 0 or sum(perm) <= 1:
                        if perm[sr.N + 1] == 0 or sum(perm) <= 1:
                            exp_new.append(perm)
    # exp_new.append(exp_new[-1])
    # exp_new.append((0, 0, 0))
    return exp_new


def observables(n: int):
    """
    Function of observables

    Parameters
    ----------
    n : int
        number of variables 

    Returns
    -------
    exp_new : list
        list of callables

    """
    # exp_new = [lambda x: 1, lambda x: x[0], lambda x: x[1], lambda x: x[2], lambda x: x[3], lambda x: x[4], lambda x: x[5], lambda x: x[6], 
    #            lambda x: x[7], lambda x: x[8], lambda x: x[9], lambda x: x[10], lambda x: x[11], lambda x: np.sin(x[0]), lambda x: np.sin(x[1]), lambda x: np.sin(x[2]), lambda x: np.sin(x[3])]
    #         0       1        2         3      4    5       6             7            8            9          10       11
    # x = [theta_1, theta_2, theta_3, theta_4, p_x, p_y, theta_1_dot, theta_2_dot, theta_3_dot, theta_4_dot, p_x_dot, p_y_dot]
    exp_new = [lambda x: 1, lambda x: x[0], lambda x: x[1], lambda x: x[2], lambda x: x[3], lambda x: x[4], lambda x: x[5], lambda x: x[6], 
               lambda x: x[7], lambda x: x[8], lambda x: x[9], lambda x: x[10], lambda x: x[11], lambda x: np.sin(x[0]), lambda x: np.sin(x[1]), lambda x: np.sin(x[2]), lambda x: np.sin(x[3]),
               # lambda x: x[0]**2, lambda x: x[1]**2, lambda x: x[2]**2, lambda x: x[3]**2,
               lambda x: np.sin(x[0])**2, lambda x: np.sin(x[1])**2, lambda x: np.sin(x[2])**2, lambda x: np.sin(x[3])**2,
               lambda x: x[4]**2, lambda x: x[5]**2, lambda x: x[6]**2, 
               lambda x: x[7]**2, lambda x: x[8]**2, lambda x: x[9]**2, lambda x: x[10]**2, lambda x: x[11]**2,
            #    lambda x: x[0]**3, lambda x: x[1]**3, lambda x: x[2]**3, lambda x: x[3]**3,lambda x: x[4]**3, lambda x: x[5]**3, lambda x: x[6]**3, 
            #    lambda x: x[7]**3, lambda x: x[8]**3, lambda x: x[9]**3, lambda x: x[10]**3, lambda x: x[11]**3,
            #    lambda x: np.sin(x[0])**2, lambda x: np.sin(x[1])**2, lambda x: np.sin(x[2])**2, lambda x: np.sin(x[3])**2, 
            #    lambda x: x[0]**2, lambda x: x[1]**2, lambda x: x[2]**2, lambda x: x[3]**2, 
            #    lambda x: np.sin(x[0]) * x[6], lambda x: np.sin(x[1]) * x[7], lambda x: np.sin(x[2]) * x[8], lambda x: np.sin(x[3]) * x[9], 
            #    lambda x: np.sin(x[0])**2 * x[6], lambda x: np.sin(x[1])**2 * x[7], lambda x: np.sin(x[2])**2 * x[8], lambda x: np.sin(x[3])**2 * x[9], 
            #    lambda x: x[0]**3, lambda x: x[1]**3, lambda x: x[2]**3, lambda x: x[3]**3,
            #    lambda x: np.sin(x[0]) * x[10], lambda x: np.sin(x[0]) * x[11], lambda x: np.sin(x[1]) * x[10], lambda x: np.sin(x[1]) * x[11], lambda x: np.sin(x[2]) * x[10], lambda x: np.sin(x[2]) * x[11], 
            #    lambda x: np.sin(x[3]) * x[10], lambda x: np.sin(x[3]) * x[11]]
                ]
               
    # exp_new = [lambda x: 1, lambda x: x[0], lambda x: x[1], lambda x: x[2], lambda x: x[3], lambda x: x[4], lambda x: x[5], lambda x: x[6], 
    #            lambda x: x[7], lambda x: x[8], lambda x: x[9], lambda x: x[10], lambda x: x[11]]
    
    return exp_new


def obs_2():
    # list_of_exp = [tuple(np.zeros(n_x))]
    # for i in range(1, n_x + 1):
    #     h = np.zeros(n_x)
    #     h[i - 1] = 1
    #     list_of_exp.append(tuple(h))
    # for i in range(n_x):
    #     if i not in [4, 5]:
    #         for j in range(n_x):
    #             if j not in [4, 5]:
    #                 h = np.zeros(n_x)
    #                 if i == j:
    #                     h[i] = 2
    #                 else:
    #                     h[i] = 1
    #                     h[j] = 1
    #                 list_of_exp.append(tuple(h))

    # for i in range(n_x):
    #     if i not in [4, 5]:
    #         for j in range(n_x):
    #             if j not in [4, 5]:
    #                 for k in range(n_x):
    #                     if k not in [4, 5]:
    #                         h = np.zeros(n_x)
    #                         if i == j and not i == k:
    #                             h[i] = 2
    #                         elif i == k and not i == j:
    #                             h[i] == 2
    #                         elif j == k and not i == j:
    #                             h[j] == 2
    #                         elif i == j and i == k:
    #                             h[i] = 3
    #                         else:
    #                             h[i] = 1
    #                             h[j] = 1
    #                             h[k] = 1
    #                         list_of_exp.append(tuple(h))

    # for i in range(1, sr.N):
    #     h = np.zeros(n_x)
    #     h[i - 1] = 1
    #     list_of_exp.append(tuple(h))

    list_of_exp = [tuple(np.zeros(n_x))]
    for i in range(1, n_x + 1):
        h = np.zeros(n_x)
        h[i - 1] = 1
        list_of_exp.append(tuple(h))
    for i in range(n_x):
        if i not in [4, 5]:
            for j in range(n_x):
                if j not in [4, 5]:
                    h = np.zeros(n_x)
                    if i == j:
                        h[i] = 2
                    else:
                        h[i] = 1
                        h[j] = 1
                    list_of_exp.append(tuple(h))

    # for i in range(n_x):
    #     if i not in [4, 5]:
    #         for j in range(n_x):
    #             if j not in [4, 5]:
    #                 for k in range(n_x):
    #                     if k not in [4, 5]:
    #                         h = np.zeros(n_x)
    #                         if i == j and not i == k:
    #                             h[i] = 2
    #                         elif i == k and not i == j:
    #                             h[i] == 2
    #                         elif j == k and not i == j:
    #                             h[j] == 2
    #                         elif i == j and i == k:
    #                             h[i] = 3
    #                         else:
    #                             h[i] = 1
    #                             h[j] = 1
    #                             h[k] = 1
    #                         list_of_exp.append(tuple(h))
    #
    # for i in range(1, sr.N):
    #     h = np.zeros(n_x)
    #     h[i - 1] = 1
    #     list_of_exp.append(tuple(h))

    return list_of_exp


def psi(X_: np.ndarray, 
        returnProjection: bool=False):
    """
    Evaluates observables at every data point and puts dem in matrix PsiX_.

    Parameters
    ----------
    X_ : numpy.ndarray
        Data matrix
    returnProjection : bool, optional
        The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # list_of_obs = observables(n_x)

    # if len(np.array(X_).shape) == 1:
    #     X_ = np.array([X_]).T
    # nPoly = len(list_of_obs) 
    # # print(nPoly)

    # PsiX_ = np.zeros((nPoly, X_.shape[1]), dtype=float)
    # for j in range(X_.shape[1]):
    #     s = 0
    #     for obs in list_of_obs:
    #         PsiX_[s, j] = obs(X_[:, j])
    #         s += 1

    # # iy_ = [i for i in range(1, 3 * sr.N + 4 + 1)] # Position of sin(theta), cos(theta), p, sin(theta_dot), cos(theta_dot), p_dot in vector of observables
    # # iy_ = [i for i in range(1, n_x + sr.N + 1)] 
    # iy_ = [i for i in range(1, n_x + 1)]
    # if returnProjection:
    #     return PsiX_, iy_
    # else:
    #     return PsiX_

    list_of_exp = obs_2()
    print('len list_of_exp= ', len(list_of_exp))
 
    if len(np.array(X_).shape) == 1:
        X_ = np.array([X_]).T

    nPoly = len(list_of_exp)
    # print(nPoly)
    PsiX_ = np.zeros((nPoly, X_.shape[1]), dtype=float)

    s = 0 
    for exp in list_of_exp:
        l = 1
        # for i in range(0, 2 * sr.N, 2):
        #     l *= np.power(X_[i, :], exp[i]) # np.power(np.sin(X_[i, :]), exp[i]) * np.power(np.cos(X_[i, :]), exp[i + 1])
        if s <= 1112:
            for i in range(n_x):
                l *= np.power(X_[i, :], exp[i])
        else:
            for i in range(n_x):
                l *= np.power(np.sin(X_[i, :]), exp[i])
        PsiX_[s, :] = l
        s += 1

    # iy_ = [i for i in range(1, n_x + sr.N + 1)] # Position of sin/cos theta, p, theta_dot, p_dot in vector of monomials
    iy_ = [i for i in range(1, n_x + 1)]

    if returnProjection:
        return PsiX_, iy_
    else:
        return PsiX_

    
def EDMD(X_: list or np.ndarray, 
          Y_: list or np.ndarray):

    PsiX, iy = psi(X_, True)
    n_z = PsiX.shape[0]

    G = PsiX @ PsiX.T           
    PsiY = psi(Y_)          # Lift Y to PsiY
    Ai = PsiX @ PsiY.T      # Define A-matrix for i-th system in V for EDMD
    # K = (pinv(G) @ Ai).T
    K = (pinv(PsiX @ PsiX.T) @ PsiX @ PsiY.T).T  # todo
    # K = PsiY * PsiOmega.T * (PsiOmega * PsiOmega.T)^(-1) solution of regression min || K PsiOmega - PsiY|| \in \R^{N x (N + n_u)}

    # Psi(x)^+ = K_A * Psi(x) + K_B * u

    return K, iy


def SUR(x0_: np.ndarray, 
        u_: np.ndarray, 
        K_: np.ndarray, 
        iy: list or np.ndarray):
    
    import copy
    
    try:
        if not u_.shape[0] == 1: 
            x_ = np.zeros((u_.shape[0], len(x0_)), dtype=float)
            # print('x0 in SUR', x0_)
            x_[0, :] = x0_
            # print(x_[0, :])
            s = 0
            # state after next time step is computed 

            for ii in range(u_.shape[0] - 1):
                x_temp = copy.deepcopy(x_[ii])
                x_test = (K_ @ psi(x_temp))[iy, 0]
                x_[ii + 1, :] = x_test
                s += 1
                # print(s)

        else: 
            x_ = np.zeros(n_x)
            x_test = (K_ @ psi(x0_))[iy, 0]
            # get theta
            # for jj in range(sr.N):
            #     x_theta = x_test[jj * 2 : (jj + 1) * 2] / np.linalg.norm(x_test[jj * 2 : (jj + 1) * 2])
            #     if x_theta[-1] >= 0:
            #         theta_ = np.arcsin(x_theta[0])
            #     else:
            #         theta_ = np.pi - np.arcsin(x_theta[0])
            #     x_[jj] = theta_ 

            # x_[sr.N :] = x_test[sr.N * 2 :]
            x_[sr.N :] = x_test
           
    except:
        z0_ = copy.deepcopy(x0_)
        x_test = (K_ @ psi(z0_))[iy, 0]
        # get theta
        # for jj in range(sr.N):
        #     x_theta = x_test[jj * 2 : (jj + 1) * 2] / np.linalg.norm(x_test[jj * 2 : (jj + 1) * 2])
        #     if x_theta[-1] >= 0:
        #         theta_ = np.arcsin(x_theta[0])
        #     else:
        #         theta_ = np.pi - np.arcsin(x_theta[0])
        #     x_[jj] = theta_ 

        # x_[sr.N :] = x_test[sr.N * 2 :]
        x_[sr.N :] = x_test
        
    return x_


main()




