import numpy as np
from casadi import *
###############################################################################
# Global variables
###############################################################################

N = 4                   # number of links
m = 1.56 # 0.310               # mass of one link in kg
l = 0.4 # 0.0445              # half the length of a link in m

c_t = 4.45 # 0.5                 # tangential ground friction coefficient 
c_n = 17.3 # 3                 # normal ground friction coefficient 

J = 0.0042 # 1/3 * m * l**2      # Moment of inertia of each link in kg * m^2

lambda2:float = 0.0120

###############################################################################
# Functions
###############################################################################

#------------------------------------------------------------------------------
# Basic definitions
#------------------------------------------------------------------------------

def D(n):
    """
    Difference matrix of dimension (n - 1, n)
    """
    d = np.eye(n - 1, n)
    d += -np.eye(n - 1, n, k=1)
    return d

def A(n):
    """
    Addition matrix of dimension (n - 1, n)
    """
    a = np.eye(n - 1, n)
    a += np.eye(n - 1, n, k=1)
    return a

def e(n):
    """
    Summation vector of dimension n
    """
    return np.ones(n)# ([n, 1])

def E(n):
    E_matrix = np.zeros((2 * n, 2))
    E_matrix[:n, 0] = e(n)
    E_matrix[n :, 1] = e(n)
    return E_matrix

def sin(theta):
    s = np.array([np.sin(theta[i]) for i in range(theta.shape[0])])
    return s

def cos(theta):
    s = np.array([np.cos(theta[i]) for i in range(theta.shape[0])])
    return s

def sgn(theta):
    s = np.array([np.sign(theta[i]) for i in range(len(theta))])
    return s

def S_theta(theta):
    S = np.diag(sin(theta))
    return S

def C_theta(theta):
    C = np.diag(cos(theta))
    return C

def V(n):
    return A(n).T @ np.linalg.inv(D(n) @ D(n).T) @ A(n)

def K(n):
    return A(n).T @ np.linalg.inv(D(n) @ D(n).T) @ D(n)

#------------------------------------------------------------------------------
# Right hand side
#------------------------------------------------------------------------------

def M_theta(n, theta):
    # print('J = ', J, 'm = ', m, 'l**2 = ', l**2, 'V = ', V(n))
    M = J * np.eye(n) + m * l**2 * S_theta(theta) @ V(n) @ S_theta(theta) + m * l**2 * C_theta(theta) @ V(n) @ C_theta(theta)
    return M

def W(n, theta):
    w = m * l**2 * S_theta(theta) @ V(n) @ C_theta(theta) - m * l**2 * C_theta(theta) @ V(n) @ S_theta(theta)
    return w

def f_R(n, theta, theta_dot, p_dot):
    """
    Viscous friction forces on the links, F = f_R = [f_Rx; f_Ry] in R^{2*n}
    """
    X_dot = l * K(n).T @ S_theta(theta) @ theta_dot + e(n) * p_dot[0]
    Y_dot = -l * K(n).T @ C_theta(theta) @ theta_dot + e(n) * p_dot[1]
    # print(X_dot.shape, Y_dot.shape, 'C_theta', C_theta(theta).shape, 'K', K(n).shape, 'theta_dot', theta_dot.shape,  (K(n).T @ C_theta(theta) @ theta_dot).reshape((2, 1)).shape, 'e', e(n).shape, 'p_dot', p_dot.shape, (e(n) * p_dot[1]).shape)
    f = np.zeros((2 * N, 2 * N))
    f[: N, : N] = c_t * C_theta(theta)**2 + c_n * S_theta(theta)**2
    f[: N, N :] = (c_t - c_n) * S_theta(theta) @ C_theta(theta)
    f[N :, : N] = (c_t - c_n) * S_theta(theta) @ C_theta(theta)
    f[N :, N :] = c_t * S_theta(theta)**2 + c_n * C_theta(theta)**2
    # print(f.shape, np.hstack((X_dot, Y_dot)).shape, X_dot.shape, Y_dot.shape, (-f @ np.hstack((X_dot, Y_dot))).shape)
    # input()
    F = -f @ np.hstack((X_dot, Y_dot))
    
    return F

def f_R_casadi(n, theta, theta_dot, p_dot):
    """
    Viscous friction forces on the links, F = f_R = [f_Rx; f_Ry] in R^{2*n}
    """
    X_dot = l * K(n).T @ S_theta(theta) @ theta_dot + e(n) * p_dot[0]
    Y_dot = -l * K(n).T @ C_theta(theta) @ theta_dot + e(n) * p_dot[1]
    # print(X_dot.shape, Y_dot.shape, 'C_theta', C_theta(theta).shape, 'K', K(n).shape, 'theta_dot', theta_dot.shape,  (K(n).T @ C_theta(theta) @ theta_dot).reshape((2, 1)).shape, 'e', e(n).shape, 'p_dot', p_dot.shape, (e(n) * p_dot[1]).shape)
    f = np.zeros((2 * N, 2 * N))
    f[: N, : N] = c_t * C_theta(theta)**2 + c_n * S_theta(theta)**2
    f[: N, N :] = (c_t - c_n) * S_theta(theta) @ C_theta(theta)
    f[N :, : N] = (c_t - c_n) * S_theta(theta) @ C_theta(theta)
    f[N :, N :] = c_t * S_theta(theta)**2 + c_n * C_theta(theta)**2
    # print(f.shape)# , np.hstack((X_dot, Y_dot)).shape, X_dot.shape, Y_dot.shape, (-f @ np.hstack((X_dot, Y_dot))).shape)
    # input()
    XY = SX.zeros(2 * N)
    XY[:N] = X_dot
    XY[N:] = Y_dot
    F = -f @ XY # np.hstack((X_dot, Y_dot))
    
    return F

def snake_robot_rhs(x, u):
    # print('rhs')
    theta = x[: N]
    p = x[N : N + 2]
    theta_dot = x[N + 2 : 2 * N + 2]
    p_dot = x[2 * N + 2 :]
    F = f_R(N, theta, theta_dot, p_dot)
    f_Rx = F[: N]
    f_Ry = F[N :]
    Tr = -lambda2 * theta_dot * 0

    a = np.linalg.inv(M_theta(N, theta)) @ (D(N).T @ u 
                                            - W(N, theta) @ theta_dot**2 + Tr + l * S_theta(theta) @ K(N) @ f_Rx 
                                            - l * C_theta(theta) @ K(N) @ f_Ry)
    
    b = 1/(m * N) * E(N).T @ F
    # b = E(N).T @ F
    # print('theta_dot = ', theta_dot)
    # print('p_dot = ', p_dot)
    # print('W = ', W(N, theta) @ theta_dot**2)
    # print('Rest = ', D(N).T @ u - W(N, theta) @ theta_dot**2 + Tr + l * S_theta(theta) @ K(N) @ f_Rx - l * C_theta(theta) @ K(N) @ f_Ry)
    # print(D(N).T @ u - W(N, theta) @ theta_dot**2 + l * S_theta(theta) @ K(N) @ f_Rx - l * C_theta(theta) @ K(N) @ f_Ry)
    # print('M = ', M_theta(N, theta))
    # print('M^-1 = ', np.linalg.inv(M_theta(N, theta)))
    # print('a = ', a)
    # print('M = ', M_theta(N, theta))
    # print('M^-1 = ', np.linalg.inv(M_theta(N, theta)))
    # print('Rest = ', (D(N).T @ u - W(N, theta) @ theta_dot**2 + l * S_theta(theta) @ K(N) @ f_Rx 
    #                             - l * C_theta(theta) @ K(N) @ f_Ry))
    # print(D(N).T @ u, W(N, theta) @ theta_dot**2, l * S_theta(theta) @ K(N) @ f_Rx, l * C_theta(theta) @ K(N) @ f_Ry)
    # print('b = ', b)
    # print('1/(m * N) = ', 1/(m * N), 'E(N).T = ', E(N).T , 'Fx = ', f_Rx, 'Fy = ', f_Ry)
    # print(E(N).T @ F)
    x_dot = np.hstack((np.hstack((theta_dot, p_dot)), np.hstack((a, b))))

    return x_dot

def snake_robot_rhs_casadi(x, u):
    # print('rhs')
    theta = x[: N]
    p = x[N : N + 2]
    theta_dot = x[N + 2 : 2 * N + 2]
    p_dot = x[2 * N + 2 :]
    F = f_R_casadi(N, theta, theta_dot, p_dot)
    f_Rx = F[: N]
    f_Ry = F[N :]

    a = solve(M_theta(N, theta), SX.eye(N)) @ (D(N).T @ u 
                                            - W(N, theta) @ theta_dot**2 + l * S_theta(theta) @ K(N) @ f_Rx 
                                            - l * C_theta(theta) @ K(N) @ f_Ry)
    
    b = 1/(m * N) * E(N).T @ F
    x_dot = SX.zeros(2 * N + 4)
    x_dot[: N] = theta_dot
    x_dot[N: N + 2] = p_dot
    x_dot[N + 2 : 2 * N + 2] = a
    x_dot[2 * N + 2 :] = b
    # x_dot = SX.hstack((SX.hstack((theta_dot, p_dot)), SX.hstack((a, b))))

    return x_dot   

def snake_robot_discrete(x, u, delt):
    theta = x[: N]
    p = x[N : N + 2]
    theta_dot = x[N + 2 : 2 * N + 2]
    p_dot = x[2 * N + 2 :]
    F = f_R(N, theta, theta_dot, p_dot)
    f_Rx = F[: N]
    f_Ry = F[N :]

    # print(np.shape(W(N, theta)), np.shape(theta_dot), np.shape(theta_dot**2))

    a = np.linalg.inv(M_theta(N, theta)) @ (D(N).T @ u 
                                            - W(N, theta) @ theta_dot**2 + l * S_theta(theta) @ K(N) @ f_Rx 
                                            - l * C_theta(theta) @ K(N) @ f_Ry)
    
    b = 1/(m * N) * E(N).T @ F

    x_plus = x + delt * np.hstack((np.hstack((theta_dot, p_dot)), np.hstack((a, b))))
    return x_plus   

# def manipulator(x, u):
#     theta = x[: N]
#     theta_dot = x[N :]

#     a = np.linalg.inv(M_theta(N, theta)) @ (D(N).T @ u 
#                                             - W(N, theta) @ theta_dot**2)
    
#     x_dot = np.hstack((theta_dot, a))

#     return x_dot

# def manipulator_discrete(x, u, delt):

#     theta = x[: N]
#     theta_dot = x[N :]

#     a = np.linalg.inv(M_theta(N, theta)) @ (D(N).T @ u 
#                                             - W(N, theta) @ theta_dot**2)
    
#     x_dot = x + delt * np.hstack((theta_dot, a))

#     return x_dot
