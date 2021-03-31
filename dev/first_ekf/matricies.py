import numpy as np
import sympy as sp
import sympy.abc as sps
from math import sin, cos, tan

Tc = 1 # Sampling period (change)
g_w = np.matrix([0,0,9.8]).T
n_w = np.matrix([1,0,0]).T

def R(theta):
    """

    Returns a 3 x 3 numpy matrix representating the total
    rotation matrix given theta, a 3 x 1 numpy matrix representing
    the current roll, pitch, and yaw angles

    """
    x = theta[0]
    y = theta[1]
    z = theta[2]
    rx = np.matrix([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    ry = np.matrix([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    rz = np.matrix([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    return rz*ry*rx # TODO: test

def U(theta):
    """

    Returns a 7 x 7 numpy matrix representing the output matrix 

    """
    U = np.matrix([R(theta).T, np.zeros(3,3), np.zeros(3,1)], 
                  [np.zeros(3,3), R(theta).T, np.zeros(3,1)],
                  [np.zeros(1,3), np.zeros(1,3), 1])
    return U

def E(theta):
    """

    Returns a 3 x 3 numpy matrix representing the system matrix
    at the gvien theta vector, a 3 x 1 numpy matrix representing
    the current roll, pitch, and yaw angles. 

    """
    r = theta[0] # roll
    p = theta[1] # pitch
    y = theta[2] # yaw
    E = -1*np.matrix([[1, sin(r)*tan(p), cos(r)*tan(p)],
                      [0, cos(r), -sin(r)],
                      [0, sin(r)/cos(p), cos(r)/cos(p)]])

    return E

def F(theta, w):
    """

    Returns a 3 x 3 numpy matrix representing the Jacobian of the system
    matrix w.r.t. the the theta vector given the t vector, a 3 x 1 numpy
    matrix representing the current roll, pitch, and yaw angles and w, a
    3 x 1 numpy matrix representing the current angular velocities

    """
    E_times_wr = sp.Matrix([sps.x + sp.sin(sps.r)*sp.tan(sps.p)*sps.y + sp.cos(sps.r)*sp.tan(sps.p)*sps.z,
                            sp.cos(sps.r)*sps.y + -1*sp.sin(sps.r)*sps.z,
                            (sp.sin(sps.r)*sps.y)/sp.cos(sps.p) + (sp.cos(sps.r)*sps.z)/sp.cos(sps.p)])
    theta_vec = sp.Matrix([sps.r, sps.p, sps.a])
    J = E_times_wr.jacobian(theta_vec) # unevaluated Jacobian
    print(J) # for testing
    J_eval = E_times_wr.jacobian(theta_vec).subs([(sps.r,theta[0]),
                                                  (sps.p,theta[1]),
                                                  (sps.a,theta[2]),
                                                  (sps.x,w[0]),
                                                  (sps.y,w[1]),
                                                  (sps.z,w[2])])
    J_eval = np.matrix(J_eval) # convert Jacobian to numpy matrix
    F = np.identity(3) + Tc*J_eval
    return F

def G(theta, w):
    """

    Returns a 3 x 3 numpy matrix representing the Jacobian of the system
    matrix given given the theta vector, a a 3 x 1 numpy matrix representing
    the current roll, pitch, and yaw angles and w, a 3 x 1 numpy matrix
    representing the current angular velocities

    """
    return Tc*E(theta)

def Q_w(w):
    """

    Returns a 3 x 3 numpy matrix covariance matrix associated with w including
    both intrinsic and measurement uncertainty contributions

    """
    Q = np.cov(w) # TODO: check
    return Q

def H(theta):
    """

    Returns a 3 x 3 numpy matrix representing the Jacobian of the
    output matrix U * o_w

    """
    H = pd('U', 'theta', theta)*o_w(theta[2])
    return H

def o_w(yaw):
    """

    Returns a 6 x 1 numpy matrix containing the gravitational acceleration
    vector, the north directed unit vector and the current yaw

    """
    o = np.matrix([g_w.T, n_w.T, yaw]).T
    return o

def Q_o():
    """

    Returns a the covariance matrix of the current measurement vector o_r

    """
    return 0 # TODO

def o_r(g_r, n_r, yaw_r):
    """

    Returns a 6 x 1 numpy matrix containing the current measurement vector

    """
    o = np.matrix([g_r.T, n_r.T, yaw_r]).T
    return o

def main():
    # TEST CASES
    # w = np.matrix([1,1,1]).T
    # f = F([0, 0, 0], w)
    print(o_w(5))

if __name__ == "__main__":
    main()