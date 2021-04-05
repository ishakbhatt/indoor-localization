import numpy as np
import sympy as sp
import sympy.abc as sps
from math import sin, cos, tan

Tc = 1 # Sampling period (change)
g_w = np.array([0,0,9.8]).T
n_w = np.array([1,0,0]).T

def R(theta):
    """

    Returns a 3 x 3 numpy array representating the total
    rotation matrix given theta, a 3 x 1 numpy matrix representing
    the current roll, pitch, and yaw angles

    """
    x = theta[0]
    y = theta[1]
    z = theta[2]
    rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    return rz*ry*rx # TODO: test

def U_with_yaw(theta):
    """

    Returns a 7 x 7 numpy array representing the output matrix

    Used when the yaw angle estimation is given

    """
    l_block = np.append(np.append(R(theta).T, np.zeros((3,3)), axis=0), np.zeros((1,3)), axis=0)
    m_block = np.append(np.append(np.zeros((3,3)), R(theta).T, axis=0), np.zeros((1,3)), axis=0)
    r_block = np.zeros((7,1))
    U = np.concatenate((np.concatenate((l_block, m_block), axis=1), r_block), axis=1)
    return U

def U_with_out_yaw(theta):
    """

    Returns a 6 x 6 numpy array representing the output matrix

    Used when the yaw angle estimation is not given

    """
    l_block = np.append(R(theta).T, np.zeros((3,3)), axis=0)
    r_block = np.append(np.zeros((3,3)), R(theta).T, axis=0)
    U = np.concatenate((l_block, r_block), axis=1)
    return U

def E(theta):
    """

    Returns a 3 x 3 numpy array representing the system matrix
    at the gvien theta vector, a 3 x 1 numpy array representing
    the current roll, pitch, and yaw angles. 

    """
    r = theta[0] # roll
    p = theta[1] # pitch
    y = theta[2] # yaw
    E = -1*np.array([[1, sin(r)*tan(p), cos(r)*tan(p)],
                      [0, cos(r), -sin(r)],
                      [0, sin(r)/cos(p), cos(r)/cos(p)]])

    return E

def F(theta, w):
    """

    Returns a 3 x 3 numpy array representing the Jacobian of the system
    matrix w.r.t. the the theta vector given the t vector, a 3 x 1 numpy
    array representing the current roll, pitch, and yaw angles and w, a
    3 x 1 numpy array representing the current angular velocities

    """
    E_times_wr = sp.Matrix([sps.x + sp.sin(sps.r)*sp.tan(sps.p)*sps.y + sp.cos(sps.r)*sp.tan(sps.p)*sps.z,
                            sp.cos(sps.r)*sps.y + -1*sp.sin(sps.r)*sps.z,
                            (sp.sin(sps.r)*sps.y)/sp.cos(sps.p) + (sp.cos(sps.r)*sps.z)/sp.cos(sps.p)])
    theta_vec = sp.Matrix([sps.r, sps.p, sps.a])
    J = E_times_wr.jacobian(theta_vec) # unevaluated Jacobian
    print("UNEVALUATED JACOBIAN")
    print(J) # for testing
    print("---------------")
    J_eval = E_times_wr.jacobian(theta_vec).subs([(sps.r,theta[0]),
                                                  (sps.p,theta[1]),
                                                  (sps.a,theta[2]),
                                                  (sps.x,w[0]),
                                                  (sps.y,w[1]),
                                                  (sps.z,w[2])])
    J_eval = np.array(J_eval) # convert Jacobian to numpy matrix
    F = np.identity(3) + Tc*J_eval
    return F

def G(theta, w):
    """

    Returns a 3 x 3 numpy array representing the Jacobian of the system
    matrix given given the theta vector, a a 3 x 1 numpy array representing
    the current roll, pitch, and yaw angles and w, a 3 x 1 numpy array
    representing the current angular velocities

    """
    return Tc*E(theta)

def Q_w(w_r):
    """

    Returns a 3 x 3 numpy array covariance matrix associated with w including
    both intrinsic and measurement uncertainty contributions

    """
    # TODO: read in measurement data

    # TODO: parse into 3 vectors: x, y, and z
    x = (np.random.rand(1,500)).squeeze(0)
    y = (np.random.rand(1,500)).squeeze(0)
    z = (np.random.rand(1,500)).squeeze(0)

    # compute the covariance with the numpy function
    Q = np.cov([x,y,z])
    Q_placeholder = np.identity(3) # assume low variance
    return Q_placeholder

def Q_g(g_r):
    """

    Returns a 3 x 3 numpy array correponding to the covariance matrix
    of the gravitational acceleration vector g_r (from accelerometer)

    """
    # TODO: read in measurement data

    # TODO: parse into 3 vectors: x, y, and z
    x = (np.random.rand(1,500)).squeeze(0)
    y = (np.random.rand(1,500)).squeeze(0)
    z = (np.random.rand(1,500)).squeeze(0)

    # compute the covariance with the numpy function
    Q = np.cov([x,y,z])
    Q_placeholder = np.identity(3)
    return Q_placeholder

def Q_n(n_r):
    """

    Returns a 3 x 3 numpy array correponding to the covariance matrix
    of the north-directed unit vector (from the compass)

    """
    # TODO: read in measurement data

    # TODO: parse into 3 vectors: x, y, and z
    x = (np.random.rand(1,500)).squeeze(0)
    y = (np.random.rand(1,500)).squeeze(0)
    z = (np.random.rand(1,500)).squeeze(0)

    # compute the covariance with the numpy function
    Q = np.cov([x,y,z])
    Q_placeholder = np.identity(3)
    return Q_placeholder

def H_with_yaw(theta, yaw):
    """

    Returns a 7 x 3 numpy array representing the Jacobian of the
    output matrix U multiplied by o_w

    Used when the yaw angle estimation is given

    """
    U_times_ow = sp.Matrix([-9.8*sp.sin(sps.y),
                             9.8*sp.cos(sps.y)*sin(sps.x),
                             9.8*sp.cos(sps.y)*sp.cos(sps.x),
                             sp.cos(sps.z)*sp.cos(sps.y),
                             sp.cos(sps.z)*sp.sin(sps.y) - sp.sin(sps.z)*sp.cos(sps.x),
                             sp.sin(sps.z)*sp.sin(sps.x) + sp.cos(sps.z)*sp.sin(sps.y)*sp.cos(sps.x),
                             yaw])
    theta_vec = sp.Matrix([sps.x, sps.y, sps.z])
    J = U_times_ow.jacobian(theta_vec) # unevaluated Jacobian
    J_eval = U_times_ow.jacobian(theta_vec).subs([theta[0],
                                                 theta[1],
                                                 theta[2]])
    J_eval = np.array(J_eval) # convert Jacobian to numpy array
    return J_eval

def H_with_out_yaw(theta):
    """

    Returns a 6 x 3 numpy array representing the Jacobian of the
    output matrix U multiplied by o_w

    Used when the yaw angle estimation is not given

    """
    U_times_ow = sp.Matrix([-9.8*sp.sin(sps.y),
                            9.8*sp.cos(sps.y)*sp.sin(sps.x),
                            9.8*sp.cos(sps.y)*sp.cos(sps.x),
                            sp.cos(sps.z)*sp.cos(sps.y),
                            sp.sin(sps.z)*sp.sin(sps.x)+sp.cos(sps.z)*sp.sin(sps.y)*sp.cos(sps.x)])
    theta_vec = sp.Matrix([sps.x, sps.y, sps.z])
    J = U_times_ow.jacobian(theta_vec)
    print("UNEVALUATED JACOBIAN")
    print(J) # for testing
    print("---------------")
    J_eval = U_times_ow.jacobian(theta_vec).subs([(sps.x,theta[0]),
                                                  (sps.y,theta[1]),
                                                  (sps.z,theta[2])])
    J_eval = np.array(J_eval) # convert Jacobian to numpy array
    return J_eval

def o_w(yaw):
    """

    Returns a 7 x 1 numpy array containing the gravitational acceleration
    vector, the north directed unit vector and the current yaw

    """
    o = np.array([0, 0, 9.8, 1, 0, 1, yaw])
    return o

def Q_o_with_out_yaw(g_r, n_r):
    """

    Returns a 6 x 6 numpy array representing the covariance matrix
    of the current measurement vector o_r

    """
    g_cov = Q_g(g_r)
    n_cov = Q_n(n_r)
    top = np.append(g_cov, np.zeros((3,3)), axis=1)
    bottom = np.append(np.zeros((3,3)), n_cov, axis=1)
    Q_o = np.concatenate((top, bottom))
    return Q_o

def o_r(g_r, n_r, yaw_r):
    """

    Returns a 7 x 1 numpy array containing the current measurement vector

    """
    o = np.append(g_r, n_r)
    o = np.append(o, yaw_r)
    return o

def main():
    # TEST CASES
    # w = np.array([1,1,1]).T
    # f = F([0, 0, 0], w)
    print(o_w(5))

if __name__ == "__main__":
    main()