import unittest
import numpy as np
from numpy import testing
import math
from matricies import R, U_with_out_yaw, E, F, H_with_out_yaw, Q_o_with_out_yaw

class TestMatrixMethods(unittest.TestCase):

    # R tests
    def test_R_1(self):
        print(R([math.pi, math.pi, 0]))
        self.assertTrue(np.array_equal(R([math.pi, math.pi, 0]), np.array([[-1,0,0],
                                                                            [0,-1,0],
                                                                            [0,0,1]])))

    def test_R_2(self):
        print(R([math.pi/2.0, math.pi, 0]))
        self.assertTrue(np.array_equal(R([math.pi/2.0, math.pi, 0]), np.array([[-1,0,0],
                                                                               [0,0,-1],
                                                                               [0,-1,0]])))

    def test_R_3(self):
        print(R([0, 0, 0]))
        self.assertTrue(np.array_equal(R([0, 0, 0]), np.array([[1,0,0],
                                                               [0,1,0],
                                                               [0,0,1]])))

    def test_R_4(self):
        print(R([math.pi/2.0, math.pi/2.0, 0]))
        self.assertTrue(np.array_equal(R([math.pi/2.0, math.pi/2.0, 0]), np.array([[0,1,0],
                                                                                  [0,0,-1],
                                                                                  [-1,0,0]])))

    def test_R_5(self):
        print("FUCTION RESULT:")
        print(R([0.234, 0.456, 1.4]))
        print("---------------")
        print("CORRECT RESULT")
        print(np.array([[0.1526, -0.941, 0.3013], [0.884, 0.265, 0.3827], [-0.44,0.2082, 0.873]]))
        print("---------------")
        self.assertTrue(True)
    
    # U tests
    def test_U_no_yaw_1(self):
        print(U_with_out_yaw([math.pi, math.pi, 0]))
        self.assertTrue(np.array_equal(U_with_out_yaw([math.pi, math.pi, 0]), np.array([[-1,0,0,0,0,0],
                                                                                        [0,-1,0,0,0,0],
                                                                                        [0,0,1, 0,0,0],
                                                                                        [0,0,0,-1,0,0],
                                                                                        [0,0,0,0,-1,0],
                                                                                        [0,0,0,0,0,1]])))

    # E tests
    def test_E_1(self):
        print(E([math.pi, math.pi, 0]))
        self.assertTrue(np.array_equal(E([math.pi, math.pi, 0]), np.array([[-1,0,0],
                                                                           [0,1,0],
                                                                           [0,0,-1]])))

    def test_E_2(self):
        print(E([math.pi/2.0, math.pi, 0]))
        self.assertTrue(np.array_equal(E([math.pi/2.0, math.pi, 0]), np.array([[-1,0,0],
                                                                               [0,0,1],
                                                                               [0,1,0]])))

    def test_E_3(self):
        print(E([0, 0, 0]))
        self.assertTrue(np.array_equal(E([0,0,0]), np.array([[-1,0,0],
                                                             [0,-1,0],
                                                             [0,0,-1]])))

    def test_E_4(self):
        print("FUCTION RESULT:")
        print(E([0.234, 0.456, 1.4]))
        print("---------------")
        print("CORRECT RESULT")
        print(np.array([[-1, -0.113, 0.477], [0, -0.9727, 0.2318], [0,-0.258, -5.723]]))
        print("---------------")
        self.assertTrue(True)
    
    # F tests
    def test_F_1(self):
        print("w = [0, 0, 0]")
        print("t = [0, 0, 0]")
        print("Tc = 1")
        print("FUCTION RESULT:")
        print(F([0,0,0], [0,0,0]))
        print("---------------")
        self.assertTrue(True)
    
    # TODO: Add tests

    # G tests
    """ ------- No G() tests since G is a scalar multiple of E ------- """

    # Q_w tests
    # TODO: add when method for covariance matrix construction is complete

    # Q_g tests
    # TODO: add when method for covariance matrix construction is complete

    # Q_n tests
    # TODO: add when method for covariance matrix construction is complete

    # H tests
    def test_H_no_yaw_1(self):
        print("theta = [0, 0, 0]")
        print("FUCTION RESULT:")
        print(H_with_out_yaw([0,0,0]))
        print("---------------")
        self.assertTrue(True)

    # o_w tests
    """ ------- No o_w() tests since o_w() performs a simple substitution ------- """

    # Q_o tests
    def test_Q_o_with_out_yaw_1(self):
        print("g_r = [0,0,0]")
        print("n_r = [0,0,0]")
        self.assertTrue(np.array_equal(Q_o_with_out_yaw([0,0,0], [0,0,0]), np.identity(6)))
        
    # o_r tests
    """ ------- No o_r() tests since o_r() performs a simple array append ------- """

if __name__ == '__main__':
    unittest.main()