import numpy as np
import tensorflow as tf

def angle_between(v1, v2, epsilon=1e-5):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    # print_v1 = tf.Print(v1, [v1])
    # print_v2 = tf.Print(v2, [v2])
    
    v1_u = tf.nn.l2_normalize(v1)
    v2_u = tf.nn.l2_normalize(v2)
    return tf.acos(tf.clip_by_value(tf.matmul(tf.transpose(v1_u), v2_u), -1.0 + epsilon, 1.0 - epsilon))

def g(u,v):
    return get_arccos_integral(u,v)

def get_arccos_integral(v1, v2):
    # 1/pi  * norm(v1)*norm(v2) * (sin(alpha) + (pi-alpha)cos(alpha))
    angle = angle_between(v1, v2)
    return ((tf.norm(v1) * tf.norm(v2)) / np.pi) * (tf.sin(angle) + (np.pi - angle)*tf.cos(angle))

def get_very_sparse_regularizer(A, W, THETA):
        # A, W, THETA = self.get_weights()
        # _W, _THETA = array_ops.split(self._kernel, num_or_size_splits=2, axis=0)
        # W = tf.transpose(_W)
        # THETA = tf.transpose(_THETA)
        # THETA = tf.Print(tmp_THETA, [tmp_THETA, W], 'weight vectors:')
        # print(self._kernel)
        # print(W, THETA)
        
        aa = tf.matmul(A, A, transpose_a=True)
        g_w_w = get_arccos_integral(W, W)
        g_w_theta = get_arccos_integral(W, THETA)
        g_theta_theta = get_arccos_integral(THETA, THETA)
        reg_loss = aa*(g_w_w - 2*g_w_theta + g_theta_theta)
        return tf.squeeze(reg_loss)

def get_expectation(A, W, THETA):
    h, d = W.get_shape().as_list()
    Q = tf.matmul(tf.transpose(A), A)
    total_expectation = 0
    for i in range(h):
        w_i = tf.reshape(W[i,:], [d, 1])
        theta_i = tf.reshape(THETA[i,:], [d, 1])
        u_i = tf.concat([theta_i, w_i], axis=0)
        v_i = tf.concat([w_i, theta_i], axis=0)
        for j in range(h):
            w_j = tf.reshape(W[j,:], [d, 1])
            theta_j = tf.reshape(THETA[j,:], [d, 1])
            u_j = tf.concat([theta_j, w_j], axis=0)
            v_j = tf.concat([w_j, theta_j], axis=0)
            total_expectation += Q[i,j]*(g(u_i, u_j) - 2*g(u_i, v_j) + g(v_i, v_j))
    return tf.squeeze(total_expectation)

def get_expectation_v2(A, W, THETA):
    h, d = W.get_shape().as_list()
    Q = tf.matmul(tf.transpose(A), A)
    total_expectation = 0
    print('building regularizer')
    for i in range(h):
        print('row {}'.format(i))
        w_i = tf.reshape(W[i,:], [d, 1])
        theta_i = tf.reshape(THETA[i,:], [d, 1])
        u_i = tf.concat([theta_i, w_i], axis=0)
        v_i = tf.concat([w_i, theta_i], axis=0)
        for j in range(i+1):
            w_j = tf.reshape(W[j,:], [d, 1])
            theta_j = tf.reshape(THETA[j,:], [d, 1])
            u_j = tf.concat([theta_j, w_j], axis=0)
            v_j = tf.concat([w_j, theta_j], axis=0)
            if j < i:
                factor = 2
            else:
                factor = 1
            total_expectation += factor*2*Q[i,j]*(g(u_i, u_j) - g(u_i, v_j))
    return tf.squeeze(total_expectation)

def get_expectation_v3(A, W, THETA, epsilon=1e-4):
    def G(U, V, norm_matrix):
        inner_prod_mat = tf.matmul(U, V, transpose_b=True)
        cos_alpha = tf.divide(inner_prod_mat, norm_matrix + epsilon)
        alpha = tf.acos(cos_alpha)
        return (norm_matrix/np.pi) * (tf.sin(alpha) + (np.pi - alpha) * cos_alpha)
        # N = tf.sqrt(tf.matmul(U,U, transpose_a=True)) * tf.sqrt(tf.matmul(V,V, transpose_a=True))

    # h, d = W.get_shape().as_list()
    Q = tf.matmul(A, A, transpose_a=True)
    U = tf.concat([THETA, W], axis=1)
    V = tf.concat([W, THETA], axis=1)
    # the norms of row i of U (they're the same for V)
    norm_per_row = tf.norm(U, axis=1, keepdims=True)
    
    # norm_matrix_ij = norm(u_i)*norm(u_j)
    norm_matrix = tf.matmul(norm_per_row, norm_per_row, transpose_b=True)

    UU = G(U, U, norm_matrix)
    UV = G(U, V, norm_matrix)

    return tf.reduce_sum(2*tf.multiply(Q, UU-UV))


if __name__=='__main__':
    from CommutativeRNNcell import CommutativeRNNcell
    # A = tf.Variable(np.random.random((10, 256)))
    # A = tf.Variable(np.random.random((10, 80)))
    A = tf.Variable(np.array([1,1,-1]).reshape(1,3), dtype=tf.float32)

    W = tf.Variable(np.array([1,0,0]).reshape(3,1), dtype=tf.float32)
    # W = tf.Variable(np.random.random((80, 10)))
    THETA = tf.Variable(np.array([-1,1,-1]).reshape(3,1), dtype=tf.float32)
    # THETA = tf.Variable(np.random.random((80, 10)))
    # A = tf.Variable(np.random.random((10, 10)))
    # W = tf.Variable(np.random.random((10, 10)))
    # THETA = tf.Variable(np.random.random((10, 10)))
    # reg_v1 = get_expectation(A, W, THETA)
    reg_v2 = get_expectation_v2(A, W, THETA)
    reg_v3 = get_expectation_v3(A, W, THETA)

    U = tf.transpose(tf.concat([THETA, W], axis=1))
    V = tf.transpose(tf.concat([W, THETA], axis=1))
    # print(U, V)
    # exit()
    guv = get_arccos_integral(U, V)
    guu = get_arccos_integral(U, U)
    gvv = get_arccos_integral(V, V)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    print(sess.run([guv, guu, gvv]))
    print(sess.run([reg_v2, reg_v3]))
