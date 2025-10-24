#Covert global homo into mesh
def H2Mesh(H2, patch_size):
    batch_size = tf.shape(H2)[0]
    h = patch_size/grid_h
    w = patch_size/grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh, 1], shape=[3], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(tf.expand_dims(p, 0),2))
    ori_pt = tf.concat(ori_pt, axis=2)
    ori_pt = tf.tile(ori_pt,[batch_size, 1, 1])
    tar_pt = tf.matmul(H2, ori_pt)
    
    x_s = tf.slice(tar_pt, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(tar_pt, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(tar_pt, [0, 2, 0], [-1, 1, -1])

    H2_local = tf.concat([x_s/z_s, y_s/z_s], axis=1)
    H2_local = tf.transpose(H2_local, perm=[0, 2, 1])
    H2_local = tf.reshape(H2_local, [batch_size, grid_h+1, grid_w+1, 2])
    
    return H2_local