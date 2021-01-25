def custom_loss_2(y_true, y_pred):
    #get len grid
    # output 13 as grid (biar flexibel)
    n_cells = y_pred.get_shape().as_list()[1]

    #ytrue and ypred every batch
    y_true = tf.reshape(y_true, tf.shape(y_pred), name='y_true')
    y_pred = tf.identity(y_pred, name='y_pred')

    # numpy buat nampung x-y pred, shape = batch_size,grid,grid,2(x,y)
    predicted_xy = tf.nn.sigmoid(y_pred[..., :2])

    # set value xy mengikuti grid x dan y
    cell_inds = tf.range(n_cells, dtype=tf.float32)
    # cell_inds output [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]

    # set value xy mengikuti nilai grid x dan y sesuai urutan vector (selalu berada pada pusat grid) misal, berada di grid 0 maka set nilai 0.5 dst samapi 12.5
    '''contoh [batch_i][5 as y][7 as x] output 
    [[7.5 5.5]
     [7.5 5.5]
     [7.5 5.5]
     [7.5 5.5]
     [7.5 5.5]] predicted_xy'''
    predicted_xy = tf.stack((
        predicted_xy[..., 0] + tf.reshape(cell_inds, [1, -1, 1]),
        predicted_xy[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])
    ), axis=-1)


    # set value wh dimana 5 nilai wh pada semua anchor di grid mengikuti nilai anchor sesuai urutan
    '''contoh [batch_i][5 as y][6 as x] output  [[5.42265442e+110 2.39281903e+133]
 [1.26952956e+000 4.57130596e+000]
 [4.34356520e+000 9.05625981e+000]
 [1.77578193e+000 6.61574849e+000]
 [7.82103790e-001 2.89223844e+000]] predicted_wh'''
    # predicted_wh = ANCHOR_BOX * tf.exp(y_pred[..., 2:4])
    predicted_wh = ANCHOR_BOX * tf.exp(y_pred[..., 2:4])



    # get nilai predict min dan max untuk proses iou antara prediksi dengan ground truth
    predicted_min = predicted_xy - predicted_wh / 2
    predicted_max = predicted_xy + predicted_wh / 2

    #get nilai objectnes pada setiap grid anchor predicted_objectedness bernilai 0.5 karena hasil sigmoid(0)
    '''contoh [batch_i][5 as y][6 as x] output
    0.73105858 0.5        0.5        0.5        0.5       ] predicted_objectedness'''
    predicted_objectedness = tf.nn.sigmoid(y_pred[..., 4])

    #get xy ground truth
    true_xy = y_true[..., :2]
    # get wh ground truth
    # +ANCHOR_BOX karena delta x & y
    true_wh = y_true[..., 2:4]+ANCHOR_BOX
    

    # get nilai true min dan max dari ground truth untuk proses iou  ground truth dgn prediksi
    true_min = true_xy - true_wh / 2
    true_max = true_xy + true_wh / 2



    # get iou antara ground truth dengan prediksi, untuk set nilai objectnes
    intersect_mins = tf.maximum(predicted_min, true_min)
    intersect_maxes = tf.minimum(predicted_max, true_max)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    #get IOU score tiap anchor
    '''contoh [batch_i][5 as y][6 as x] output [5.97051471e-240 0.00000000e+000 0.00000000e+000 0.00000000e+000
 0.00000000e+000] iou_scores'''
    iou_scores = intersect_areas / union_areas

    #get nilai objectnes dari ground truth setiap anchor
    responsibility_selector = y_true[..., 4]

    #vetcor nilai loss xy prediksi, dimana jika objectnes anchor =1 maka akan memiliki nilai loss xy, namun jika objectnes anchor =0 maka memiliki nilai 0
    ''' [[24336.  2601.]
  [    0.     0.]
  [    0.     0.]
  [    0.     0.]
  [    0.     0.]]'''
    xy_diff = tf.square(true_xy - predicted_xy) * responsibility_selector[..., None]
    #total xy loss dalam 1 citra
    xy_loss = tf.reduce_sum(xy_diff, axis=[1, 2, 3, 4])

    # vetcor nilai loss wh prediksi, dimana jika objectnes anchor =1 maka akan memiliki nilai loss wh, namun jika objectnes anchor =0 maka memiliki nilai 0
    '''[[5.42265442e+110 2.39281903e+133]
  [0.00000000e+000 0.00000000e+000]
  [0.00000000e+000 0.00000000e+000]
  [0.00000000e+000 0.00000000e+000]
  [0.00000000e+000 0.00000000e+000]]'''
    wh_diff = tf.square(tf.sqrt(true_wh) - tf.sqrt(predicted_wh)) * responsibility_selector[..., None]
    # total wh loss dalam 1 citra
    wh_loss = tf.reduce_sum(wh_diff, axis=[1, 2, 3, 4])

    # vector nilai loss objek prediksi, dimana jika objectnes anchor =1 maka akan memiliki nilai loss objek, namun jika objectnes anchor =0 maka memiliki nilai 0
    '''[[0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.53444665 0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.        ]]'''
    obj_diff = tf.square(iou_scores - predicted_objectedness) * responsibility_selector
    # total obj loss dalam 1 citra
    obj_loss = tf.reduce_sum(obj_diff, axis=[1, 2, 3])

    #get best iou tiap grid
    '''contoh [batch_i][5 as y] output [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
 0.00000000e+000 0.00000000e+000 5.97051471e-240 0.00000000e+000
 0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
 0.00000000e+000]'''
    best_iou = tf.reduce_max(iou_scores, axis=-1)

    temp=tf.compat.v1.to_float(best_iou < 0.6)[..., None]
    # temp_2=tf.cast(temp, dtype=tf.float64)
    # vector nilai loss no_obj , dimana jika objectnes anchor =1 maka akan memiliki nilai 0 loss no obj, namun jika objectnes anchor =0 maka memiliki nilai loss 0.25
    '''[[0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.   0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]
 [0.25 0.25 0.25 0.25 0.25]]'''
    #(1 - responsibility_selector) membuat nilai loss no obj 0 jika GT=1 karena (1 - responsibility_selector(GT)(0/1)),
    # no_obj_diff = tf.square(0 - predicted_objectedness) * temp_2 * (1 - responsibility_selector)
    no_obj_diff = tf.square(0 - predicted_objectedness) * tf.compat.v1.to_float(best_iou < 0.6)[..., None] * (1 - responsibility_selector)
    #total all vector no obj
    no_obj_loss = tf.reduce_sum(no_obj_diff, axis=[1, 2, 3])


    #Total all LOSS
    loss = object_coord_scale * (xy_loss + wh_loss) + object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss


    # '''
    # Kesimpulan :
    # 1.nilai prediksi x dan y diset 0-12 mengkuti vektor jumlah grid,
    # 2.nilai prediksi w dan h (tiap anchor) pada masing masing grid mengikuti nilai anchor yang dibuat ketika training sesuai urutan
    # 3.
    # '''
  
   

    return loss
