# -*- coding: utf-8 -*-
"""train Model(14 April).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rvrx4gCvWiGdF1IY1_fZFTibs_SO-GD0
"""

from google.colab import drive
drive.mount('/content/drive')

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam
from keras import regularizers, initializers
import tensorflow as tf
import tensorflow.keras
import numpy as np
import copy
import cv2
import os

folder_citra_train_anno = "/content/drive/My Drive/KULIAH/TA/PROGRAM/dataset/INRIAPerson/Train/annotations"
folder_citra_test_anno = "/content/drive/My Drive/KULIAH/TA/PROGRAM/dataset/INRIAPerson/Test/annotations"

#anchor box dari kmean

# [[0.07578406 0.28114057]
#  [0.13725029 0.48540107]
#  [0.26304421 0.68111173]]

ANCHOR_BOX=np.array([[0.20392486 ,0.63848327],
 [0.09765612, 0.35163892],
 [0.3341204,  0.69663537],
 [0.13659861, 0.50890373],
 [0.06016183, 0.22247988]])

ANCHOR_BOX = ANCHOR_BOX * 416 / 32
BATCH_SIZE=64

object_coord_scale = 1
object_conf_scale = 5
noobject_conf_scale = 1

def parse_anotasi(annotasi_dir):
    
    all_img = []
    for file_annotasi in sorted(os.listdir(annotasi_dir)):
     
        temp = open(annotasi_dir+ '/' + file_annotasi, "r",encoding="ISO-8859-1")
        label_percitra = {}
        temp_labelobj = []

        for line in temp:
            if 'Image filename' in line:
                a = (line.index('"'))
                temp_lokasi_citra = (line[a + 1:-2])
                label_percitra['filename'] = '/content/drive/My Drive/KULIAH/TA/PROGRAM/dataset/INRIAPerson/'+temp_lokasi_citra

            if 'Image size' in line:
                w = line[25:29]
                h = line[31:35]
                label_percitra['height'] = h
                label_percitra['width'] = w

            if 'Center point on object' in line:
                if 'person' in line:
                    temp_label = 'person'


                ##CARI ANOTASI BOUNDING BOX
                for line_2 in temp:
                    a = line_2.index(":")
                    temp_line_xyminmax = line_2[a + 2:]
                    for char in temp_line_xyminmax:
                        if char in "()-,\n":
                            temp_line_xyminmax = temp_line_xyminmax.replace(char, '')
                    temp_line_xyminmax = temp_line_xyminmax.split(" ")
                    xmin = temp_line_xyminmax[0]
                    ymin = temp_line_xyminmax[1]
                    xmax = temp_line_xyminmax[3]
                    ymax = temp_line_xyminmax[4]
                    break

                temp_dic = {}

                temp_dic['xmin'] = xmin
                temp_dic['ymin'] = ymin
                temp_dic['xmax'] = xmax
                temp_dic['ymax'] = ymax
                temp_labelobj.append(temp_dic)

            label_percitra['object'] = temp_labelobj

        all_img.append(label_percitra)

    return all_img



class praposes_citra(object):
    def __init__(self):
        self.W_citra_reshape=416
        self.H_citra_reshape = 416

    def get_lokasi_file(self,data_train):
        image_name = data_train['filename']
        return (image_name)

    def reshape_citra(self,data_train):
        image_name=self.get_lokasi_file(data_train)
        # print(image_name)
        image = cv2.imread(image_name)

        # get ukuran citra ori
        H_citra_ori, W_citra_ori, C_citra_ori = image.shape

        #reshape ukuran citra to 416x416
        image=cv2.resize(image,(self.H_citra_reshape,self.W_citra_reshape))

        #mengenbalikan urutan warna
        image = image[:, :, ::-1]

        #get dict
        all_objs = copy.deepcopy(data_train['object'])
        all_ground_truth=[]
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                #reshape value (penyesuaian ukuran) xmin sama xmax
                obj[attr] = int(int(obj[attr]) * float(self.W_citra_reshape) / W_citra_ori)
            for attr in ['ymin','ymax']:
                #reshape value (penyesuaian ukuran) ymin sama ymax
                obj[attr] = int(int(obj[attr]) * float(self.H_citra_reshape) / H_citra_ori)
            # xmin,ymin,xmax,ymax sudah dinormalisasi (/ukuran w,h asli citra ketika reshape)
            # /416 dinormalkan lagi,416 ukuran w,h citra stlh reshape

            xc=  (obj['xmin'] + obj['xmax'])/2
            yc = (obj['ymin'] + obj['ymax']) / 2
            wc = obj['xmax'] - obj['xmin']
            hc = obj['ymax'] - obj['ymin']


            all_ground_truth.append([xc,yc,wc,hc])

        return image,all_objs,all_ground_truth
       

class best_anchor(object):
    def get_best_anchor(list_ground_truth, matrix,grid_x,grid_y):
        # index list_ground_truth [xg,yg,wg,hg]
        temp_all_iou = []
        for anchor_i in range(len(ANCHOR_BOX)):
            # cek nilai obj anchor yg 0, jika != 0 maka pake anchor yang lain
            if matrix[grid_y,grid_x,anchor_i,4] == 0:
                xg=list_ground_truth[0]
                yg=list_ground_truth[1]
                wg=list_ground_truth[2]
                hg=list_ground_truth[3]
                anchor_wg=ANCHOR_BOX[anchor_i][0]
                anchor_hg=ANCHOR_BOX[anchor_i][1]
                temp_all_iou.append(best_anchor.IOU([xg,yg,wg,hg], [0.5,0.5,anchor_wg,anchor_hg]))

            else:
                temp_all_iou.append(0)

        iou_terbaik = max(temp_all_iou)
        anchor_terbaik = temp_all_iou.index(iou_terbaik)
        # print("anchor terbaik ke",temp_all_iou.index(max(temp_all_iou)),"all iou",temp_all_iou,"anchor box yang dipakai",ANCHOR_BOX[temp_all_iou.index(max(temp_all_iou))])
        return anchor_terbaik, iou_terbaik

    def IOU(box1, box2):
        # index list_ground_truth [xg,yg,wg,hg]

        A_xmin=box1[0]-(box1[2]/2)
        A_xmax = box1[0] + (box1[2] / 2)
        A_ymin = box1[1] - (box1[3] / 2)
        A_ymax = box1[1] + (box1[3] / 2)

        B_xmin = box2[0] - (box2[2] / 2)
        B_xmax = box2[0] + (box2[2] / 2)
        B_ymin = box2[1] - (box2[3] / 2)
        B_ymax = box2[1] + (box2[3] / 2)




        Box_Intersect_xmin = max(A_xmin, B_xmin)
        Box_Intersect_ymin = max(A_ymin, B_ymin)
        Box_Intersect_xmax = min(A_xmax, B_xmax)
        Box_Intersect_ymax = min(A_ymax, B_ymax)

        intersection_area = (max(Box_Intersect_xmin,Box_Intersect_xmax)-min(Box_Intersect_xmin,Box_Intersect_xmax)) * (max(Box_Intersect_ymin,Box_Intersect_ymax)-min(Box_Intersect_ymin,Box_Intersect_ymax))
        object_area =box1[2]*box1[3]
        anchor_area = box2[2]*box2[3]
        union = object_area + anchor_area
        iou = intersection_area / (union - intersection_area)
        # print(iou,"iou",obj['wg'],obj['hg'],"ke",anchor_ke)

        return iou




class buat_data_target(object):
    def __init__(self,all_ground_truth,all_obj):
        self.size_grid=416/13
        self.all_obj=all_obj
        self.all_ground_truth=all_ground_truth
        self.mapping_grid()


    


    def mapping_grid(self):
        temp_matrix = np.zeros((13, 13, len(ANCHOR_BOX), 5))
        a=0
        for obj in self.all_ground_truth:
            # index list_ground_truth [xg,yg,wg,hg]

            
            # # get kordinat grid ---Cara Baru
            xg = (obj[0] / 32)
            yg = (obj[1] / 32)
            # wg = (obj[2] / 32)
            # hg = (obj[3] / 32)

            # get grid lokasi ke x y
            grid_x = int(np.floor(xg))
            grid_y =int(np.floor(yg))

            # # get kordinat grid ----Cara Lama
            xg = (obj[0] - grid_x * self.size_grid) / (self.size_grid - 1)
            yg = (obj[1] - grid_y * self.size_grid) / (self.size_grid - 1)
            wg = (obj[2] / 32)
            hg = (obj[3] / 32)

            

           # get anchor, iuo
            anchor, iou = best_anchor.get_best_anchor([xg, yg, wg, hg], temp_matrix, grid_x, grid_y)

            # get delta_xywh  (Selisih antara x y w h anotasi - x y w h anchor box yang terpilih
            # x dan y anchorbox 0.5 karena berada di tengah grid
            delta_x = xg - 0.5
            delta_y = yg - 0.5
            delta_w = (wg - ANCHOR_BOX[anchor][0]) / ANCHOR_BOX[anchor][0]
            delta_h = (hg - ANCHOR_BOX[anchor][1]) / ANCHOR_BOX[anchor][1]

            temp_matrix[grid_y, grid_x, anchor, 4] = 1  # set 1 karena ada objek
            temp_matrix[grid_y, grid_x, anchor, 0] =delta_x
            temp_matrix[grid_y, grid_x, anchor, 1] = delta_y
            temp_matrix[grid_y, grid_x, anchor, 2] = delta_w
            temp_matrix[grid_y, grid_x, anchor, 3] = delta_h

            # HELP
            self.all_obj[a]['grid_x'] = grid_x
            self.all_obj[a]['grid_y'] = grid_y
            self.all_obj[a]['anchor'] = anchor
            self.all_obj[a]['xc'] = obj[0]
            self.all_obj[a]['yc'] = obj[1]
            self.all_obj[a]['wc'] = obj[2]
            self.all_obj[a]['hc'] = obj[3]
            self.all_obj[a]['xg'] = xg
            self.all_obj[a]['yg'] = yg
            self.all_obj[a]['wg'] = wg
            self.all_obj[a]['hg'] = hg
            self.all_obj[a]['delta_x'] = delta_x
            self.all_obj[a]['delta_y'] = delta_y
            self.all_obj[a]['delta_w'] = delta_w
            self.all_obj[a]['delta_h'] = delta_h
            a += 1

        return temp_matrix

from keras.utils import Sequence
class SimpleBatchGenerator(Sequence):
    def __init__(self, images, shuffle=False):

        self.images = images
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / BATCH_SIZE))

    def __getitem__(self, idx):

        l_bound = idx * BATCH_SIZE
        r_bound = (idx + 1) * BATCH_SIZE

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - BATCH_SIZE

        instance_count = 0

        ## prepare empty storage space: this will be output
        # Y_BATCH == MATRIX TARGET 13x13x5x5
        # X_BATCH == MATRIX ASLI 416x416x3

        x_batch = np.zeros((r_bound - l_bound, 416, 416, 3))  # input images
        y_batch = np.zeros((r_bound - l_bound, 13, 13, len(ANCHOR_BOX), 4 + 1))  # desired network output

        for train_image_i in self.images[l_bound:r_bound]:
         
            img, all_obj_1_citra, all_ground_truth = praposes_citra().reshape_citra(train_image_i)
            # print(all_obj_1_citra,all_ground_truth)
            # get / buat data target ("grid_x,grid_y,xg,yg,wg,hg,achor dll")
            d = buat_data_target(all_ground_truth, all_obj_1_citra)
            d.mapping_grid()

            for obj in all_obj_1_citra:
                grid_x = int(np.floor(obj['xg']))
                grid_y = int(np.floor(obj['yg']))
                best_anchor = obj['anchor']
                # print(all_obj_1_citra)
                if grid_x < 13 and grid_x < 13:
                    box_coor = [obj['delta_x'], obj['delta_y'], obj['delta_w'], obj['delta_h']]
                    
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1  # nilai confidence objectness ground truth
                    y_batch[instance_count, grid_y, grid_x, best_anchor, :4] = box_coor

            img = img / 255
            x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        # kernel_regularizer=regularizers.l2(0.0005),
                        # kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        # use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)

#MODEL
def TinyYOLO2Model():
  model_in = Input((416, 416, 3))
  model = model_in
  for i in range(0, 5):
        model = conv_batch_lrelu(model, 16 * 2**i, 3)
        model = MaxPooling2D(2, padding='valid')(model)

  model = conv_batch_lrelu(model, 512, 3)
  model = MaxPooling2D(2, 1, padding='same')(model)

  model = conv_batch_lrelu(model, 1024, 3)
  model = conv_batch_lrelu(model, 1024, 3)
        
  n_outputs = len(ANCHOR_BOX) * (5)

  model = Conv2D(n_outputs, (1, 1), padding='same', activation='linear')(model)

  model_out = Reshape([13, 13, 5, 4 + 1])(model)
 
  return Model(inputs=model_in, outputs=model_out)
mmodel=TinyYOLO2Model()
mmodel.summary()


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]



wt_path='/content/drive/My Drive/KULIAH/TA/PROGRAM/dataset/MODEL/yolov2-tiny-voc.weights'
weight_reader = WeightReader(wt_path)
# weight_reader.reset()
nb_conv = 8


# transfer learning antar weight layers 2 arsitektur
for i in range(1, nb_conv + 1):
    conv_layer = mmodel.get_layer('conv2d_' + str(i))

    if i < nb_conv:
        norm_layer = mmodel.get_layer('batch_normalization_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])


# perlu ngfreeze 8 layers  
for layer in mmodel.layers:
    layer.trainable = False
# Add new randomized final 3 layers  
connecting_layer = mmodel.layers[-4].output
top_model = Conv2D(len(ANCHOR_BOX) * (4 + 1), (1, 1), strides=(1, 1), kernel_initializer='he_normal') (connecting_layer)
top_model = Activation('linear') (top_model)
top_model = Reshape((13, 13, len(ANCHOR_BOX), 4 + 1)) (top_model)
new_model = Model(mmodel.input, top_model)
new_model.summary()

#DATA TRAIN

all_data_train=parse_anotasi(folder_citra_train_anno)
train_batch_generator = SimpleBatchGenerator(all_data_train, shuffle=False)
# [train_x_batch,train_b_batch],train_y_batch = train_batch_generator.__getitem__(idx=3)
# print(train_x_batch.shape,train_b_batch.shape,train_y_batch.shape)
# print(train_x_batch[1],"\n",train_b_batch[1],"\n",train_y_batch[1])


#DATA VALIDASI
all_data_valid=parse_anotasi(folder_citra_test_anno)
valid_batch_generator = SimpleBatchGenerator(all_data_valid, shuffle=False)
# [valid_x_batch,valid_b_batch],valid_y_batch = valid_batch_generator.__getitem__(idx=3)

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
checkpoint = ModelCheckpoint('/content/drive/My Drive/KULIAH/TA/PROGRAM/dataset/07_7_CUSTOM_YOLO_bn=64_epoch=100_linear(pretrained).h5',monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
new_model.compile(loss=custom_loss_2, optimizer=Adam(lr=1e-4),metrics=['acc'])
new_model.fit_generator(generator        = train_batch_generator,
                    steps_per_epoch  = len(train_batch_generator),
                    epochs           = 300,
                    verbose          = 1,
                    validation_data  = valid_batch_generator,
                    validation_steps = len(valid_batch_generator),
                    callbacks        = [early_stop, checkpoint],
                    max_queue_size   = 3)
