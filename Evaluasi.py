import numpy as np
class Evaluasi(object):
    def __init__(self,prediksi,GT,treshold_iou=0.3):
        self.data_prediksi=self.parse_prediksi(prediksi)
        self.data_GT = self.parse_GT(GT)
        self.treshold_iou=treshold_iou
        self.finder()
        # print(self.data_GT)
        # print(self.data_prediksi)
        # print(self.finder())

    def IOU(self,A, B):

        # index list_ground_truth [xg,yg,wg,hg]

        A_xmin = A[0]
        A_xmax = A[2]
        A_ymin = A[1]
        A_ymax = A[3]

        B_xmin = B[0]
        B_xmax = B[2]
        B_ymin = B[1]
        B_ymax = B[3]
        # print(A_xmin,A_ymin,A_xmax,A_ymax)
        # print(B_xmin,B_ymin,B_xmax,B_ymax)

        Box_Intersect_xmin = max(A_xmin, B_xmin)
        Box_Intersect_ymin = max(A_ymin, B_ymin)
        Box_Intersect_xmax = min(A_xmax, B_xmax)
        Box_Intersect_ymax = min(A_ymax, B_ymax)
        # print(Box_Intersect_xmin,Box_Intersect_ymin,Box_Intersect_xmax,Box_Intersect_ymax)

        if Box_Intersect_xmax < Box_Intersect_xmin or Box_Intersect_ymax < Box_Intersect_ymin :
            return 0

        else:

            intersection_area = (max(Box_Intersect_xmin, Box_Intersect_xmax) - min(Box_Intersect_xmin,
                                                                               Box_Intersect_xmax)) * (
                                    max(Box_Intersect_ymin, Box_Intersect_ymax) - min(Box_Intersect_ymin,
                                                                                      Box_Intersect_ymax))
            W_A = A_xmax - A_xmin
            H_A = A_ymax - A_ymin
            W_B = B_xmax - B_xmin
            H_B = B_ymax - B_ymin

            object_area = W_A * H_A
            anchor_area = W_B * H_B
            union = object_area + anchor_area
            iou = intersection_area / (union - intersection_area)

            return iou

    def parse_prediksi(self,data_prediksi):
        temp_prediksi=[]
        for data in data_prediksi:
            if [data['xmin'],data['ymin'],data['xmax'],data['ymax']] not in temp_prediksi:

                temp_prediksi.append([data['xmin'],data['ymin'],data['xmax'],data['ymax']])
        return temp_prediksi

    def parse_GT(self, data_GT):
        temp_GT = []
        for data in data_GT['object']:
            temp_GT.append([data['xmin'], data['ymin'], data['xmax'], data['ymax']])
        return temp_GT

    def finder(self):
        prediksi=self.data_prediksi
        GT=self.data_GT

        temp_data=np.zeros((len(GT),len(prediksi)))
        # print(temp_data.shape,len(GT),len(prediksi))

        #calc nilai iou dr GT ke all prediksi
        for GT_i in range(len(GT)):
            for pred_i in range(len(prediksi)):
                iou=self.IOU(prediksi[pred_i],GT[GT_i])

                # print(pred_i,GT_i,prediksi[pred_i],GT[GT_i],iou)
                if iou > self.treshold_iou:
                    temp_data[GT_i,pred_i]=iou



        TP=0
        FP=0
        # print(temp_data)
        list_pred_TP=np.where(temp_data.any(axis=0))[0]
        #find True Postif (len data column np not contain all 0)
        TP=len(list_pred_TP)

        # matiin jika len TP > lebih dari len GT ex, TP=3 pdhl GT=1 krn akan menambah nilai f1-score and precision
        if TP > len(GT):
            TP=len(GT)

        #find FP (len prediksi - TP)
        FP=len(prediksi)-TP

        FN=0
        #FN (==0 if len(GT) < len (prediksi)
        #else
        if len(GT) > TP:
            FN = len(GT)-TP

        # print(TP,FP,FN)
        data_eval={'TP':TP,'FP':FP,'FN':FN}


        # print(data_eval)
        self.data_iou=temp_data
        return data_eval


    def get_precision(self):
        data=self.finder()#return value {'TP': 0, 'FP': 1, 'FN': 3}
        TP=data['TP']
        FP = data['FP']
        FN = data['FN']


        if TP ==0:
            precision= 0
        else:
            precision=TP/(TP+FP)
        return precision

    def get_recall(self):
        data=self.finder()#return value {'TP': 0, 'FP': 1, 'FN': 3}
        TP=data['TP']
        FP = data['FP']
        FN = data['FN']


        if TP ==0:
            recall= 0.0
        else:
            recall=TP/(TP+FN)
        return recall

    def get_accuracy(self):
        data=self.finder()#return value {'TP': 0, 'FP': 1, 'FN': 3}
        TP=data['TP']
        FP = data['FP']
        FN = data['FN']

        if TP ==0:
            accuracy= 0.0
        else:
            accuracy=TP/(TP+FP+FN)
        return accuracy

    def get_f1score(self):
        data = self.finder()  # return value {'TP': 0, 'FP': 1, 'FN': 3}

        if data['TP']  ==0:

            f1score= 0.0
        else:
            f1score = (2 * self.get_recall() * self.get_precision()) / (self.get_recall() + self.get_precision())
        return f1score
