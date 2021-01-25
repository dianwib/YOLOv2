# print(sorted(data.items(),reverse=True))
class NMS(object):
    def __init__(self,data,max_putbbox_from_sort_high_prob=None,merge_nms=True,merge_berulang=True):
        self.data=data
        self.merge_nms=merge_nms
        self.merge_berulang=merge_berulang
        self.limit=max_putbbox_from_sort_high_prob

    def sort_best_conf(self):
        self.data_sort=sorted(self.data.items(),reverse=True)
        return self.data_sort
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



    def calc(self):
        self.data_sort=self.sort_best_conf()
        list_bbox=[]

        for data in self.data_sort:
           list_bbox.append(data[1])

        if self.limit==None:
            return list_bbox

        else:
            return list_bbox[:self.limit]




    def pred(self,threshold=0.5):
        list_bbox=self.calc()

        temp={}
        for i in range(len(list_bbox)):
            temp[i]=1 #set 1 semua jika tampil,0 jika tdk tampil

        for i in range (len(list_bbox)):
            # temp[i] = 0
            box_a=list_bbox[i]
            for j in range(i+1,len(list_bbox)):


                box_b = list_bbox[j]

                if i==j or temp[j]==0:
                    continue
                else:
                    iou=self.IOU(box_a,box_b)

                    if iou > threshold:

                        temp[j] = 0 #bbox tdk ditampilkan / dihapus

        temp_box=[] #cek dan ambil yg nilai 1
        for i in temp.keys():
            if temp[i]==1:
              temp_box.append(list_bbox[i])

        # print(temp_box)
        #update hasil merge

        if self.merge_nms==True:
            temp_box=self.merge(temp_box)


        temp_all_box = []

        for i in range(len(temp_box)):
            temp_dic = {}
            temp_dic['xmin'] = temp_box[i][0]
            temp_dic['ymin'] = temp_box[i][1]
            temp_dic['xmax'] = temp_box[i][2]
            temp_dic['ymax'] = temp_box[i][3]
            temp_dic['obj'] = self.data_sort[i][0]
            temp_all_box.append(temp_dic)

        return temp_all_box
        #
        # fig, ax = plt.subplots(1)
        # for i in range(len(temp)):
        #     if temp[i]==1:
        #
        #         rec = patches.Rectangle((list_bbox[i][0], list_bbox[i][1]), list_bbox[i][2] - list_bbox[i][0], list_bbox[i][3] - list_bbox[i][1],
        #                         linewidth=2, edgecolor='green', facecolor='none')
        #         ax.add_patch(rec)

        # return temp


    def merge(self,  temp_box,theshold_merge=.0):
        ULANGI=True
        def hapus_duplikat(list_a):

            list = list_a.copy()
            for i in list:
                count = 0
                for j in list_a:
                    if j == i:

                        count += 1
                        if count > 1:
                            list_a.remove(j)

            return list_a
        # print((temp_box),"Aaa")


        temp_index={}
        for index_i in range(len(temp_box)):
            temp_for_every_index=[]
            for index_j in range(len(temp_box)):
                if index_j != index_i:
                    iou=self.IOU(temp_box[index_i],temp_box[index_j])
                    if iou > theshold_merge:
                        temp_for_every_index.append(index_j)

            temp_index[index_i]=temp_for_every_index


        # print(temp_box)
        # print(temp_index)#hasil {0: [5], 1: [2, 9], 2: [1, 9], 3: [5] dll


        #cek node data,ex b1=b2,b4 dll
        # hasil {0: [], 1: [5], 2: [3], 3: [2, 7], 4: [5], 5: [1, 4, 6], 6: [5], 7: [3]}


        new_bbox=[]
        for index_i in temp_index.keys():
           if temp_index[index_i] == []:


               # nyalain kalo box yang gada temennya di keep / tetap di tampilin
               new_bbox.append([index_i])
               # continue
           else:
               a=[index_i]
               for index_j in temp_index[index_i]:
                  a.append(index_j)

               new_bbox.append(sorted(a))



        #hapus duplikat data
        # print(new_bbox,"s")
        new_bbox = hapus_duplikat(new_bbox)
        a_new_bbox=new_bbox

        # gabung data cek subset to superset
        # hasil [[0], [3, 2, 7], [5, 1, 4, 6]]
        # print(new_bbox)
        for data in (new_bbox):

            for data_pembanding in (new_bbox):
                if data != data_pembanding:
                    # print(data,data_pembanding,"Aa")

                    if set(data_pembanding).issubset(data) == True  :
                        if data_pembanding in a_new_bbox:
                            a_new_bbox.remove(data_pembanding)

                    elif set(data_pembanding).issuperset(data) == True:
                        if data in a_new_bbox:
                            a_new_bbox.remove(data)

            # hapus duplikat data
        # a_new_bbox = hapus(a_new_bbox)
        # print(a_new_bbox)

        if len(a_new_bbox)==len(temp_box):
            ULANGI=False

        def merge_bbox(A,B):
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

            Box_Intersect_xmin = min(A_xmin, B_xmin)
            Box_Intersect_ymin = min(A_ymin, B_ymin)
            Box_Intersect_xmax = max(A_xmax, B_xmax)
            Box_Intersect_ymax = max(A_ymax, B_ymax)

            return [Box_Intersect_xmin,Box_Intersect_ymin,Box_Intersect_xmax,Box_Intersect_ymax]

        #merge bbox jika ada
        a=[]
        for bbox_i in a_new_bbox:
            if len(bbox_i) > 1:

                temp=temp_box[bbox_i[0]]
                for j in range (1,len(bbox_i)):
                    temp=merge_bbox(temp,temp_box[bbox_i[j]])

                a.append(temp)

            else:
                a.append(temp_box[bbox_i[0]])


        #return from merge berupa[xmin,ymin,xmax,ymax]
        # print(a,"A")

        if ULANGI==True and self.merge_berulang==True:
            return self.merge(a)
        else:

            return a


    def nms(self, threshold=.2):
        detections= self.calc()
        detections = sorted(detections, key=lambda detections: detections[2],
                            reverse=True)

        new_detections = []

        new_detections.append(detections[0])

        del detections[0]

        for index, detection in enumerate(detections):
            for new_detection in new_detections:
                if self.IOU(detection, new_detection) > threshold:
                    del detections[index]
                    break
            else:
                new_detections.append(detection)
                del detections[index]

   
        temp_box=new_detections
        temp_all_box = []
        for i in range(len(temp_box)):
            temp_dic = {}
            temp_dic['xmin'] = temp_box[i][0]
            temp_dic['ymin'] = temp_box[i][1]
            temp_dic['xmax'] = temp_box[i][2]
            temp_dic['ymax'] = temp_box[i][3]
            temp_dic['obj'] = self.data_sort[i][0]
            temp_all_box.append(temp_dic)

        return temp_all_box
        # return new_detections

    
