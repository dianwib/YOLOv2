label = ['person']
folder_citra_train = "../INRIAPerson (use)/Images/pos"
folder_citra_anno = "../INRIAPerson (use)/Annotations"
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2


def parse_anotasi(annotasi_dir=folder_citra_anno):
    all_img = []
    for file_annotasi in sorted(os.listdir(annotasi_dir)):
        temp = open(annotasi_dir+ '/' + file_annotasi, "r",encoding="ISO-8859-1")
        label_percitra = {}
        temp_labelobj = []

        for line in temp:
            if 'Image filename' in line:
                a = (line.index('"'))
                temp_lokasi_citra = (line[a + 1:-2])
                label_percitra['filename'] = temp_lokasi_citra

            if 'Image size' in line:
                w = line[25:29]
                h = line[31:35]
                label_percitra['height'] = h
                label_percitra['width'] = w

            if 'Center point on object' in line:
                if 'person' in line:
                    temp_label = 'person'
                else:
                    temp_label = 'no person'


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
                temp_dic['name'] = temp_label
                temp_dic['xmin'] = xmin
                temp_dic['ymin'] = ymin
                temp_dic['xmax'] = xmax
                temp_dic['ymax'] = ymax
                temp_labelobj.append(temp_dic)

            label_percitra['object'] = temp_labelobj

        all_img.append(label_percitra)

    return all_img


def liat_banyak_label(data):
    temp_banyak_label={}
    for temp_data in data:
        for i in temp_data['object']:
            objek = (i['name'])
            if objek not in temp_banyak_label:
                temp_banyak_label[objek] = 1
            else:
                temp_banyak_label[objek] += 1
    return temp_banyak_label


data_train=parse_anotasi()
total_label=liat_banyak_label(data_train)

print((total_label),"len")
def liat_train_plot(total_label,data_train):
    y_pos = np.arange(len(total_label))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(y_pos, list(total_label.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(total_label.keys()))
    ax.set_title("total data : " + str(len(data_train)) + ", total objek : " + str(sum(list(total_label.values()))))
    plt.show()

def normalisasi_anotasi(data_train):
    wh=[]
    for annotasi in data_train:
        ori_w=annotasi['width']
        ori_h = annotasi['height']
        #print(ori_w,ori_h)
        for objek in annotasi['object']:
            #NOMALISASI W H buat clustering anchor (jumlah n bentuk) buat grid
            norm_w=(float(objek['xmax']) - float(objek ['xmin'])) / float(ori_w)
            norm_h = (float(objek['ymax']) -float( objek['ymin'])) / float(ori_h)
            temp=[norm_w,norm_h]
            wh.append(temp)
    wh=np.array(wh)
    # plt.figure(figsize=(10, 10))
    # plt.scatter(wh[:, 0], wh[:, 1], alpha=0.1)
    # plt.title("Clusters", fontsize=20)
    # plt.xlabel("normalized width", fontsize=20)
    # plt.ylabel("normalized height", fontsize=20)
    # plt.show()
    #print(np.mean(wh,axis=0))
    return wh
data_normalisasi=(normalisasi_anotasi(data_train))
#print(data_train)
#liat_train_plot(total_label,data_train)


def ploting_kmean(data):
    list_color =['red','green','blue','black','orange','brown','orange']
    for i in data:
        plt.figure(figsize=(10, 10))
        plt.title("Banyak Kluster : " + str(i['kluster']) + " Mean IOU : " + str(i['mean IOU']), fontsize=20)
        plt.xlabel("width", fontsize=20)
        plt.ylabel("height", fontsize=20)
        a = []

        for kluster in range(len(i['data k'])):
            temp_data = np.array(i['data k'][kluster])
            # print (temp_data)
            plt.scatter(temp_data[:, 0], temp_data[:, 1], alpha=0.1, color=list_color[kluster%len(list_color)])
            # plt.fill_between(temp_data[:, 0], temp_data[:, 1],color=list_color[kluster])
            a.append(len(temp_data))
            plt.gca().legend(a)

    plt.show()


def silhoutte(data):

    temp_s=[]
    for i in range(len(data)):

        for data_i in data[i]:
            ai=[]
            bi=[]

            for kluster in range(len(data)):
                temp_bi=[]
                for data_pembanding in data[kluster]:

                    if data_i == data_pembanding:
                        continue
                    else:
                        #print(data_i,data_pembanding)
                        if data_pembanding in data[kluster] and data_i in data[kluster]:
                            ai.append(1 - IOU(data_normalisasi[data_i],data_normalisasi[data_pembanding]))
                        else:
                            temp_bi.append(1- IOU(data_normalisasi[data_i],data_normalisasi[data_pembanding]))

                if data_i not in data[kluster]:
                    hasil_temp_bi = np.average(temp_bi)
                    bi.append(hasil_temp_bi)
                # print(data_i)
                # print(bi,"kluster")
                # print(kluster,hasil_temp_bi)

            hasil_ai=np.average(ai)
            hasil_bi = min(bi)
            # print(len(ai),len(bi))
            # print(data_i,hasil_ai,hasil_bi)

            if hasil_ai > hasil_bi:
                si=hasil_bi/hasil_ai -1
                temp_s.append(si)
            elif hasil_ai < hasil_bi:
                si=1- hasil_ai/hasil_bi
                temp_s.append(si)
            elif hasil_ai == hasil_bi:
                si=0
                temp_s.append(si)


    mean_si=np.average(temp_s)
    print(mean_si)

def liat_anchor(data):
    for i in data:
        plt.figure(figsize=(10, 10))
        plt.title("Banyak Anchor : " + str(len(i['centroid'])), fontsize=20)

        ax=plt.gca()
        for anchor in range(len(i['centroid'])):
            temp_data = np.array(i['centroid'][anchor])
            w_anchor=temp_data[0]
            h_anchor=temp_data[1]
            x_min=1/2 - w_anchor/2
            y_min = 1 / 2 - h_anchor / 2

            rec=patches.Rectangle((x_min,y_min),w_anchor,h_anchor,linewidth=1,edgecolor='r',facecolor='none')

            ax.add_patch(rec)
    plt.show()
def IOU(box,centroid_k):
    intersection=(min(box[0],centroid_k[0])) * min(box[1],centroid_k[1])
    box_area= box[0] * box [1]
    centroid_k_area=centroid_k[0] * centroid_k[1]
    union=box_area+centroid_k_area
    iou=intersection / (union-intersection)

    return iou

def kmean(banyak_k,min_k,data_yang_normalisasi):
    import random
    hasil=[]

    def update_mean_kluster(temp_kluster,temp_data_k,temp_mean_k):
        for i in range(temp_kluster):
            if len(temp_data_k[i])!=0:
                temp_k=np.array(temp_data_k[i])
                #UPDATE mean /  centroid tiap kluster
                temp_mean_k[i]=np.mean(temp_k,axis=0)


    for temp_kluster in range(min_k,banyak_k):
        temp_hasil={}
        temp_mean_k = []
        temp_data_k = []
        # print(data_yang_normalisasi)
        langkah_random = True
        iterasi = 0
        temp_IOU = []

        dict_kluster = []
        dict_kluster_sebelum = []

        while True:
            # Random step
            if langkah_random:
                for kluster in range(temp_kluster):
                    temp = [random.uniform(0.1, 0.2), random.uniform(0.4, 0.5)]
                    temp_mean_k.append(temp)

                    temp_data_k.append([])
                    dict_kluster.append([])

                temp_mean_k = np.array(temp_mean_k)
                #print(temp_mean_k)
                langkah_random = False

            else:
                iterasi += 1

                for data_i in range(len(data_yang_normalisasi)):
                    box = np.array(data_yang_normalisasi[data_i])

                    temp_all_k = []#buat nampung sementara nilai iou data i keseluruh k
                    # itung nilai data ke tiap centroid cluster
                    for rata_k in temp_mean_k:
                        # dikurangi 1 buat cek mininimum jarak ke cluster
                        temp_all_k.append(1 - IOU(box, rata_k))

                    # cek nilai terkecil
                    temp_min = min(temp_all_k)
                    #cari index ke berapa data yng kecil
                    temp_min_index = temp_all_k.index(temp_min)
                    # add bbbox ke datakluster
                    temp_data_k[temp_min_index].append(box)
                    dict_kluster[temp_min_index].append(data_i)
                    # add IOU berdasr cluster terbaik  ke temp
                    temp_IOU.append(1-temp_min)#dimunuskan kembali agar nilainya lbh besar

                    #print(1-temp_min,"best IOU",temp_min,"min",IOU(box, rata_k),"ambg",temp_all_k)


                temp_k_IOU = np.array(temp_IOU)
                mean_IOU = np.average(temp_k_IOU)


                if (dict_kluster == dict_kluster_sebelum):
                    print("________")
                    # print(temp_k_IOU)
                    print("kluster : " ,temp_kluster)
                    print("iterasi",iterasi)
                    print("rata IOU : ", mean_IOU)
                    print("centroid : ", temp_mean_k)
                    temp_hasil['kluster']=temp_kluster
                    temp_hasil['iterasi']= iterasi
                    temp_hasil['mean IOU']= mean_IOU
                    temp_hasil['centroid']= temp_mean_k
                    temp_hasil['data k'] = temp_data_k

                    hasil.append(temp_hasil)
                    #silhoutte(dict_kluster,temp_data_k)

                    #ploting_kmean(temp_data_k,temp_mean_k,mean_IOU,dict_kluster,temp_kluster)
                    break
                else:

                    dict_kluster_sebelum = dict_kluster
                    update_mean_kluster(temp_kluster,temp_data_k, temp_mean_k)

                    # RESET DATA K
                    dict_kluster = []
                    temp_data_k = []
                    for kluster in range(temp_kluster):
                        temp_data_k.append([])
                        dict_kluster.append([])

                    if iterasi == 100:
                        break

    ploting_kmean(hasil)
    liat_anchor(hasil)
(kmean(7,2,data_normalisasi))

