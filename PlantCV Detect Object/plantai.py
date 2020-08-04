""" Deteksi tumbuhan pada planting space hidroponik
    Digunakan opencv bersama dengan plantcv untuk mendeteksi gambar
    Workflow dari pendeteksi tumbuhan sebagai berikut:
        1. Ambil gambar dari IP Cam melalui fungsi di opencv
        2. Proses gambar dengan pcv
        3. PROFIT???
    
    Diprogram oleh M. Hilal Yahya Hidayat
    LosDolEng
"""

#Memasukan package-package penting
import numpy as np
import argparse 
from plantcv import plantcv as pcv

#Menangkap gambar IP Cam dengan opencv
    ##Ngosek, koding e wis ono tapi carane ngonek ning gcloud rung reti wkwk

#Mengambil gambar yang sudah didapatkan dari opencv untuk diproses di plantcv
gambarTumbuhanRaw = pcv.readImage('Image test\capture (1).jpg')

#Mengatur white balance dari gambar
#usahakan gambar rata (tanpa bayangan dari manapun!)!
#GANTI nilai dari region of intrest (roi) berdasarkan ukuran gambar!!
koreksiWhiteBal = pcv.white_balance(gambarTumbuhanRaw, roi=(400,800,200,200)) 

#mengubah kontras gambar agar berbeda dengan warna background
#tips: latar jangan sama hijaunya
kontrasBG = pcv.rgb2gray_lab(koreksiWhiteBal, channel='a')

#binary threshol gambar
#sesuaikan thresholdnya
binthres = pcv.threshold.binary(gray_img=kontrasBG, threshold=120, max_value=255, object_type='dark')

#hilangkan noise dengan fill noise
resiksitik = pcv.fill(binthres, size=10)

#haluskan dengan dilate
dilasi = pcv.dilate(resiksitik, ksize=2, i=1)

#ambil objek dan set besar roi
id_objek, hirarki_objek = pcv.find_objects(gambarTumbuhanRaw, mask=dilasi)
roi_contour, roi_hierarchy = pcv.roi.rectangle(img=img1, x=6, y=90, h=200, w=390)

objek_roi, hirarki_objek_roi, kept_mask, luas_objek = pcv.roi_objects  (img=gambarTumbuhanRaw, 
                                                                        roi_contour=roi_contour, 
                                                                        roi_hierarchy=roi_hierarchy,
                                                                        object_contour=id_objek,
                                                                        obj_hierarchy=hirarki_objek, 
                                                                        roi_type='partial')

#ubah nrow dan ncol sesuai dengan planting space
klaster_i, kontours, hirarkis = pcv.cluster_contours   (img=gambarTumbuhanRaw,
                                                        roi_objects=objek_roi, 
                                                        roi_obj_hierarchy=hirarki_objek_roi, 
                                                        nrow=4, ncol=6)

#pisah gambar per tumbuhan
output_path, imgs, masks = pcv.cluster_contour_splitimg(img=gambarTumbuhanRaw, grouped_contour_indexes=klaster_i, 
                                                            contours=kontours, hierarchy=hirarki_objek, 
                                                            outdir=out, file=filename, filenames=names)
