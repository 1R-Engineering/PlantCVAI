""" 
    Deteksi tumbuhan pada planting space hidroponik
    Digunakan opencv bersama dengan plantcv untuk mendeteksi gambar
    Workflow dari pendeteksi tumbuhan sebagai berikut:
        1. Ambil gambar dari IP Cam melalui fungsi di opencv
        2. Proses gambar dengan pcv
        3. PROFIT???
    
    Algoritma pendeteksi:
        InputGambar->KoreksiKecerahan->AmbilWarnaHijau->AturROI->GabunMaskdenganRaw->IdentifikasiObj->Klustring
    Diprogram oleh M. Hilal Yahya Hidayat
    LosDolEng
"""

#Memasukan package-package penting
import numpy as np
import argparse 
import cv2
from plantcv import plantcv as pcv

def main():
    #Menangkap gambar IP Cam dengan opencv
        ##Ngosek, koding e wis ono tapi carane ngonek ning gcloud rung reti wkwk

    #Mengambil gambar yang sudah didapatkan dari opencv untuk diproses di plantcv
    path = 'Image test\capture (1).jpg'
    gmbTumbuhanRaw, path, filename = pcv.readimage(path, mode='native')

    #benarkan gambar yang miring
    koreksiRot = pcv.rotate(gmbTumbuhanRaw, 2, True)
    gmbKoreksi = koreksiRot
    pcv.print_image(gmbKoreksi, 'Image test\Hasil\gambar_koreksi.jpg')

    #Mengatur white balance dari gambar
    #usahakan gambar rata (tanpa bayangan dari manapun!)!
    #GANTI nilai dari region of intrest (roi) berdasarkan ukuran gambar!!
    koreksiWhiteBal = pcv.white_balance(gmbTumbuhanRaw, roi=(2,100,1104,1200)) 
    pcv.print_image(koreksiWhiteBal, 'Image test\Hasil\koreksi_white_bal.jpg')

    #mengubah kontras gambar agar berbeda dengan warna background
    #tips: latar jangan sama hijaunya
    kontrasBG = pcv.rgb2gray_lab(koreksiWhiteBal, channel='a')
    pcv.print_image(kontrasBG, 'Image test\Hasil\koreksi_kontras.jpg')

    #binary threshol gambar
    #sesuaikan thresholdnya
    binthres = pcv.threshold.binary(gray_img=kontrasBG, threshold=115, max_value=255, object_type='dark')

    #hilangkan noise dengan fill noise
    resiksitik = pcv.fill(binthres, size=10)
    pcv.print_image(resiksitik, 'Image test\Hasil\\noiseFill.jpg')

    #haluskan dengan dilate
    dilasi = pcv.dilate(resiksitik, ksize=12, i=1)

    #ambil objek dan set besar roi
    id_objek, hirarki_objek = pcv.find_objects(gmbTumbuhanRaw, mask=dilasi)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=gmbKoreksi, x=20, y=96, h=1100, w=680)

    #keluarkan gambar (untuk debug aja sih)
    roicontour = cv2.drawContours(gmbKoreksi, roi_contour, -1, (0,0,255), 3)
    pcv.print_image(roicontour, 'Image test\Hasil\\roicontour.jpg')
    """
    #cv2.imshow("debugROI", gmbKoreksi)
    objek_roi, hirarki_objek_roi, kept_mask, luas_objek = pcv.roi_objects  (img=gmbTumbuhanRaw, 
                                                                            roi_contour=roi_contour, 
                                                                            roi_hierarchy=roi_hierarchy,
                                                                            object_contour=id_objek,
                                                                            obj_hierarchy=hirarki_objek, 
                                                                            roi_type='partial')

    #ubah nrow dan ncol sesuai dengan planting space
    klaster_i, kontours, hirarkis = pcv.cluster_contours   (img=gmbTumbuhanRaw,
                                                            roi_objects=objek_roi, 
                                                            roi_obj_hierarchy=hirarki_objek_roi, 
                                                            nrow=4, ncol=6)

    #cv2.waitKey(0)
    """
if __name__ == "__main__":
    main()