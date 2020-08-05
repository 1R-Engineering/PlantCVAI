"""

    Program pendeteksi tumbuhan
    Los Dol Eng

"""

import os
import csv
import cv2
import numpy as np
from plantcv import plantcv as pcv 

def main():
    #Import file gambar
    path = 'Image test\capture (2).jpg'
    imgraw, path, img_filename = pcv.readimage(path, mode='native')

    nilaiTerang = np.average(imgraw)

    if nilaiTerang < 50:
        pcv.fatal_error("Night Image")
    else:
        pass

    rotateimg = pcv.rotate(imgraw, -3, True)
    imgraw = rotateimg

    bersih1=pcv.white_balance(imgraw)

    hitamputih=pcv.rgb2gray_lab(bersih1, channel='a')

    img_binary = pcv.threshold.binary(hitamputih, threshold=110, max_value=255, object_type='dark')

    fill_image = pcv.fill(bin_img=img_binary, size=10)
    dilated = pcv.dilate(gray_img=fill_image, ksize=6, i=1)
    id_objects, obj_hierarchy = pcv.find_objects(img=imgraw, mask=dilated)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=imgraw, x=280, y=96, h=1104, w=1246) 
    print(type(roi_contour))
    print(type(roi_hierarchy))
    print(roi_hierarchy)
    print(roi_contour)
    roicontour = cv2.drawContours(imgraw, roi_contour, -1, (0,0,255), 3)
    #cv2.rectangle(imgraw, roi_contour[0], roi_contour[3])
    

    roi_obj, hier, kept_mask, obj_area = pcv.roi_objects(img=imgraw, roi_contour=roi_contour, 
                                                                          roi_hierarchy=roi_hierarchy,
                                                                          object_contour=id_objects,
                                                                          obj_hierarchy=obj_hierarchy, 
                                                                          roi_type='partial')
    cnt_i, contours, hierarchies = pcv.cluster_contours(img=imgraw, roi_objects=roi_obj, 
                                                             roi_obj_hierarchy=hier, 
                                                             nrow=4, ncol=3)
    clustered_image = pcv.visualize.clustered_contours(img=imgraw, grouped_contour_indices=cnt_i, 
                                                   roi_objects=roi_obj,
                                                   roi_obj_hierarchy=hier) 
    obj, mask = pcv.object_composition(imgraw, roi_obj, hier)
    hasil = pcv.analyze_object(imgraw, obj, mask)                           
    pcv.print_image(imgraw, 'Image test\Result\wel.jpg')   
    pcv.print_image(clustered_image, 'Image test\Result\clustred.jpg')
    pcv.print_image(hitamputih, 'Image test\Result\Bersihe.jpg')
    pcv.print_image(dilated, 'Image test\Result\dilated.jpg')
    pcv.print_image(hasil, 'Image test\Result\hasil.jpg')
    plantHasil = pcv.outputs.observations['area']
    data1 = pcv.outputs.observations['area']['value']
    print(data1)
    print(plantHasil)
if __name__ == '__main__':
    main()