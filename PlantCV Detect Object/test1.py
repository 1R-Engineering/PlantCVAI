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
    path = 'E:\Project\KEMINFO20-IOT-HydroponicsAutomation\OpenCVCAM\openCV\Testimg.png'
    imgraw, path, img_filename = pcv.readimage(path, mode='native')

    nilaiTerang = np.average(imgraw)

    if nilaiTerang < 50:
        pcv.fatal_error("Night Image")
    else:
        pass

    bersih1=pcv.white_balance(imgraw)

    hitamputih=pcv.rgb2gray_lab(bersih1, channel='a')

    img_binary = pcv.threshold.binary(hitamputih, threshold=110, max_value=255, object_type='dark')

    fill_image = pcv.fill(bin_img=img_binary, size=10)
    dilated = pcv.dilate(gray_img=fill_image, ksize=6, i=1)
    id_objects, obj_hierarchy = pcv.find_objects(img=imgraw, mask=dilated)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=imgraw, x=6, y=10, h=785, w=522) 
    roi_obj, hier, kept_mask, obj_area = pcv.roi_objects(img=imgraw, roi_contour=roi_contour, 
                                                                          roi_hierarchy=roi_hierarchy,
                                                                          object_contour=id_objects,
                                                                          obj_hierarchy=obj_hierarchy, 
                                                                          roi_type='partial')
    cnt_i, contours, hierarchies = pcv.cluster_contours(img=imgraw, roi_objects=roi_obj, 
                                                             roi_obj_hierarchy=hier, 
                                                             nrow=3, ncol=2)
    clustered_image = pcv.visualize.clustered_contours(img=imgraw, grouped_contour_indices=cnt_i, 
                                                   roi_objects=roi_obj,
                                                   roi_obj_hierarchy=hier) 
    obj, mask = pcv.object_composition(imgraw, roi_obj, hier)
    hasil = pcv.analyze_object(imgraw, obj, mask)                              
    pcv.print_image(clustered_image, 'E:\Project\KEMINFO20-IOT-HydroponicsAutomation\OpenCVCAM\openCV\clustred.jpg')
    pcv.print_image(hitamputih, 'E:\Project\KEMINFO20-IOT-HydroponicsAutomation\OpenCVCAM\openCV\Bersihe.jpg')
    pcv.print_image(dilated, 'E:\Project\KEMINFO20-IOT-HydroponicsAutomation\OpenCVCAM\openCV\dilated.jpg')
    pcv.print_image(hasil, 'E:\Project\KEMINFO20-IOT-HydroponicsAutomation\OpenCVCAM\openCV\hasil.jpg')
    plantHasil = pcv.outputs.observations['area']
    data1 = pcv.outputs.observations['area']['value']
    print(data1)
    print(plantHasil)

if __name__ == '__main__':
    main()