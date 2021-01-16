# import libraries here
import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = 10,7

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    red_blood_cell_count = 0
    white_blood_cell_count = 0
    has_leukemia = None

    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb, 'gray')
    plt.show()

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    #img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    ret, img_bin = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    img_bitw = cv2.bitwise_not(img_bin)

    img_open = open(img_bitw)
    img_close = close(img_bin)

    img_er = er(img_bin)
    img_dil = dil(img_bin)

    white_blood_cell_count = len(find_white_blood_cells(img_close))
    print('BELIH: '+  str(white_blood_cell_count))
    #plt.imshow(img_close, 'gray')
    #plt.show()

    # CRVENE
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0 * img_rgb[:, :, 0] + 0.1 * img_rgb[:, :, 1] + 0.9 * img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    img_bitw = cv2.bitwise_not(img_bin)
    img_open = open(img_bitw)

    red_blood_cell_count = len(find_red_blood_cells(img_open))
    print('CRVENIH: ' + str(red_blood_cell_count))
    #plt.imshow(img_open, 'gray')
    #plt.show()

    if(white_blood_cell_count * 12 > red_blood_cell_count):
        has_leukemia = True
    else:
        has_leukemia = False


    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure

    return red_blood_cell_count, white_blood_cell_count, has_leukemia

def find_white_blood_cells(img):
    white_blood_cells = []
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    white_cells_temp = []
    for contour in contours:
        area = cv2.contourArea(contour)  # oblast konture
        x, y, width, height = cv2.boundingRect(contour)
        if width > 40 and height > 40:
            white_cells_temp.append(contour)

    if (len(white_cells_temp) <= 3):
        img_er = er(img)
        img, contours, hierarchy = cv2.findContours(img_er, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)  # oblast konture
            x, y, width, height = cv2.boundingRect(contour)
            if width > 120 and height > 120 and area > 18000:
                white_blood_cells.append(contour)
        #plt.imshow(img_er, 'gray')
        #plt.show()
    elif (len(white_cells_temp) >= 10):
        img_wat = watershed(255 - img)
        img_inv = 255 - img_wat
        img, contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            white_blood_cells.append(contour)
        #plt.imshow(img_inv, 'gray')
        #plt.show()
    else:
        white_blood_cells = white_cells_temp

    return white_blood_cells

def find_red_blood_cells(img):
    pom = 0
    red_blood_cells = []
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)  # oblast konture
        if area > 220 and hierarchy[0, pom, 3] == -1:
            red_blood_cells.append(contour)
        pom += 1
    return red_blood_cells

def open(img):
    kernel_ero = np.ones((3, 3))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    img_erode = cv2.erode(img, kernel_dil, iterations=1)

    img_open = cv2.dilate(img_erode, kernel_dil, iterations=1)

    return img_open

def close(img):
    kernel_ero = np.ones((3, 3))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    img_dilate = cv2.dilate(img, kernel_dil, iterations=1)

    img_close = cv2.erode(img_dilate, kernel_dil, iterations=2)

    return img_close

def er(img):
    kernel_ero = np.ones((3, 3))
    img_erode = cv2.erode(img, kernel_ero, iterations=2)

    return img_erode

def dil(img):
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_dil = cv2.dilate(img, kernel_dil, iterations=2)
    return img_dil

def watershed(img):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    '''# Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bin, markers)
    img_bin[markers == -1] = [255, 0, 0]'''

    return sure_fg



