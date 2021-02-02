# import libraries here
import cv2
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import collections
from fuzzywuzzy import fuzz

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

matplotlib.rcParams['figure.figsize'] = 14,8

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans

def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    # Slika sa velikim slovima za treniranje
    image_color_big_letters = load_image(train_image_paths[0])
    image_bin = invert_image(image_bin_threshold_otsu(image_gray(image_color_big_letters)))  # Dodaj invert_image ako je potrevno
    image_bin = dilate(erode(image_bin))
    image_with_conturs, big_characters = select_roi_big_letters(image_color_big_letters.copy(), image_bin)
    display_image(image_with_conturs, False)

    # Slika sa malim slovima za treniranje
    image_color_smal_letters = load_image(train_image_paths[1])
    image_bin = invert_image(image_bin_threshold_otsu(image_gray(image_color_smal_letters)))  # Dodaj invert_image ako je potrevno
    image_bin = dilate(erode(image_bin))
    image_with_conturs, small_characters = select_roi_small_letters(image_color_smal_letters.copy(), image_bin)
    display_image(image_with_conturs, False)

    alphabet = ['A','B','C','Č','Ć','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','Š','T','U','V','W','X','Y','Z','Ž',
                'a','b','c','č','ć','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','š','t','u','v','w','x','y','z','ž']

    characters = big_characters
    for small_character in small_characters:
        characters.append(small_character)

    inputs = prepare_for_ann(characters)
    outputs = convert_output(alphabet)

    print('Ukupan broj kontura: ' + str(len(characters)))
    print('Ukupan broj prepoznatih kontura: ' + str(len(outputs)))

    '''for c in characters: # Ispis slicica kontura
        display_image(c, False)'''

    ann = load_trained_ann() # Probaj da ucitas prethodno istreniran model

    if ann == None:
        print("Treniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        serialize_ann(ann) # Serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put

    # Provera istreniranosti modela
    result = ann.predict(np.array(inputs, np.float32))
    #print(display_result(result, alphabet))

    model = ann
    return model

def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
    image_color_sentece = load_image(image_path)
    ret, image_otsu = image_bin_threshold_otsu2(image_gray(image_color_sentece))
    if(ret < 200): # Recenice koje za pozadinu imaju sliku
        avg_colors = image_color_sentece.mean(0).mean(0)
        r = avg_colors[0]
        g = avg_colors[1]
        b = avg_colors[2]
        #print('R - ' + str(r) + ' G - ' + str(g) + ' B - ' + str(b))
        if(b > 190): # Slike koje imaju plavu pozadinu (vodu)
            hsv = cv2.cvtColor(image_color_sentece, cv2.COLOR_RGB2HSV)
            #ret, thresh = cv2.threshold(image_gray(image_color_sentece), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret, thresh = cv2.threshold(hsv[:, :, 0], 55, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            image_bin = invert_image(thresh)

            image_bin = invert_image(image_bin)
            image_bin = dilate(erode(image_bin))
            # display_image(image_bin, False)
            image_with_conturs, characters, region_distances = select_roi_with_distances_blue(image_color_sentece.copy(),image_bin)
            # display_image(image_with_conturs, False)
        else: # Sve sa pozadinom koja nije voda
            image_with_color = image_bin_threshold_otsu(image_gray_params(image_color_sentece))
            image_bin = image_with_color

            image_bin = invert_image(image_bin)
            image_bin = dilate(erode(image_bin))
            # display_image(image_bin, False)
            image_with_conturs, characters, region_distances = select_roi_with_distances(image_color_sentece.copy(),image_bin)
            # display_image(image_with_conturs, False)
    else:
        image_bin = image_otsu
        #display_image(image_bin, False)

        image_bin = invert_image(image_bin)
        image_bin = dilate(erode(image_bin))
        # display_image(image_bin, False)
        image_with_conturs, characters, region_distances = select_roi_with_distances(image_color_sentece.copy(),image_bin)
        # display_image(image_with_conturs, False)

    print('Broj prepoznatih regiona:', len(characters))
    print('Broj distanci:', len(region_distances))
    display_image(image_with_conturs, False)

    try:
        # Podešavanje centara grupa K-means algoritmom
        distances = np.array(region_distances).reshape(len(region_distances), 1)

        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)
    except Exception as e:
        print("Neuspesan K-means!")
        return

    alphabet = ['A','B','C','Č','Ć','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','Š','T','U','V','W','X','Y','Z','Ž',
                'a','b','c','č','ć','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','š','t','u','v','w','x','y','z','ž']
    
    inputs = prepare_for_ann(characters)
    results = trained_model.predict(np.array(inputs, np.float32))
    sentence = display_result_sentence(results, alphabet, k_means) # Vraca recenicu
    print(sentence) # Ispis recenice koju je prepoznala neurnska mreza

    # Trazenje najslicnijih reci iz recnika
    words_in_sentece = sentence.split(" ") # Dobije se lista svih reci iz recenice
    words_in_dictionary = vocabulary.keys() # Dobije se lista svih reci iz recnika
    max_identity_sentece = ""
    max_identity_word = ""
    max_identity_ratio = 0
    for word in words_in_sentece:
        '''if(word == "mz"):
            word = "my"
        elif(word == "wIll"):
            word = "will"
        elif(word == "WIll" or word == "Will"):
            word = "will"'''
        for dictionary_word in words_in_dictionary:
            if(fuzz.ratio(word, dictionary_word) > max_identity_ratio):
                max_identity_ratio = fuzz.ratio(word, dictionary_word)
                max_identity_word = dictionary_word
        max_identity_sentece += max_identity_word + " "
        max_identity_word = ""
        max_identity_ratio = 0

    extracted_text = max_identity_sentece.strip() # Poziva se strip() da bi se uklonio razmak sa kraja
    print(max_identity_sentece) # Ispis najslicnije recenice
    return extracted_text

# Osnovne metode sa prvog izazova
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_gray_params(image):
    image_gray = np.ndarray((image.shape[0], image.shape[1]))
    image_gray = 0 * image[:, :, 0] + 0.1 * image[:, :, 1] + 0.9 * image[:, :, 2] # Podesavanje parametara
    image_gray = image_gray.astype('uint8')
    return image_gray

def image_bin_adaptive_threshold_mean(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5) # Podesavanje parametara
    return image_bin

def image_bin_adaptive_threshold_gaussian(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5) # Podesavanje parametara
    return image_bin

def image_bin_threshold_binary(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY) # Podesavanje praga
    return image_bin

def image_bin_threshold_otsu(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)
    return image_bin

def image_bin_threshold_otsu2(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)
    return ret, image_bin

def invert_image(image):
    return 255-image

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

# Dodaj metode za Open i Close !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ROI - Izdvajanje regiona od interesa
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def find_parts_of_letters(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    parts_of_letters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if y < 80:
            if area > 200 and area < 1100:
                #print('x = ' + str(x) + ' y = ' + str(y) + ' w = ' + str(w) + ' h = ' + str(h))
                #print('area = ' + str(area))
                region = image_bin[y:y + h + 1, x:x + w + 1]
                parts_of_letters.append([resize_region(region), (x, y, w, h)])
                #cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return parts_of_letters

def select_roi_small_letters(image_orig, image_bin):
    part_of_letter_regions = find_parts_of_letters(image_orig, image_bin) # Vraca konture koje su delovi od latinicnih slova
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        pom_letter_regions = []
        exist = False
        if area > 2000 and h > 110 and w > 15:
            region = image_bin[y:y+h+1,x:x+w+1]
            pom_letter_regions.append([resize_region(region), (x,y,w,h)])
            for part_of_letter_region in part_of_letter_regions:
                if(x < part_of_letter_region[1][0] and part_of_letter_region[1][0] < x + w and x != 0 and y != 0):
                    exist = True
                    region = image_bin[part_of_letter_region[1][1]:part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15, x:x + w + 1]
                    #print('Xod - ' + str(x) + ' Xdo - '+ str(x+w+1) + ' Yod ' + str(part_of_letter_region[1][1]) + ' Ydo - ' + str(part_of_letter_region[1][1] + part_of_letter_region[1][3] + h) )
                    regions_array.append([resize_region(region), (x, part_of_letter_region[1][1], w, part_of_letter_region[1][3] + h + 15)])
                    cv2.rectangle(image_orig, (x, part_of_letter_region[1][1]), (x + w, part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15), (0, 255, 0), 2)
            if(exist == False and x!= 0 and y != 0):
                    regions_array.append([resize_region(region), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0),2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions # Vraca org sliku i listu koja sadrzi regione sa org slike

def select_roi_big_letters(image_orig, image_bin):
    part_of_letter_regions = find_parts_of_letters(image_orig, image_bin) # Vraca konture koje su delovi od latinicnih slova
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        pom_letter_regions = []
        exist = False
        if area > 2000 and h > 150 and w > 20:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            pom_letter_regions.append([resize_region(region), (x, y, w, h)])
            for part_of_letter_region in part_of_letter_regions:
                if(x < part_of_letter_region[1][0] and part_of_letter_region[1][0] < x + w and x != 0 and y != 0):
                    exist = True
                    region = image_bin[part_of_letter_region[1][1]:part_of_letter_region[1][1] + part_of_letter_region[1][3] + h, x:x + w + 1]
                    #print('Xod - ' + str(x) + ' Xdo - '+ str(x+w+1) + ' Yod ' + str(part_of_letter_region[1][1]) + ' Ydo - ' + str(part_of_letter_region[1][1] + part_of_letter_region[1][3] + h) )
                    regions_array.append([resize_region(region), (x, part_of_letter_region[1][1], w, part_of_letter_region[1][3] + h)])
                    cv2.rectangle(image_orig, (x, part_of_letter_region[1][1]), (x + w, part_of_letter_region[1][1] + part_of_letter_region[1][3] + h), (0, 255, 0), 2)
            if(exist == False and x!= 0 and y != 0):
                    regions_array.append([resize_region(region), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0),2)

    # Sortiranje regiona po x osi (sa leva na desno)
    regions_array = sorted(regions_array, key=lambda item: item[1][0]) # [1][0] - uzima x i sortira po toj koortdinati
    sorted_regions = [region[0] for region in regions_array] # Uzima Prvi param koji predstavlja region sa org slike skaliran na 28x28

    return image_orig, sorted_regions # Vraca org sliku i listu koja sadrzi regione sa org slike

def select_roi_with_distances(image_orig, image_bin):
    # Za pomoc pri rotiranju slike koriscen kod sa sajta: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    image_coords = np.column_stack(np.where(image_bin > 0))
    leaning_angle = cv2.minAreaRect(image_coords)[-1]
    if leaning_angle < -30:
        leaning_angle = -90 - leaning_angle
    else:
        leaning_angle = -leaning_angle
    if (float(format(leaning_angle)) > 1.0 or float(format(leaning_angle)) < -1.0): # Ako je tekst slike iskrivljen
        (height, width) = image_orig.shape[:2]
        image_center = (width // 2, height // 2)
        m = cv2.getRotationMatrix2D(image_center, leaning_angle, 1.0)
        rotated_image = cv2.warpAffine(image_orig, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #print("[INFO] angle: {:.3f}".format(leaning_angle))
        image_orig = rotated_image

        image_bin = invert_image(image_bin_threshold_otsu(image_gray(image_orig)))  ## *********************

        #display_image(image_bin)
        part_of_letter_regions = find_parts_of_letters(image_orig,image_bin)  # Vraca konture koje su delovi od latinicnih slova
        img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_regions = []
        regions_array = []
        pom = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            pom_letter_regions = []
            exist = False
            if area > 150 and hierarchy[0, pom, 3] == -1:
                # print('x = ' + str(x) + ' y = ' + str(y) + ' w = ' + str(w) + ' h = ' + str(h))
                # print('area = ' + str(area))
                region = image_bin[y:y + h + 1, x:x + w + 1]
                pom_letter_regions.append([resize_region(region), (x, y, w, h)])
                for part_of_letter_region in part_of_letter_regions:
                    if (x < part_of_letter_region[1][0] and part_of_letter_region[1][0] < x + w and x != 0 and y != 0):
                        exist = True
                        region = image_bin[part_of_letter_region[1][1]:part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15, x:x + w + 1]
                        # print('Xod - ' + str(x) + ' Xdo - '+ str(x+w+1) + ' Yod ' + str(part_of_letter_region[1][1]) + ' Ydo - ' + str(part_of_letter_region[1][1] + part_of_letter_region[1][3] + h) )
                        regions_array.append([resize_region(region), (x, part_of_letter_region[1][1], w, part_of_letter_region[1][3] + h + 15)])
                        cv2.rectangle(image_orig, (x, part_of_letter_region[1][1]), (x + w, part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15), (0, 255, 0), 2)
                if (exist == False and x != 0 and y != 0):
                    regions_array.append([resize_region(region), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pom += 1

        regions_array = sorted(regions_array, key=lambda item: item[1][0])

        sorted_regions = [region[0] for region in regions_array]
        sorted_rectangles = [region[1] for region in regions_array]
        region_distances = []
        # Izdvojiti sortirane parametre opisujućih pravougaonika
        # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
        for index in range(0, len(sorted_rectangles) - 1):
            current = sorted_rectangles[index]
            next_rect = sorted_rectangles[index + 1]
            distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
            region_distances.append(distance)

        return image_orig, sorted_regions, region_distances
    else:
        (height, width) = image_orig.shape[:2]
        image_center = (width // 2, height // 2)
        m = cv2.getRotationMatrix2D(image_center, leaning_angle, 1.0)
        rotated_image = cv2.warpAffine(image_bin, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #print("[INFO] angle: {:.3f}".format(angle))
        #display_image(invert_image(rotated), False) # Prikaz rotirane slike
        #image_bin = invert_image(rotated) # Stavljanje rotiranje slike u procesiranje
        image_bin = rotated_image
        #display_image(image_bin, False)

    part_of_letter_regions = find_parts_of_letters(image_orig, image_bin)  # Vraca konture koje su delovi od latinicnih slova
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    pom = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        pom_letter_regions = []
        exist = False
        if area > 820 and hierarchy[0, pom, 3] == -1:
            #print('x = ' + str(x) + ' y = ' + str(y) + ' w = ' + str(w) + ' h = ' + str(h))
            #print('area = ' + str(area))
            region = image_bin[y:y + h + 1, x:x + w + 1]
            pom_letter_regions.append([resize_region(region), (x, y, w, h)])
            for part_of_letter_region in part_of_letter_regions:
                if (x < part_of_letter_region[1][0] and part_of_letter_region[1][0] < x + w and x != 0 and y != 0):
                    exist = True
                    region = image_bin[part_of_letter_region[1][1]:part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15, x:x + w + 1]
                    # print('Xod - ' + str(x) + ' Xdo - '+ str(x+w+1) + ' Yod ' + str(part_of_letter_region[1][1]) + ' Ydo - ' + str(part_of_letter_region[1][1] + part_of_letter_region[1][3] + h) )
                    regions_array.append([resize_region(region),(x, part_of_letter_region[1][1], w, part_of_letter_region[1][3] + h + 15)])
                    cv2.rectangle(image_orig, (x, part_of_letter_region[1][1]),(x + w, part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15),(0, 255, 0), 2)
            if (exist == False and x != 0 and y != 0):
                regions_array.append([resize_region(region), (x, y, w, h)])
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pom += 1

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) # X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

def select_roi_with_distances_blue(image_orig, image_bin):
    # Za pomoc pri rotiranju slike koriscen kod sa sajta: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    image_coords = np.column_stack(np.where(image_bin > 0))
    leaning_angle = cv2.minAreaRect(image_coords)[-1]
    if leaning_angle < -30:
        leaning_angle = -90 - leaning_angle
    else:
        leaning_angle = -leaning_angle
    if (float(format(leaning_angle)) > 1.0 or float(format(leaning_angle)) < -1.0): # Ako je tekst slike iskrivljen
        (height, width) = image_orig.shape[:2]
        image_center = (width // 2, height // 2)
        m = cv2.getRotationMatrix2D(image_center, leaning_angle, 1.0)
        rotated_image = cv2.warpAffine(image_bin, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #print("[INFO] angle: {:.3f}".format(leaning_angle))
        image_bin = rotated_image

        #display_image(image_bin)
        part_of_letter_regions = find_parts_of_letters(image_orig,image_bin)  # Vraca konture koje su delovi od latinicnih slova
        img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_regions = []
        regions_array = []
        pom = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            pom_letter_regions = []
            exist = False
            if area > 150 and hierarchy[0, pom, 3] == -1:
                region = image_bin[y:y + h + 1, x:x + w + 1]
                pom_letter_regions.append([resize_region(region), (x, y, w, h)])
                for part_of_letter_region in part_of_letter_regions:
                    if (x < part_of_letter_region[1][0] and part_of_letter_region[1][0] < x + w and x != 0 and y != 0):
                        exist = True
                        region = image_bin[part_of_letter_region[1][1]:part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15, x:x + w + 1]
                        # print('Xod - ' + str(x) + ' Xdo - '+ str(x+w+1) + ' Yod ' + str(part_of_letter_region[1][1]) + ' Ydo - ' + str(part_of_letter_region[1][1] + part_of_letter_region[1][3] + h) )
                        regions_array.append([resize_region(region), (x, part_of_letter_region[1][1], w, part_of_letter_region[1][3] + h + 15)])
                        cv2.rectangle(image_orig, (x, part_of_letter_region[1][1]), (x + w, part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15), (0, 255, 0), 2)
                if (exist == False and x != 0 and y != 0):
                    regions_array.append([resize_region(region), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pom += 1

        regions_array = sorted(regions_array, key=lambda item: item[1][0])

        sorted_regions = [region[0] for region in regions_array]
        sorted_rectangles = [region[1] for region in regions_array]
        region_distances = []
        # Izdvojiti sortirane parametre opisujućih pravougaonika
        # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
        for index in range(0, len(sorted_rectangles) - 1):
            current = sorted_rectangles[index]
            next_rect = sorted_rectangles[index + 1]
            distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
            region_distances.append(distance)

        return image_orig, sorted_regions, region_distances
    else:
        (height, width) = image_orig.shape[:2]
        image_center = (width // 2, height // 2)
        m = cv2.getRotationMatrix2D(image_center, leaning_angle, 1.0)
        rotated_image = cv2.warpAffine(image_bin, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        image_bin = rotated_image
        #display_image(image_bin, False)

    part_of_letter_regions = find_parts_of_letters(image_orig, image_bin)  # Vraca konture koje su delovi od latinicnih slova
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    pom = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        pom_letter_regions = []
        exist = False
        if area > 820 and hierarchy[0, pom, 3] == -1:
            #print('x = ' + str(x) + ' y = ' + str(y) + ' w = ' + str(w) + ' h = ' + str(h))
            #print('area = ' + str(area))
            region = image_bin[y:y + h + 1, x:x + w + 1]
            pom_letter_regions.append([resize_region(region), (x, y, w, h)])
            for part_of_letter_region in part_of_letter_regions:
                if (x < part_of_letter_region[1][0] and part_of_letter_region[1][0] < x + w and x != 0 and y != 0):
                    exist = True
                    region = image_bin[part_of_letter_region[1][1]:part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15, x:x + w + 1]
                    # print('Xod - ' + str(x) + ' Xdo - '+ str(x+w+1) + ' Yod ' + str(part_of_letter_region[1][1]) + ' Ydo - ' + str(part_of_letter_region[1][1] + part_of_letter_region[1][3] + h) )
                    regions_array.append([resize_region(region),(x, part_of_letter_region[1][1], w, part_of_letter_region[1][3] + h + 15)])
                    cv2.rectangle(image_orig, (x, part_of_letter_region[1][1]),(x + w, part_of_letter_region[1][1] + part_of_letter_region[1][3] + h + 15),(0, 255, 0), 2)
            if (exist == False and x != 0 and y != 0):
                regions_array.append([resize_region(region), (x, y, w, h)])
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pom += 1

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) # X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

# Priprema podataka za obucavanje NM
def scale_to_range(image):
    return image/255 # Skalira elemente slike na opseg od 0 do 1

def matrix_to_vector(image):
    return image.flatten() # Sliku koja je zapravo matrica 28x28 transformise u vektor sa 784 elementa

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region) # Skalira elemente regiona na opsek od 0 do 1
        ready_for_ann.append(matrix_to_vector(scale)) # Pretrvara skalirane elemente u vektor

    return ready_for_ann # Vraca regione koji su skalirani u opseg od 0 do 1 i transformisani u vektor od 784 elemenata

def convert_output(alphabet):
    return np.eye(len(alphabet))

# Obucavanje neuronske mreze
def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid')) # Prvi sloj, ima 128 izlaznih neurona, ulaz je dimenzije 784
    ann.add(Dense(60, activation='sigmoid')) # Drugi (izlazni) sloj, ima 60 izlaznih neurona (za svako slovo alfabeta)
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)  # Ulazi
    y_train = np.array(y_train, np.float32)  # Zeljeni izlazi

    # Definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(X_train, y_train, epochs=3000, batch_size=1, verbose=0, shuffle=False) # Obucavanje

    return ann

# Odredjivanje pobednickog neurona
def winner(output): # Output je vektor sa izlaza neuronske mreze
    return max(enumerate(output), key=lambda x: x[1])[0] # Vraca indeks neurona koji je najvise pobudjen

# Prikaz rezultata bez K-means
def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result # Iz alfabeta vraca slovo ciji je indeks najvise pobudjen

# Prikaz rezultata sa K-means
def display_result_sentence(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result # Vraca recenicu

# Serijalizacija i ucitavanje neuronske mreze
def serialize_ann(ann):
    model_json = ann.to_json() # Serijalizuj arhitekturu neuronske mreze u JSON fajl
    with open("serialized_model/neuronska.json", "w") as json_file:
        json_file.write(model_json)

    ann.save_weights("serialized_model/neuronska.h5") # Serijalizuj tezine u HDF5 fajl

def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialized_model/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)

        ann.load_weights("serialized_model/neuronska.h5") # Ucitaj tezine u prethodno kreirani model
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # Ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        print("Ucitavanje neuronske mreze nije uspelo!")
        return None