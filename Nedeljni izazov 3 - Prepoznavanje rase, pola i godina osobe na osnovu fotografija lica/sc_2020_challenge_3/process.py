# import libraries here
import cv2 # OpenCV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load

from imutils import face_utils
import argparse
import imutils
import dlib

matplotlib.rcParams['figure.figsize'] = 10,6

def train_or_load_age_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    '''for age_label in train_image_labels:
            print(age_label)'''

    train_images = []  # Ucitane slike za treniranje
    for train_image_path in train_image_paths:
        train_images.append(load_image(train_image_path))

    '''for train_image in train_images: # Iscrtavanje rgb slika
        print(train_image.shape) # Dimenzije slika su (200, 200, 3)
        display_image(train_image, True)'''

    age_features = []
    age_labels = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cnt = 0
    for train_image in train_images:
        gray_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(train_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(train_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features =  shape_points_distances(shape)
            age_features.append(features)
            age_labels.append(train_image_labels[cnt])
        cnt += 1

    age_features = np.array(age_features)
    x = age_features
    y = np.array(age_labels)

    # Podela trening skupa na trening i validacioni
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Transformisanje x_train i x_test u oblik pogodan za scikit-learn
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)
    x = reshape_data(x)
    print('X_train_AGE after resharpe: ' + str(x.shape))

    # KNN klasifikator
    clf_svm = KNeighborsClassifier(n_neighbors=11)
    clf_svm = clf_svm.fit(x, y)
    # SVM klasifikatora
    '''clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x, y)  # Treniranje modela'''
    # Serijalizacija i deserijalizacija modela
    #dump(clf_svm, 'serialized_folder/svm_age.joblib')
    clf_svm = load('serialized_folder/svm_age.joblib')
    '''y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))'''

    model = clf_svm
    return model

def train_or_load_gender_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    train_images = []  # Ucitane slike za treniranje
    for train_image_path in train_image_paths:
        train_images.append(load_image(train_image_path)) # Dimenzije slike (200, 200)

    # Pronalazenje slika na kojima su muske i zenske osobe
    man_images = []
    woman_images = []
    i = 0
    for gender_label in train_image_labels:
        if(int(gender_label) == 0):
            man_images.append(train_images[i])
            i += 1
        elif (int(gender_label) == 1):
            woman_images.append(train_images[i])
            i += 1

    '''# HOG Deskriptor
    nbins = 9  # Broj binova
    cell_size = (8, 8)  # Broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(train_images[0].shape[1] // cell_size[1] * cell_size[1], # Sve su slike istih dimenzija pa moze train_images[0]
                                      train_images[0].shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)'''

    '''for man_image in man_images:
            man_features.append(hog.compute(man_image))
            gender_labels.append(0)

        for woman_image in woman_images:
            woman_features.append(hog.compute(woman_image))
            gender_labels.append(1)'''

    man_features = []
    woman_features = []
    gender_labels = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    for man_image in man_images:
        gray_image = cv2.cvtColor(man_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(man_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(man_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features = shape_points_distances(shape)
            man_features.append(features)
            gender_labels.append(0)

    for woman_image in woman_images:
        #display_image(woman_image, False)
        gray_image = cv2.cvtColor(woman_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1) # Na 6-toj slici ne pronadje lice !!!!!!
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(woman_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(woman_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features = shape_points_distances(shape)
            woman_features.append(features)
            gender_labels.append(1)
            #display_image(woman_image, False)

    man_features = np.array(man_features)
    woman_features = np.array(woman_features)
    x = np.vstack((man_features, woman_features))
    y = np.array(gender_labels)

    # Podela trening skupa na trening i validacioni
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Transformisanje x_train i x_test u oblik pogodan za scikit-learn
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)
    x = reshape_data(x)
    print('X_train_GENDER after resharpe: ' + str(x.shape))

    # SVM klasifikatora
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x, y) # Treniranje modela
    # Serijalizacija i deserijalizacija modela
    #dump(clf_svm, 'serialized_folder/svm_gender.joblib')
    clf_svm = load('serialized_folder/svm_gender.joblib')
    '''y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))'''

    # KNN klasifikator
    '''clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn = clf_knn.fit(x_train, y_train)
    # Serijalizacija i deserijalizacija modela klasifikatora
    #dump(clf_knn, 'knn.joblib')
    #clf_knn = load('knn.joblib')
    y_train_pred = clf_knn.predict(x_train)
    y_test_pred = clf_knn.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))'''

    model = clf_svm
    return model


def train_or_load_race_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    train_images = []  # Ucitane slike za treniranje
    for train_image_path in train_image_paths:
        train_images.append(load_image(train_image_path))

    # Pronalazenje slika na kojima su rase (0 - Bela, 1 - Crna, 2 - Azijati, 3 - Indijci, 4 - Ostali)
    white_race_images = []
    black_race_images = []
    asian_race_images = []
    indian_race_images = []
    others_race_images = []
    i = 0
    for race_label in train_image_labels:
        if (int(race_label) == 0):
            white_race_images.append(train_images[i])
            i += 1
        elif (int(race_label) == 1):
            black_race_images.append(train_images[i])
            i += 1
        elif (int(race_label) == 2):
            asian_race_images.append(train_images[i])
            i += 1
        elif (int(race_label) == 3):
            indian_race_images.append(train_images[i])
            i += 1
        elif (int(race_label) == 4):
            others_race_images.append(train_images[i])
            i += 1

    white_race_features = []
    black_race_features = []
    asian_race_features = []
    indian_race_features = []
    others_race_features = []
    race_labels = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    for white_race_image in white_race_images:
        gray_image = cv2.cvtColor(white_race_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(white_race_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(white_race_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features = shape_points_distances(shape)
            white_race_features.append(features)
            race_labels.append(0)

    for black_race_image in black_race_images:
        gray_image = cv2.cvtColor(black_race_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(black_race_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(black_race_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features = shape_points_distances(shape)
            black_race_features.append(features)
            race_labels.append(1)

    for asian_race_image in asian_race_images:
        gray_image = cv2.cvtColor(asian_race_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(asian_race_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(asian_race_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features = shape_points_distances(shape)
            asian_race_features.append(features)
            race_labels.append(2)

    for indian_race_image in indian_race_images:
        gray_image = cv2.cvtColor(indian_race_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(indian_race_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(indian_race_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features = shape_points_distances(shape)
            indian_race_features.append(features)
            race_labels.append(3)

    for others_race_image in others_race_images:
        gray_image = cv2.cvtColor(others_race_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
            shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

            # Konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # Crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(others_race_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(others_race_image, (x, y), 1, (0, 0, 255), -1)
            features = shape
            #features =  shape_points_distances(shape)
            others_race_features.append(features)
            race_labels.append(4)

    white_race_features = np.array(white_race_features)
    black_race_features = np.array(black_race_features)
    asian_race_features = np.array(asian_race_features)
    indian_race_features = np.array(indian_race_features)
    others_race_features = np.array(others_race_features)
    x = np.vstack((white_race_features, black_race_features, asian_race_features, indian_race_features, others_race_features))
    y = np.array((race_labels))

    # Podela trening skupa na trening i validacioni
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Transformisanje x_train i x_test u oblik pogodan za scikit-learn
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)
    x = reshape_data(x)
    print('X_train_RACE after resharpe: ' + str(x.shape))

    # KNN klasifikator
    clf_svm = KNeighborsClassifier(n_neighbors=11)
    clf_svm = clf_svm.fit(x, y)
    # SVM klasifikatora
    '''clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x, y) # Treniranje modela'''
    # Serijalizacija i deserijalizacija modela
    #dump(clf_svm, 'serialized_folder/svm_race.joblib')
    clf_svm = load('serialized_folder/svm_race.joblib')
    '''y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))'''

    model = clf_svm
    return model

def predict_age(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje godina i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati godine.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje godina
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati godine lica
    :return: <Int> Prediktovanu vrednost za goinde  od 0 do 116
    """
    age = 0
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    image_for_predict = load_image(image_path) # Ucitavanje slike

    #print(image_for_predict.shape) # Dimenzije slika su razne!!!!!!!
    #display_image(image_for_predict, True) # Iscrtavanje RGB slike

    '''# Setajuci prozor
    for (width, height, window) in sliding_window(image_for_predict, stepSize=20, windowSize=(200, 200)):

        if window.shape[0] != 200 or window.shape[1] != 200:
            continue

        # Racunaj za svaki sliding window HOG
        # ............

        image_with_sliding_window = image_for_predict.copy()
        cv2.rectangle(image_with_sliding_window, (width, height), (width + 200, height + 200), (0, 255, 0), 2)
        display_image(image_with_sliding_window, True)'''

    detector = dlib.get_frontal_face_detector()  # Inicijalizaclija dlib detektora (HOG)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka

    gray_image = cv2.cvtColor(image_for_predict, cv2.COLOR_BGR2GRAY)

    rects = detector(gray_image, 1)  # Vraca detektovano lice

    for (i, rect) in enumerate(rects):
        shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
        shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

        # Konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # Crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image_for_predict, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crtanje kljucnih tacaka
        for (x, y) in shape:
            cv2.circle(image_for_predict, (x, y), 1, (0, 0, 255), -1)
        features = shape
        #features = shape_points_distances(shape)
        #x_for_predict = np.array(res)
        #display_image(image_for_predict, True)
        x_for_predict = features.reshape(1, -1)
        # print('X_predict after resharpe: ' + str(x_for_predict.shape))

        predict_value_of_age = trained_model.predict(x_for_predict)
        print('Predvidjen broj godina: ' + str(int(predict_value_of_age)))

        for i in range(0, 116):
            if(int(predict_value_of_age) == i):
                age = int(i)
                break

    return age

def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """
    gender = 1
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    detector = dlib.get_frontal_face_detector() # Inicijalizaclija dlib detektora (HOG)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # Ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka

    image_for_predict = load_image(image_path)
    gray_image = cv2.cvtColor(image_for_predict, cv2.COLOR_BGR2GRAY)

    rects = detector(gray_image, 1)  # Vraca detektovano lice

    for (i, rect) in enumerate(rects):
        shape = predictor(gray_image, rect) # Vraca kljucne tacke prepoznatog lica
        shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata
        '''print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        print("Prva 3 elementa matrice")
        print(shape[:3])'''

        # Konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # Crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image_for_predict, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crtanje kljucnih tacaka
        for (x, y) in shape:
            cv2.circle(image_for_predict, (x, y), 1, (0, 0, 255), -1)
        features = shape
        #features = shape_points_distances(shape)
        #x_for_predict = np.array(res)
        #display_image(image_for_predict, True)
        x_for_predict = features.reshape(1, -1)
        #print('X_predict after resharpe: ' + str(x_for_predict.shape))

        predict_value_of_gender = trained_model.predict(x_for_predict)
        print('Predvidjen pol: ' + str(int(predict_value_of_gender)))

        if(int(predict_value_of_gender) == 0):
            gender = 0
        elif(int(predict_value_of_gender) == 1):
            gender = 1

    return gender

def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    race = 4
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    detector = dlib.get_frontal_face_detector()  # Inicijalizaclija dlib detektora (HOG)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka

    image_for_predict = load_image(image_path)
    gray_image = cv2.cvtColor(image_for_predict, cv2.COLOR_BGR2GRAY)

    rects = detector(gray_image, 1)  # Vraca detektovano lice

    for (i, rect) in enumerate(rects):
        shape = predictor(gray_image, rect)  # Vraca kljucne tacke prepoznatog lica
        shape = face_utils.shape_to_np(shape)  # Konverzija u NumPy niz, shape predstavlja 68 koordinata

        # Konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # Crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image_for_predict, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crtanje kljucnih tacaka
        for (x, y) in shape:
            cv2.circle(image_for_predict, (x, y), 1, (0, 0, 255), -1)
        features = shape
        #features = shape_points_distances(shape)
        #x_for_predict = np.array(res)
        #display_image(image_for_predict, True)
        x_for_predict = features.reshape(1, -1)
        # print('X_predict after resharpe: ' + str(x_for_predict.shape))

        predict_value_of_race = trained_model.predict(x_for_predict)
        print('Predvidjena rasa: ' + str(int(predict_value_of_race)))

        if (int(predict_value_of_race) == 0):
            race = 0
        elif (int(predict_value_of_race) == 1):
            race = 1
        elif (int(predict_value_of_race) == 2):
            race = 2
        elif (int(predict_value_of_race) == 3):
            race = 3
        elif (int(predict_value_of_race) == 4):
            race = 4

    return race

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

# Transformisanje x_train i x_test u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

# Setajuci prozor
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]]) # Vraca trenutni window

def shape_points_distances(shape):
    eyebrows_points_distances = []
    eye_points_distances = []
    nose_points_distances = []
    mouth_points_distances = []
    result = []

    # Leva i desna obrva
    eyebrows_points_distances.append(calc_distance(shape[18][0], shape[18][1], shape[20][0], shape[20][1]))
    eyebrows_points_distances.append(calc_distance(shape[20][0], shape[20][1], shape[22][0], shape[22][1]))
    eyebrows_points_distances.append(calc_distance(shape[23][0], shape[23][1], shape[25][0], shape[25][1]))
    eyebrows_points_distances.append(calc_distance(shape[25][0], shape[25][1], shape[27][0], shape[27][1]))

    # Levo i desno oko
    eye_points_distances.append(calc_distance(shape[37][0], shape[37][1], shape[40][0], shape[40][1]))
    eye_points_distances.append(calc_distance(shape[38][0], shape[38][1], shape[42][0], shape[42][1]))
    eye_points_distances.append(calc_distance(shape[39][0], shape[39][1], shape[41][0], shape[41][1]))
    eye_points_distances.append(calc_distance(shape[43][0], shape[43][1], shape[46][0], shape[46][1]))
    eye_points_distances.append(calc_distance(shape[44][0], shape[44][1], shape[48][0], shape[48][1]))
    eye_points_distances.append(calc_distance(shape[45][0], shape[45][1], shape[47][0], shape[47][1]))

    # Nos
    nose_points_distances.append(calc_distance(shape[28][0], shape[28][1], shape[31][0], shape[31][1]))
    nose_points_distances.append(calc_distance(shape[32][0], shape[32][1], shape[34][0], shape[34][1]))
    nose_points_distances.append(calc_distance(shape[34][0], shape[34][1], shape[36][0], shape[36][1]))

    # Usta
    mouth_points_distances.append(calc_distance(shape[49][0], shape[49][1], shape[55][0], shape[55][1]))
    mouth_points_distances.append(calc_distance(shape[50][0], shape[50][1], shape[60][0], shape[60][1]))
    mouth_points_distances.append(calc_distance(shape[51][0], shape[51][1], shape[59][0], shape[59][1]))
    mouth_points_distances.append(calc_distance(shape[52][0], shape[52][1], shape[58][0], shape[58][1]))
    mouth_points_distances.append(calc_distance(shape[53][0], shape[53][1], shape[57][0], shape[57][1]))
    mouth_points_distances.append(calc_distance(shape[54][0], shape[54][1], shape[56][0], shape[56][1]))

    for distance in eyebrows_points_distances:
        result.append(distance)

    for distance in eye_points_distances:
        result.append(distance)

    for distance in nose_points_distances:
        result.append(distance)

    for distance in mouth_points_distances:
        result.append(distance)

    '''result.append(eyebrows_points_distances)
    result.append(eye_points_distances)
    result.append(nose_points_distances)
    result.append(mouth_points_distances)'''

    return  result

def calc_distance(x1, y1, x2, y2):
    x_dist = (x2 - x1)
    y_dist = (y2 - y1)
    return np.sqrt(x_dist * x_dist + y_dist * y_dist)