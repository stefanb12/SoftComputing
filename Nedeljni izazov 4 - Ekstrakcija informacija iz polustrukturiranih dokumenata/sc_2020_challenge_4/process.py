# import libraries here
import datetime

import cv2 # OpenCV
import sys
import numpy as np
import matplotlib
import matplotlib.pylab as plt

# Detekcija lica
from imutils import face_utils
import argparse
import imutils
import dlib

from PIL import Image
import pyocr
import pyocr.builders

plt.rcParams['figure.figsize'] = 10, 6

class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """
    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company

def extract_info(models_folder: str, image_path: str) -> Person:
    """
    Procedura prima putanju do foldera sa modelima, u slucaju da su oni neophodni, kao i putanju do slike sa koje
    treba ocitati vrednosti. Svi modeli moraju biti uploadovani u odgovarajuci folder.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param models_folder: <str> Putanja do direktorijuma sa modelima
    :param image_path: <str> Putanja do slike za obradu
    :return:
    """
    person = Person('test', datetime.date.today(), 'test', 'test', 'test')

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    image = load_image(image_path)

    (h, w, c) = image.shape
    start_height = h
    start_width = w

    # PyOCR podrzava i neke druge alate, tako da je potrebno proveriti koji su sve alati instalirani
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #display_image(gray, False)

    rects = detector(gray, 1) # Detekcija svih lica na grayscale slici

    new_x = 0
    new_y = 0
    new_w = 0
    new_h = 0
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect) # Konvertovanje pravougaonika u bounding box koorinate

        new_x = x - 450
        new_y = y - 200
        new_w = w + 400
        new_h = h + 300

        # Ogranicavanje da ne ode van granica slike
        if (new_x < 0):
            new_x = 12
        if (new_y < 0):
            new_y = 12
        if (new_x + new_w > start_width):
            new_w = start_width - 12
        if (new_y + new_h > start_height):
            new_h = start_height - 12

        cv2.rectangle(image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)  # Crtanje pravougaonika

        '''# Crtanje kljucnih tacaka
        for (new_x, new_y) in shape:
            cv2.circle(image, (new_x, new_y), 1, (0, 0, 255), -1)'''

    # Kropovana slika sa licnom kartom
    #cropped = image[new_y:new_y + new_h, new_x:new_x + new_w]  # Isecanje licne karte sa slike
    '''if(new_x + new_w < start_width and new_y + new_h < start_height and new_x > 0 and new_y > 0):
        cropped = image[new_y:new_y + new_h, new_x:new_x + new_w]
    else:
        cropped = image'''

    cropped = image[new_y:new_y + new_h, new_x:new_x + new_w]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #ret, image_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    #image_bin = invert(image_bin)
    #thresh = dilate(erode(image_bin))
    gray = cv2.bitwise_not(gray)
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print("[INFO] angle: {:.3f}".format(angle))
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = cropped.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cropped, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Ispis ugla na slici

    image_card = thresh

    # Odaberemo Tessract - prvi na listi ako je jedini alat
    tool = tools[0]
    print("Koristimo backend: %s" % (tool.get_name()))

    text = tool.image_to_string(
        Image.fromarray(image_card),
        lang='eng',
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)  # Izbor segmentacije (PSM)
    )
    print('Ceo tekst: ' + str(text))

    '''word_boxes = tool.image_to_string(
        Image.fromarray(image_card),
        lang='eng',
        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=3)
    )
    for i, box in enumerate(word_boxes):
        print("word %d" % i)
        print(box.content, box.position, box.confidence)
        print()'''

    '''line_and_word_boxes = tool.image_to_string(
        Image.fromarray(image_card),
        lang='eng',
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=3)
    )
    for i, line in enumerate(line_and_word_boxes):
        print('line %d' % i)
        print(line.content, line.position)
        print('boxes')
        for box in line.word_boxes:
            print(box.content, box.position, box.confidence)
        print()'''

    '''digits = tool.image_to_string(
        Image.fromarray(image_card),
        lang='eng',
        builder=pyocr.tesseract.DigitBuilder(tesseract_layout=3)  # ocekivani text je single line, probati sa 3,4,5..
    )
    print('Brojevi: ' + str(digits))'''

    person_name = ''
    person_company = ''
    person_job = ''
    person_date_of_birth = ''
    try:
        person_date_of_birth = datetime.datetime.strptime(person_date_of_birth, '%d, %b %Y').strftime('%Y-%m-%d')
    except Exception as e:
        print('Greska pri konvertovanju datuma')
    person_ssn = ''

    person.name = person_name
    person.company = person_company
    person.job = person_job
    person.date_of_birth = person_date_of_birth
    person.ssn = person_ssn

    print(str(person_name) + " -> " + str(person_company) + " -> " + str(person_job) + " -> " + str(person_date_of_birth) + " -> " + str(person_ssn))

    display_image(image_card, True)

    return person

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image
        
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

