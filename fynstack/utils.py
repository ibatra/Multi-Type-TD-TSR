import os

import cv2
import numpy as np
import pandas as pd
from google.cloud import vision
from pdf2image import convert_from_path

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'fynstack-a5916aa2d765.json'

def convert_pdf_2_img(path, img_filename=None):
    pages = convert_from_path(path)
    for i, page in enumerate(pages):
        if not img_filename:
            img_filename = path.split('/')[-1].split('.')[0]
        page.save(f"pdfs/{img_filename}_{i}.jpg", "jpeg")


def make_prediction(img_path, predictor):
    img = cv2.imread(img_path)
    outputs = predictor(img)

    table_list = []
    table_coords = []

    for i, box in enumerate(outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()):
        x1, y1, x2, y2 = box
        table_list.append(np.array(img[int(y1):int(y2), int(x1):int(x2)], copy=True))
        table_coords.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    return table_list, table_coords


def detect_text(content):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    success, encoded_image = cv2.imencode('.jpg', content)
    roi_image = encoded_image.tobytes()
    roi_image = vision.Image(content=roi_image)

    response = client.text_detection(image=roi_image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors')

    if len(texts) > 0:
        return texts[0].description

    return 'There is no text here'


def crop_white(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = img[y:y + h, x:x + w]
        return rect
    except:
        return img


def convert_boxes_2_dataframe(finalboxes, img):
    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if len(finalboxes[i][j]) == 0:
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]

                    finalimg = img[x * 2:(x + h) * 2, y * 2:(y + w) * 2]
                    finalimg = crop_white(finalimg)

                    border = cv2.copyMakeBorder(finalimg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    border = cv2.copyMakeBorder(border, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    out = detect_text(border)

                    inner = inner + " " + out[:]
                outer.append(inner)

    # Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(finalboxes), len(finalboxes[0])))
    data = dataframe.style.set_properties(align="left")
    return data
