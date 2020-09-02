from __future__ import print_function
import time
import cloudmersive_ocr_api_client
from cloudmersive_ocr_api_client.rest import ApiException
import pandas as pd
import pdf2image
from pprint import pprint
import cv2 
import pytesseract
import numpy as np
import io
import os
import webbrowser
from flask import Flask, request, render_template, send_from_directory

configuration = cloudmersive_ocr_api_client.Configuration()
configuration.api_key['Apikey'] = '00f20d27-4d1b-4a27-ba88-527f8b41bc68'

x = "../Desktop/bb.png"
# input('Select Image or PDF file: ')
if x.endswith('.pdf'):
    print("Converting PDF into Image...\n")
    pil_images = pdf2image.convert_from_path(x)
    for i, image in enumerate(pil_images):
        fname = ("pages" + str(i) + ".jpg")
        image.save(fname, 'JPEG')
        print("Convertion Successful.\nNow Extracting Text from Image file...\n")

    api_instance = cloudmersive_ocr_api_client.ImageOcrApi(cloudmersive_ocr_api_client.ApiClient(configuration))
    pdf_file = fname
    recognition_mode = 'Advanced'
    language = 'JPN'
    preprocessing = 'Auto'

    try:
        api_response = api_instance.image_ocr_post(pdf_file, recognition_mode=recognition_mode, language=language, preprocessing=preprocessing)
        z = api_response.text_result
        print("Text Extraction Done! \n")
        a = z.replace("\n", ",")
        l = a.split(",")
        df = pd.DataFrame(l, columns=['table'])
        df.to_csv('test.txt', sep=' ')
        print('Saved data in Excel Sheet!')


    except ApiException as e:
        print("Exception when calling ImageOcrApi->image_ocr_post: %s\n" % e)

elif x.endswith('.jpg') or x.endswith('jpeg') or x.endswith('.png'):
    print("Extracting Text from Image...\n")
    api_instance = cloudmersive_ocr_api_client.ImageOcrApi(cloudmersive_ocr_api_client.ApiClient(configuration))
    # api_instance = cloudmersive_ocr_api_client.ReceiptsApi(cloudmersive_ocr_api_client.ApiClient(configuration))
    image_file = x
    recognition_mode = 'Advanced'
    language = 'JPN'
    preprocessing = 'Auto'

    try:
        api_response = api_instance.image_ocr_post(image_file, recognition_mode=recognition_mode, language=language, preprocessing=preprocessing)
        # api_response = api_instance.receipts_photo_to_csv(image_file)
        z = api_response.text_result
        print("Text Extraction Done! \n")
        a = z.replace("\n", "|")
        l = a.split("|")
        df = pd.DataFrame(l)
        df.to_csv('file2jtext.txt', index = False, sep=',')
        img = cv2.imread(x)

        # get grayscale image
        def get_grayscale(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # noise removal
        def remove_noise(img):
            return cv2.medianBlur(img,5)
         
        #thresholding
        def thresholding(img):
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #dilation
        def dilate(image):
            kernel = np.ones((5,5),np.uint8)
            return cv2.dilate(image, kernel, iterations = 1)
            
        #erosion
        def erode(image):
            kernel = np.ones((5,5),np.uint8)
            return cv2.erode(image, kernel, iterations = 1)

        #opening - erosion followed by dilation
        def opening(image):
            kernel = np.ones((5,5),np.uint8)
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        #canny edge detection
        def canny(image):
            return cv2.Canny(image, 100, 200)

        #skew correction
        def deskew(image):
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

        #template matching
        def match_template(image, template):
            return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Adding custom options
        custom_config = r'-l jpn --oem 3 --psm 6'
        gray = get_grayscale(img)
        thresh = thresholding(gray)
        # opening = opening(gray)
        # canny = canny(gray)
        d = pytesseract.image_to_string(img, config=custom_config)
        print(d)
        with io.open('file2jnum.txt', 'w',  encoding='utf-8') as f:
            f.write(d)
        print('Saved data in Excel Sheet!')

    except ApiException as e:
        print("Exception when calling ImageOcrApi->image_ocr_post: %s\n" % e)
