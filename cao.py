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
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, send_file

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/test.html")
def home2():
    return render_template("test.html")

@app.route("/im.png")
def hom():
    return send_file('templates/im.png', mimetype='image/gif')

@app.route("/op2.txt")
def op2():
    return render_template("op2.txt")

@app.route("/out3.txt")
def out3():
    return render_template("out3.txt")

@app.route("/out1.txt")
def out1():
    return render_template("out1.txt")

@app.route("/output.txt")
def output():
    return render_template("output.txt")

@app.route("/file2jnum.txt")
def file2jnum():
    return render_template("file2jnum.txt")

@app.route("/file2jtext.txt")
def file2jtext():
    return render_template("file2jtext.txt")

@app.route("/mhd-num")
def mhdnum():
    return render_template("mhd-num.txt")

@app.route("/tes")
def out4():
    configuration = cloudmersive_ocr_api_client.Configuration()
    configuration.api_key['Apikey'] = '51bd80fa-7c00-485e-a847-e779a5156784'
    vl = request.args.get('vl')
    x = "../Desktop/"+vl
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
            '''api_response = api_instance.image_ocr_post(image_file, recognition_mode=recognition_mode, language=language, preprocessing=preprocessing)
            # api_response = api_instance.receipts_photo_to_csv(image_file)
            z = api_response.text_result
            print("Text Extraction Done! \n")
            a = z.replace("\n", "|")
            l = a.split("|")
            df = pd.DataFrame(l)
            df.to_csv('templates/file2jtext.txt', index = False, sep=',')'''
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
            custom_config = r'-l jpn -c preserve_interword_spaces=1 --oem 3 --psm 6'
            gray = get_grayscale(img)
            thresh = thresholding(gray)
            # opening = opening(gray)
            # canny = canny(gray)
            d = pytesseract.image_to_string(img, config=custom_config)
            print(d)
            # with io.open('templates/file2jnum.txt', 'w',  encoding='utf-8') as f:
                # f.write(d)
                # print('Saved data in Excel Sheet!')

        except ApiException as e:
            print("Exception when calling ImageOcrApi->image_ocr_post: %s\n" % e)


    return "success"

@app.route("/extractnum", methods=["POST"])
def extractnum():
    fname = request.form.getlist('fname')
    x1 = request.form.getlist('coordinates[0][x1]')
    y1 = request.form.getlist('coordinates[0][y1]')
    x2 = request.form.getlist('coordinates[0][x2]')
    y2 = request.form.getlist('coordinates[0][y2]')
    img = Image.open(r'../Desktop/'+str(fname[0]))
    img2 = img.crop((int(x1[0]), int(y1[0]), int(x2[0]), int(y2[0])))
    img2.save("img2.jpg")
    custom_config = r'-c preserve_interword_spaces=1 --oem 3 --psm 6 -l jpn'
    d = pytesseract.image_to_data(img2, config=custom_config, output_type='data.frame')
    # df = pd.DataFrame.from_dict(d, orient='index')
    df = pd.DataFrame(d)
    # print(df)
    # clean up blanks
    df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]
    # sort blocks vertically
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    for block in sorted_blocks:
        curr = df1[df1['block_num']==block]
        sel = curr[curr.text.str.len()>3]
        char_w = (sel.width/sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        for ix, ln in curr.iterrows():
            # add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # num of spaces that should be added
            if ln['left']/char_w > prev_left + 1:
                added = int((ln['left'])/char_w) - prev_left
                text += ' ' * added
            if str(ln['text'])=='nan':
                ln['text'] = ''  
            # print(ln['text'])
            text += str(ln['text']) + ' '
            prev_left += len(str(ln['text'])) + added + 1
        text += '\n'
        print(text)
        # with io.open('templates/mhd-num.txt', 'w',  encoding='utf-8') as f:
                # f.write(text)
                # print('Saved data in text file!')
    return "proceed"


@app.route("/upload", methods=["POST"])
def upload():

    target = os.path.join(APP_ROOT, 'Athena_XML_files/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    return "heyyyyyyyyyyyyyyyyyy"


    
if __name__ == "__main__":
    app.run()
