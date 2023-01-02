
from flask import Flask, request, jsonify
import cv2
import sys
from imutils import contours
import numpy as np
import pytesseract


def preProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(points):
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew


app = Flask(__name__)


@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")

    response = {}

    # Check if the user sent a name at all
    if not name:
        response["ERROR"] = "No name found. Please send a name."
    # Check if the user entered a number
    elif str(name).isdigit():
        response["ERROR"] = "The name can't be numeric. Please send a string."
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome API!"

    # Return the response in json format
    return jsonify(response)


@app.route('/parse/', methods=['POST'])
def post_something():
    heightImg = 450
    widthImg = 450
    param = request.files.get('photo')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        # pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
        # Load image, grayscale, and adaptive threshold

        sudoku_string = ""

        file_bytes = np.fromfile(param, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # image = cv2.imread('sudoku.png')
        image = cv2.resize(image, (widthImg, heightImg))
        thresh = preProcess(image)

        # Filter out all numbers and noise to isolate only boxes
        cnts, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, maxArea = biggestContour(cnts)
        if biggest.size != 0:
            biggest = reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(image, matrix, (widthImg, heightImg))
            imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            cpThresh = cv2.adaptiveThreshold(imgWarpColored, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
            thresh = cv2.adaptiveThreshold(imgWarpColored, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

            # Filter out all numbers and noise to isolate only boxes
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 1000:
                    cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
            # Fix horizontal and vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

            # Sort by top to bottom and each row by left to right
            invert = 255 - thresh
            cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
            sudoku_rows = []
            row = []
            for (i, c) in enumerate(cnts, 1):
                area = cv2.contourArea(c)
                if area < 50000:
                    row.append(c)
                    if i % 9 == 0:
                        (cnts, _) = contours.sort_contours(row, method="left-to-right")
                        sudoku_rows.append(cnts)
                        row = []
            # Iterate through each box
            for row in sudoku_rows:
                for c in row:
                    x, y, w, h = cv2.boundingRect(c)
                    rect = cv2.rectangle(cpThresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cropped = cpThresh[y:y + h, x:x + w]
                    (cH, cW) = cropped.shape[:2]
                    enlarged = cv2.resize(cropped, (cH*3, cW*3))
                    text = pytesseract.image_to_string(enlarged, lang="eng", config='--psm 6 --oem 1 -c tessedit_char_whitelist=123456789')
                    text = text.strip()
                    sudoku_string += text or '0'
        return jsonify({'sudokuString': sudoku_string, "METHOD": "POST"})
    else:
        return jsonify({
            "ERROR": "No photo found. Please send a photo."
        })


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Hello World</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
