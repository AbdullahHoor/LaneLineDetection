import cv2
import numpy as np

def roi(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image

def draw_lines(img, lines):
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    print(image.shape)
    h = image.shape[0]
    w = image.shape[1]
    roi_vertices = [(0, h), (w/2, h/2),(w, h)]
    #gaussian_image=cv2.GaussianBlur(image,(5,5),0)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 150, 170)
    cropped_image = roi(canny,
                    np.array([roi_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,rho=2, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100)
    image_lines = draw_lines(image, lines)
    return image_lines

cap = cv2.VideoCapture('VID-20191229-WA0064.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()