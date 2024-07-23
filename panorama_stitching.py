import cv2
import numpy as np
import glob
import imutils


video = cv2.VideoCapture("video13.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_length = frame_count / fps
num_frames_to_capture = int( fps * video_length)

def PostProcessing(stitched_image):
#creating a black border around the image, 10 pixels in all directions
    stitched_image = cv2.copyMakeBorder(stitched_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    #cv2.imshow("CopyMakeBorder image", stitched_image)
    #finding out where the border is so we can subctract all the black pixels, the picture is white while the border is black
    
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Threshold image", thresh_img)
    cv2.waitKey(0)


    # finding all the rounded contours in our binary image
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # grabbing a maximum contour of our stitched image
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key = cv2.contourArea)
    #cv2.drawContours(stitched_image, contours, -1, (0, 255, 0), 3)
    #cv2.drawContours(stitched_image, areaOI, -1, (0, 255, 0), 3)
    #cv2.imshow("Contoured image", stitched_image)
    
    #we use uint8 because we are working with binary image
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    #print(mask)
    # creating rectangle from our area of interest which is max contour, we are cropping the image 
    # boundingRect is used to draw an approximate rectangle around the binary image, used to highlight region of interest after obtaining contours
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
    minRectangle = mask.copy()
    #cv2.imshow("min rectangle before subtract", minRectangle)
    sub = mask.copy()

    # this loop is running as long as the count isn't zero
    # minRectangle is where we want our image to be cropped
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)
        #cv2.imshow('minRectangle image', minRectangle)
        #cv2.waitKey(0)
        
    #finding contours now for the cropped image  
    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key = cv2.contourArea)
    #cv2.drawContours(stitched_image, contours, -1, (0, 255, 0), 3)
    #cv2.drawContours(stitched_image, areaOI, -1, (0, 255, 0), 3)
    #cv2.imshow('minRectangle image', minRectangle)
    #cv2.waitKey(0)

    #cropping our image to rectangle size
    x, y, w, h = cv2.boundingRect(areaOI)
    stitched_image = stitched_image[y: y + h, x: x + w]
    cv2.namedWindow("Stitched and Processed Image", cv2.WINDOW_NORMAL)
    cv2.imwrite("stitchedProcessed.png", stitched_image)
    cv2.imshow("Stitched and Processed Image", stitched_image)
    return stitched_image



images = []
for i in range(num_frames_to_capture):
    ret, frame = video.read()
    if i % int(fps) == 0 or (fps == fps // 2):
        # Write the frame
        # it worked for 0.5 0.2
        resized = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        cv2.imwrite('panorama_images/frame_%d.png' % i, resized)  
        images.append(resized)      
video.release()


  
imageStitcher = cv2.Stitcher_create()
error, stitched_image = imageStitcher.stitch(images)


if not error:
    cv2.imwrite('stitchedOutput.png', stitched_image)
    cv2.namedWindow("Stitched image", cv2.WINDOW_NORMAL)
    cv2.imshow('Stitched image', stitched_image)
    #cv2.resizeWindow('Stitched image', (500, 200))
    stitched_image = PostProcessing(stitched_image)
    cv2.waitKey(0)
    
    
else:
    print("Images could not be stitched together")