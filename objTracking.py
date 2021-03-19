'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import cv2
from Detector import detect
from KalmanFilter import KalmanFilter
import math
import gc


def dist(x, y, x1, y1):
    return math.sqrt((x-x1)**2 + (y-y1)**2)


def getClosestKF(x, y, predicted_x, predicted_y, distance_threshold):
    index = -1
    minDist = 4444  # Length of diagonal of 4k image
    for i in range(len(predicted_x)):
        currDistance = dist(x, y, predicted_x[i], predicted_y[i])
        if(currDistance < minDist):
            minDist = currDistance
            index = i
    if(minDist <= distance_threshold):
        return index
    else:
        return -1


def predictKFs(KFs):
    predicted_x = []
    predicted_y = []
    for i in KFs:
        # Predict
        (x, y) = i.predict()
        predicted_x.append(x)
        predicted_y.append(y)
    return predicted_x, predicted_y


def main():
    # Create opencv video capture object
    # VideoCap = cv2.VideoCapture('./video/randomball.avi')
    VideoCap = cv2.VideoCapture('./video/multi.mp4')
    # VideoCap = cv2.VideoCapture('./video/cars2.mp4')
    frame_width = int(VideoCap.get(3))
    frame_height = int(VideoCap.get(4))
    fps = VideoCap.get(cv2.CAP_PROP_FPS)
    outVid = cv2.VideoWriter('trackedObjects.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                             (frame_width, frame_height))
    largestDistance = math.sqrt(frame_width**2+frame_height**2)
    # Variable used to control the speed of reading the video
    ControlSpeedVar = 100  # Lowest: 1 - Highest:100

    HiSpeed = 100

    # Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    # debugMode = 1
    debugMode = 0

    KFs = []
    KFs_used = []
    j = 0
    while(True):
        j += 1
        # Read frame
        ret, frame = VideoCap.read()
        if not ret and cv2.waitKey(2) & 0xFF == ord('q'):
            outVid.release()
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        # Detect object
        try:
            centers = detect(frame, debugMode)
        except:
            outVid.release()
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        # If centroids are detected then track them
        if (len(centers) > 0):
            # getting rid of unused Kalman Filters
            temp = KFs.copy()
            KFs = []
            for i in range(len(temp)):
                if(KFs_used[i]):
                    KFs.append(temp[i])
            gc.collect()
            # predict all KFs
            predicted_x, predicted_y = predictKFs(KFs)
            KFs_used = [False]*len(KFs)
            for i in range(len(centers)):
                # Draw the detected circle
                cv2.circle(frame, (int(centers[i][0]), int(
                    centers[i][1])), 10, (0, 191, 255), 2)

                curr_KF = getClosestKF(
                    centers[i][0], centers[i][1], predicted_x, predicted_y, largestDistance/24)
                if(curr_KF < 0):
                    KF = KalmanFilter(1.0/fps, 1, 1, 1, 0.1, 0.1)
                    x, y = KF.predict()
                    KFs.append(KF)
                    KFs_used.append(True)
                else:
                    KF = KFs[curr_KF]
                    KFs_used[curr_KF] = True
                    x, y = predicted_x[curr_KF], predicted_y[curr_KF]

                # Draw a rectangle as the predicted object position
                x, y = int(x), int(y)
                cv2.rectangle(frame, (x - 15, y - 15),
                              (x + 15, y + 15), (255, 0, ), 2)

                # Update
                (x1, y1) = KF.update(centers[i])
                x1, y1 = int(x1), int(y1)
                # Draw a rectangle as the estimated object position
                cv2.rectangle(frame, (x1 - 15, y1 - 15),
                              (x1 + 15, y1 + 15), (0, 0, 255), 2)

                cv2.putText(frame, "Estimated Position"+str(i),
                            (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Predicted Position"+str(i),
                            (x + 15, y), 0, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, "Measured Position"+str(i),
                            (int(centers[i][0] + 15), int(centers[i][1] - 15)), 0, 0.5, (0, 191, 255), 2)

        cv2.imshow('image', frame)
        outVid.write(frame)

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    # execute main
    main()
