import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import keras
import time
from pynput.keyboard import Key, Controller

# webcam settings
video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# holds the cnn and prediction code
class GesturePredictor():

    def __init__(self):
        # webcam stream
        self.cap = cv2.VideoCapture(video_id)
        self.WINDOW_WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.WINDOW_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # aruco marker detection
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        # prediction 
        self.model = keras.models.load_model("gesture_recognition_media_paper")        
        self.CONDITIONS = ['like', 'no_gesture', 'stop', 'dislike', 'paper']
        self.IMG_SIZE = 64
        self.SIZE = (self.IMG_SIZE, self.IMG_SIZE)
        self.COLOR_CHANNELS = 3
        self.prediction_array = []
        self.predicted_index = 1 # no_gesture
        self.gesture_predicted = False

    # from GRIPS - selects inner corners from aruco marker 
    def _get_corner_points(self, corners, ids):
        top_left = corners[list(ids).index(1)][0][3] #0 2
        bottom_left = corners[list(ids).index(2)][0][0] # 1 3
        bottom_right = corners[list(ids).index(3)][0][1] #2 0
        top_right = corners[list(ids).index(0)][0][2] # 3 1

        return np.float32([top_left, top_right, bottom_left, bottom_right])
    
    # from GRIPS - transforms selected aruco marker paper
    def _get_warped_image(self, corner_points, frame):
        x0 = 0
        y0 = 0
        x1 = self.WINDOW_WIDTH
        y1 = self.WINDOW_HEIGHT
        FLIP_IMAGE = False

        if FLIP_IMAGE:
            pts2 = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        else:
            pts2 = np.float32([[x1, y1], [x0, y1], [x1, y0], [y0, y0]])

        matrix = cv2.getPerspectiveTransform(corner_points, pts2)
        warped_image = cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

        return warped_image
    
    # reshapes image for prediction model
    def _get_reshaped_image(self, frame, warped_image):
        if self.COLOR_CHANNELS == 1:
            warped_image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

        resized = cv2.resize(warped_image, self.SIZE)
        reshaped = resized.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, self.COLOR_CHANNELS)

        return reshaped


    # predicts current webcam image
    def predict_image(self):
        ret, frame = self.cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        cv2.imshow('frame', frame)

        if ids is not None:
            # only proceed if four markers are detected
            if len(ids) == 4:
                # find inner corner of each marker and apply perspective transformation
                pts1 = self._get_corner_points(corners, ids)
                warped_image = self._get_warped_image(pts1, frame)

                # Display the frame
                cv2.imshow('frame', warped_image)

                # reshape frame for cnn
                reshaped_image = self._get_reshaped_image(frame, warped_image)
                
                # predict current frame
                prediction = self.model.predict(reshaped_image)

                # get predicted index (for label)
                self.predicted_index = np.argmax(prediction)
                #self.predicted_index = pred_label_index

                #print(self.CONDITIONS[pred_label_index], np.max(prediction))

        else:
            # if no aruco markers are detected just show webcam image
            cv2.imshow('frame', frame)

# holds the media control code
class MediaController():

    def __init__(self) -> None:
        self.keyboard = Controller()
        self.stop_timeout_start = False
        self.stop_timeout = 1.5 # seconds
        self.stop_time_counter = 0 # counts up to stop_timout

    # media control keys are pressed to specific predictions
    def _control_media(self, predicted_index):
        if predicted_index == 0: # like
            self.keyboard.press(Key.media_volume_up)
            self.keyboard.release(Key.media_volume_up)
        elif predicted_index == 2: #stop  
            # I found the prediction of stop a little bit too fast
            # because it was switching between pause and play too often.
            # That's why I implemented this timout
            # if stop is predicted, the control keys are only triggered again after 1.5 seconds
            if not self.stop_timeout_start:  
                self.keyboard.press(Key.media_play_pause)
                self.keyboard.release(Key.media_play_pause)
                self.stop_timeout_start = True
            elif self.stop_timeout_start and self.stop_time_counter >= self.stop_timeout:
                self.stop_timeout_start = False
                self.stop_time_counter = 0
                return
            elif self.stop_timeout_start and self.stop_time_counter <= self.stop_timeout:
                self.stop_time_counter += 0.1
        elif predicted_index == 3: # dislike
            self.keyboard.press(Key.media_volume_down)
            self.keyboard.release(Key.media_volume_down)


if __name__ == "__main__":
    print("starting application...")
    predictor = GesturePredictor()
    controller = MediaController()
    
    while True:
        predictor.predict_image()
        controller._control_media(predictor.predicted_index)

        time.sleep(0.1)

        # Wait for a key press and check if it's the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    predictor.cap.release()
    cv2.destroyAllWindows()
