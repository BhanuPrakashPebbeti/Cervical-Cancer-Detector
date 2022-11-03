import sys
import time
import cv2
from classifierLite import classification_module

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


def run(path,camera_id=0,width = 640, height = 480):
  classifier = classification_module(path)
  counter, fps = 0, 0
  start_time = time.time()
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi_start_time = time.time()
    image = classifier.apply_roi_cutting(image)
    roi_end_time = time.time()
    print("ROI CUTTING ",roi_end_time-roi_start_time)
    image = classifier.preprocess(image)
    pre_end_time = time.time()
    print("Preprocessing ",pre_end_time-roi_end_time)
    label,score = classifier.classify(image)
    classify_end_time = time.time()
    print("Classify ",classify_end_time-pre_end_time)
    score = round(score, 2)
    result_text = label + ' (' + str(score) + ')'
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Calculate the FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('image_classification', rgb_image)

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  run("models/model.tflite")