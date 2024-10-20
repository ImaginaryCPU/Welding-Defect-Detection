import cv2
import supervision as sv
from ultralytics import YOLO

VIDEO_PATH = "VideoFolder/WeldingEx.mp4"
MODEL_PATH = "Model/best.pt"

video = cv2.VideoCapture(VIDEO_PATH)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

model = YOLO(MODEL_PATH)
if video.isOpened():

  while True:
    (grabbed, frame) = video.read()

    if not grabbed:#end of vid
      break

    # frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    frame = cv2.resize(frame, (640, 640))
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    

    labels = [f'Class: {model.names[int(cls)]} Conf: {conf:.2f}' for cls, conf in zip(detections.class_id, detections.confidence)]
    annonated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annonated_frame = label_annotator.annotate(scene=annonated_frame, detections=detections, labels = labels)

    cv2.imshow("Result Vid", annonated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


video.release()
cv2.destroyAllWindows()