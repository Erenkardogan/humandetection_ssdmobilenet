import cv2
import numpy as np

def goruntuisleme():
    global humans
    i = 0
    thres = 0.45  # Threshold to detect object
    nms_threshold = 0.2
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 150)
    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        # print(indices)
        # print("bboxlar:",bbox)

        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            cam_height = 720
            cam_width = 1280

            cv2.rectangle(img, (x, y), (x + w, h + y), color=(255, 0, 0), thickness=2)
            cv2.putText(img,"person:"+str(i), (box[0] + 20, box[1] + 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        humans = i + 1
        print("People: ", humans)
        cv2.imshow("Output", img)
        # cv2.imwrite("output.jpg", img)
        cv2.waitKey(1)
        """if humans > 2:    # if you want to close the code when the number reaches a certain level you can use this
            break"""



if __name__ == "__main__":
    goruntuisleme()
    print("Number of People: ", humans)  # Number of people

