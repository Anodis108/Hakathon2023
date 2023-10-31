from ultralytics import YOLO
import cv2 
import supervision as sv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def trackkk():
    model = YOLO('yolov8x.pt')

    file_path = "result.txt"
    file = open(file_path, "a")

    no_frame = 1

    for result in model.track(source="Video_Phase1\Doto_103.mp4",stream=True, conf=0.3, iou=0.7, show=True, classes = [0]):
        
        idbox = result.boxes.id.cpu().numpy().astype(int)
        bbox = result.boxes.xyxy.cpu().numpy().astype(int)

        print(idbox.size)
        for i in range(0, idbox.size):
            file.write("{},{},{},{},{},{},{},{}\n".format(no_frame ,i ,bbox[i][0],bbox[i][1],bbox[i][2] - bbox[i][0],bbox[i][3]-bbox[i][1],0, 0)) #
        no_frame += 1
        
if __name__ == "__main__":
    
    trackkk()