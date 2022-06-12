# Automatic Number Plate Recognition

This project aims to detect licence plates from vehicles and extract the text and number from detected licence plate. The model is trained using YOLOv5 and for extraction of text from images, Tessaract OCR is used.

## Creating the Dataset

Dataset was downloaded from Kaggle and Roboflow. The dataset is partitioned into train, validation and test sets containing 75%, 15%, and 10% of the data respectively. Annotations are in YOLOv5 PyTorch format.

## Training Options

The model is trained using YOLOv5. We use various flags to set options regarding training :
  
 1. img : Size of image. The image is a square one. The original image is resized while maintaining the aspect ratio. 
 2. batch : The batch size. 
 3. epochs : Number of epochs to train for. 
 4. data : Data YAML file that contains information about the dataset (path of images, labels)
 5. workers : Number of CPU workers.
 6. cfg : Model architecture. There are 4 choices available: yolo5s.yaml, yolov5m.yaml, yolov5l.yaml, yolov5x.yaml. The size and complexity of these models increases in the ascending order and we can choose a model which suits the complexity of our object detection task. 
 7. weights : Pre-trained weights we want to start training from. 
 8. name : Various things about training such as train logs. 
 9. hyp : YAML file that describes hyperparameter choices. 
 

## Data Config File

Custom data config file(lpr.yaml) is used. Details for the dataset we want to train our model on are defined by the data config YAML file. The following parameters have to be defined in a data config file :
  
1. train, test, and val : Locations of train, test, and validation images.
2. nc : Number of classes in the dataset. (nc = 1 in our case)
3. names : Names of the classes in the dataset. The index of the classes in this list would be used as an identifier for the class names in the code.


## Hyperparameter Config File

Default hyperparameter config file, `hyp.scratch-low.yaml` is used. The hyperparameter config file helps us define the hyperparameters for our neural network.

## Custom Network Architecture

YOLO v5 allows us to define our own custom architecture and anchors if one of the pre-defined networks doesn't fit the bill. For this we will have to define a custom weights config file. For this model, I have used the `yolov5s.yaml`.

## Train the Model

The location of train, val and test images and labels, the number of classes(nc), and the names of the classes are defined. Since the dataset is small, and there are not many objects per image, the smallest of pre-trained models, `yolo5s` is used to keep things simple and avoid overfitting. Batch size is `16` and image size is `416` and the training is done for `5000` epochs.

The command to train the model is : <br />
 `python train.py --img 416 --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch 16 --epochs 5000 --data lpr.yaml --weights yolov5s.pt --workers 24 --name lpr_detection_new`
 
 ## Evaluating the Model
 
 After training, we need to evaluate the training losses and performance metrics for YOLOv5. For this evaluation, I have used tensorboard.
 
 ### mAP@0.5

![mAP_0 5](https://user-images.githubusercontent.com/37297441/173238951-0912485a-941a-47d5-aaab-da0a3d3811dc.PNG)

### mAP@[0.5:0.95]

![mAP_0 95](https://user-images.githubusercontent.com/37297441/173239050-14f8cce3-77f7-42de-9629-a38cc6f46469.PNG)

### Precision

![precision](https://user-images.githubusercontent.com/37297441/173239084-dca3b81f-1398-47ed-82c7-b1bd7e3e51c8.PNG)

### Recall

![recall](https://user-images.githubusercontent.com/37297441/173239092-db075b58-d04e-4ae0-8c4e-c140c1cf671e.PNG)

### Train Loss

![box_loss_train](https://user-images.githubusercontent.com/37297441/173239140-507aca29-631e-449b-86af-44c26ea66c8e.PNG)

![obj_loss_train](https://user-images.githubusercontent.com/37297441/173239143-5573ab29-86ea-4198-9e1b-68771761e28e.PNG)


### Validation Loss

![box_loss_val](https://user-images.githubusercontent.com/37297441/173239157-819c09f8-2f16-481b-8fd2-ab9ccd4b6451.PNG)

![obj_loss_val](https://user-images.githubusercontent.com/37297441/173239161-3f01c8e5-8290-4fe3-bc5c-675603127a45.PNG)

### Confusion Matrix

The dataset which I used for training has one class label i.e licence. So to know how well my model detects the class label, a confusion matrix will be useful.

![confusion_matrix](https://user-images.githubusercontent.com/37297441/173239174-85cfdb0e-021d-450d-b1c4-90e88ac48da5.png)

The trained model detects licence 88% correctly as licence.

The below graph shows all metric plot in one figure : 

![results](https://user-images.githubusercontent.com/37297441/173239302-287d4fca-376b-4ad6-a5f7-ef2a25d25fad.png)

## Inference

There are many ways to run inference using the `detect.py` file. 

1. The `source` flag defines the source of our detector, which can be : </br>
A single image </br>
A folder of images </br>
Video</br>
Webcam</br>

2. The `weights` flag defines the path of the model which we want to run our detector with. `best.pt` contains the best-performing weights saved during training.
3. `conf` flag is the thresholding objectness confidence.
4. `name` flag defines where the detections are stored. 

## Region of Interest

We need only the bounding box which contains the license plate so that we can input only that license plate to OCR for extracting text from images. This is called the region of interest(ROI). We need to extract ROI from the image inside the bounding box of the license plate automatically.<br/>

`detect.py` file of YOLOv5 is modified to automatically crop the license plate whenever it detects the object.<br/>

Following code is added to `detect.py` : <br/>

```
  if save_obj:
                        if int(cls) == 0:
                            for k in range(len(det)):
                                x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]),   int(xyxy[3] - xyxy[1])
                                img_ = im0.astype(np.uint8)
                                crop_img = img_[y:y+h, x:x+w]
                                filename=p.name
                                filepath=os.path.join('D://2022_ML_DL//yolov5-master//datasets//lpr//images//test//result//', filename)
                                print(filepath)
                                cv2.imwrite(filepath, crop_img)
                        else:
                            print("No object detected!")
                            continue
  
```

The command to run inference on test images is : <br/>

`python detect.py --weights D:\2022_ML_DL\yolov5-master\runs\train\lpr_detection_new2\weights --img 416 --conf 0.8 --source D:\2022_ML_DL\yolov5-master\datasets\lpr\images\test `

## Inference Results

![b1a50a3824887ee2_jpg rf 68a4fd34fce20184287592f2680f895b](https://user-images.githubusercontent.com/37297441/173239994-f921b86c-9107-474d-bacd-8d07b89f781d.jpg)
![b6ecda23586a6ba5_jpg rf d737139968dd3f08447305aa7b7f6002](https://user-images.githubusercontent.com/37297441/173240005-d1ba2a1e-9323-4926-9f88-4aed37f8bcbe.jpg)

![b193070a9c45b5ab_jpg rf 57e5987eb896a7bf9fc7a1a96a660c7e](https://user-images.githubusercontent.com/37297441/173240029-0f2658a4-f03b-4251-95b6-613ce15f5522.jpg)

## ROI Results

Following are the images after cropping licence plates from detected objects.<br/>

![Cars261_png rf 97efb9454753ec7417329cb6abd13df6](https://user-images.githubusercontent.com/37297441/173240165-99fc90e4-bdbe-470d-bcca-05e271c59900.jpg)

![Cars381_png rf 2a3141d9aa457bb1408419d183fac0a3](https://user-images.githubusercontent.com/37297441/173240178-6f0b00ce-3aa5-47a1-aab5-dd1e6d991025.jpg)

## Optical Character Recognition(OCR)

For the OCR we will be using Pytessaract. `tesseract-ocr.ipynb` is executed to extract text from cropped licence plates.

## Future Works

Extracting text from an image is a challenging task because character recognition from an image is always difficult for a model due to noise, color, variance, skewness in an image. There is still a lot that needs to improve on image processing in the future. 






















