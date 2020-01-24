# Image Classification

 The first challenge is to solve a classification problem on the proposed dataset, 
 which contains images of several objects belonging to 20 different classes. 
 Being a classification problem, given an image, the goal is to predict the correct class label. 
 
 ## Data Details
 
* **Image size**: variable
* **Color space**: RGB/Grayscale (read as 'rgb' in ImageDataGenerator.flow_from_directory ('color_mode' attribute) or use PIL.Image.open('imgname.jpg').convert('RGB'))
* **File Format**: JPG
* **Number of classes**: 20 (Classes 0-19)
* **Classes**:
  * 'owl' : 0
  * 'galaxy' : 1
  * 'lightning' : 2
  * 'wine-bottle' : 3
  * 't-shirt' : 4
  * 'waterfall' : 5
  * 'sword' : 6
  * 'school-bus' : 7
  * 'calculator' : 8
  * 'sheet-music' : 9
  * 'airplanes' : 10
  * 'lightbulb' : 11
  * 'skyscraper' : 12
  * 'mountain-bike' : 13
  * 'fireworks' : 14
  * 'computer-monitor' : 15
  * 'bear' : 16
  * 'grand-piano' : 17
  * 'kangaroo' : 18
  * 'laptop' : 19

## Dataset Structure
* **Two folders**:
  * training: 1554 images
  * test: 500 images
* **Images per class**:
  * school-bus : 73
  * laptop : 100
  * t-shirt : 100
  * grand-piano : 70
  * waterfall : 70
  * galaxy : 56
  * mountain-bike : 57
  * sword : 77
  * wine-bottle : 76
  * owl : 95
  * fireworks : 75
  * calculator : 75
  * sheet-music : 59
  * lightbulb : 67
  * bear : 77
  * computer-monitor : 100
  * airplanes : 100
  * skyscraper : 70
  * lightning : 100
  * kangaroo : 57
