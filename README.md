# Shadow Neato

Robot behavior to find and hide in closest available shadow.

![shadow gazebo gif](media/shadow_gazebo.gif)

Neato Camera View (click on image to view video):
[![Neato camera view screenshot](media/video_screenshot.png)](https://youtu.be/U_qABB3b_0g)

## Requirements

* PyTorch
* NumPy
* OpenCV
* PIL
* [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf)
* [MTMT Shadow Detection Model](https://github.com/eraserNut/MTMT)

## Usage

```sh
# Start roscore
roscore

# Launch gazebo with camera
roslaunch neato_gazebo neato_gauntlet_world.launch load_camera:=true

# Start shadow neato behavior
roslaunch computer_vision shadow_neato.launch

# Optional: move with teleop
rosrun teleop_twist_keyboard teleop_twist_keyboard.py  
```

## Project Goal

The project's goal was to create a robotic behvior for the neato so that it can find and hide in nearby shadows using computer vision. The use case for this is that for robots that are performing search and rescue, they can detect shade and move injured patients into them and away from the sun. After building out this functionality, it is quite easy to now create the oppoiste behavior, avoiding shadows or chasing after sunlight. This could be useful for robots carrying plants for example.

Through this project I wanted to get a better understanding of OpenCV image processing, using a ML model in practice, and building a ROS project from scratch.

## Methodology

The core of the shadow behavior comes from the shadow detection and image processing done to select the shadow to move to. To accomplish this, I created two ROS nodes that communicate back and forth to each other. The first node is in `shadow_approacher.py` and handles interfacing with the Neato robot. This means retrieving the raw images from the camera and sending the motor information to the Neato, and other controller logic. The other node is in `shadow_mask.py` and handles performing inference on the raw camera images using the shadow detection model, creating the shadow masks and then publishing these masks onto a ros topic. This was done because performing inference using CPU takes a long time and doing all of this in one node would cause the rest of the operations to hang waiting on creating the shadow masks. Therefore two seperate nodes were created as ROS's pusblisher/subscriber model lends itself well to asynchronous operations. The following block diagram gives a general overview of the system:

![block diagram](media/diagram.jpeg)

The first step in choosing nearby shadows to hide into is detecting and isolating each shadow in view. To do this, a "shadow mask" is created where the source image is turned into a binary image where white areas are shadows and black areas are the rest of the image. This is done using the raw camera images received from the `video_frame` topic and running it through the MTMT Shadow Detection Model described [here](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Multi-Task_Mean_Teacher_for_Semi-Supervised_Shadow_Detection_CVPR_2020_paper.pdf). With the mask created, back in `shadow_approacher.py` further image processing is done to remove noise that may have came about from areas that kinda looked like shadows. To do this, I use morphological transformations and thresholding with OpenCV. Now with a clean binary image with potential shadows isolated, I use OpenCV's contour approximation to extract the coordinates of the perimeter surrounding each shadow. These contours can be seen below outlined in green.

![camera view](media/camera_view.png)
![shadow mask](media/shadow_mask.png)


To build out this behavior, I created 4 states using boolean flags:

1. Default
2. Seeking
3. Within Shadow
4. Hunting

### Specific Design Decisions

Describe a design decision
* only get shadows below the horizon, helping push, slow movement to let inference catch up

## Reflection

What challenges did you face along the way
* Inference w/ cpu, model not working right, opencv ros bug

What would you do to improve the project if you had more time
* Move project to a machine that can use gpu acceleartion

Did you learn any interesting lessons for future robotics programmin projects?
