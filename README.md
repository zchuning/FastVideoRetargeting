# Fast Video Retargeting Based on Seam Carving with Parental Labeling
Implementation of the fast video retargeting method proposed in [this](https://arxiv.org/abs/1903.03180) paper. [OpenCV 3](https://opencv.org/) and [NumPY](https://www.numpy.org/) are required to run these code.

**SeamCarverPL.py** contains the sinlge-image seam carving function **seamCarverVertical(img, nwidth)**. 

  * **img** is a numpy matrix encoding the cource image, which can be created via cv2.imread(). 
  * **nwidth** is the target width after carving.

**VideoRetargetingPL.py** contains the video retargeting function **videoRetarget(name, outn, a)**.  

  * **name** is the path of the source video.
  * **outn** is the path of the output video.
  * **a** is the compression factor, namely the width of the output video to the width of the source video.
