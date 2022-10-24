# Human Skin Detection via Color Space Segmentation inPython

<img src="images/Banner--Skin-Detection.jpg" width = "500"/>


## 1. Objective

The objective of his project is to implement and demonstrate how to reliably detect human skin using color space segmentation.

## 2. Skin Detection

Skin detection is the process of finding skin-colored pixels and regions in an image or a video. This process is typically used as a preprocessing step to find regions that potentially have human faces and limbs in images.  The primary key for skin recognition from an image is the skin color. But color cannot be the only deciding factor due to the variation in skin tone according to different races. Other factors such as the light conditions also affect the results. Therefore, the skin tone is often combined with other cues like texture and edge features. This is achieved by breaking down the image into individual pixels and classifying them into skin colored and non-skin colored. One simple method is to check if each skin pixel falls into a defined color range or values in some coordinates of a color space.  The selection of the color range threshold values depends on several factors, including:

  * Illumination conditions
  * Individual characteristics such as age, sex and body parts.
  * Varying skin tone with respect to different races. 
  * Other factors such as background colors, shadows and motion blur.


There are many skin color spaces like RGB, HSV, YCbCr, YIQ, YUV, etc. that are used for skin color segmentation. In this project, we implement a threshold based on the combination of HSV and YCbCr color spaces and experiment with different color range threshold values.


## 3. Data

We shall use the following image, which has many people with different ethnicities and skin tones, under good illumination conditions. 

<img src="images/test-image-001.jpg" width = "1000"/>

## 4. Development

In this section, we shall walkthrough the development and illustration of the watershed segmentation algorithm using OpenCV built-in functionalities. 

  * Project: Human skin detection based on color segmentation: 
  * The objective of this project is to demonstrate how to detect human skin based on color space segmentation: 
  * This is a pixelwise process, which involves the following steps: 
    1. The RGB image value is converted to HSV color space: 
        * Potential skin pixels are detected based on a simple thresholding operation
        * HSV color space: 0<=H<=17 and 15<=S<=170 and 0<=V<=255 
    2.  The RGB image value is converted to YCrCb color space: 
        * Potential skin pixels are detected based on a simple thresholding operation
        * YCbCr color space: 0<=Y<=255 and 135<=Cr<=180 and 85<=Cb<=135 
    3. The HSV and YCbCr skin segmentations are merged together to obtain a more accurate skin segmentation. 

 We shall implement and illustrate each of these steps below.

* Author: Mohsen Ghazel (mghazel)
* Date: March 29th, 2021


### 4.1. Step 1: Imports and global variables:

#### 4.1.1. Python imports:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#======================================================</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#======================================================</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>image <span style="color:#200080; font-weight:bold; ">as</span> mpimg

<span style="color:#595979; "># input/output OS</span>
<span style="color:#200080; font-weight:bold; ">import</span> os 

<span style="color:#595979; "># date-time to show date and time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime

<span style="color:#595979; "># to display the figures in the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">4.5</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">1</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

#### 4.1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Global variales</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The RGB image value is converted to HSV color space</span>
<span style="color:#595979; "># Potential skin pixels are detected based on a simple </span>
<span style="color:#595979; "># thresholding operation:</span>
<span style="color:#595979; ">#</span>
<span style="color:#595979; "># HSV colorspace: 0&lt;=H&lt;=17 and 15&lt;=S&lt;=170 and 0&lt;=V&lt;=255</span>
<span style="color:#595979; ">#</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># H-channel thresholds:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The minimum H-value</span>
H_MIN <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#595979; "># The maximum H-value</span>
H_MAX <span style="color:#308080; ">=</span> <span style="color:#008c00; ">17</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># S-channel thresholds:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The minimum S-value</span>
S_MIN <span style="color:#308080; ">=</span> <span style="color:#008c00; ">15</span>
<span style="color:#595979; "># The maximum S-value</span>
S_MAX <span style="color:#308080; ">=</span> <span style="color:#008c00; ">170</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># V-channel thresholds:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The minimum V-value</span>
V_MIN <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#595979; "># The maximum V-value</span>
V_MAX <span style="color:#308080; ">=</span> <span style="color:#008c00; ">255</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The RGB image value is converted to YCrCb color space</span>
<span style="color:#595979; "># Potential skin pixels are detected based on a simple </span>
<span style="color:#595979; "># thresholding operation:</span>
<span style="color:#595979; ">#</span>
<span style="color:#595979; "># YCbCr colorspace: 0&lt;=Y&lt;=255 and 135&lt;=Cr&lt;=180 and 85&lt;=Cb&lt;=135</span>
<span style="color:#595979; ">#</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Y-channel thresholds:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The minimum Y-value</span>
Y_MIN <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#595979; "># The maximum Y-value</span>
Y_MAX <span style="color:#308080; ">=</span> <span style="color:#008c00; ">255</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># -channel thresholds:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The minimum Cr-value</span>
Cr_MIN <span style="color:#308080; ">=</span> <span style="color:#008c00; ">135</span>
<span style="color:#595979; "># The maximum Cr-value</span>
Cr_MAX <span style="color:#308080; ">=</span> <span style="color:#008c00; ">180</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Cb-channel thresholds:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># The minimum Cb-value</span>
Cb_MIN <span style="color:#308080; ">=</span> <span style="color:#008c00; ">85</span>
<span style="color:#595979; "># The maximum V-value</span>
Cb_MAX <span style="color:#308080; ">=</span> <span style="color:#008c00; ">135</span>
</pre>


### 4.2. Step 2: Input data

#### 4.2.1. Read and visualize the input template image

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Read the test image:</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># template test file name</span>
test_img_file_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../data/test-images/test-image-001.jpg"</span>
<span style="color:#595979; "># check if the teste image file exists</span>
<span style="color:#200080; font-weight:bold; ">if</span><span style="color:#308080; ">(</span>os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>exists<span style="color:#308080; ">(</span>test_img_file_path<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Test image file name DOES NOT EXIST! = '</span> <span style="color:#44aadd; ">+</span> template_img_file_path<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Read the test image </span>
img <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>imread<span style="color:#308080; ">(</span>test_img_file_path<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>IMREAD_COLOR<span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># axis off</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the template image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Test image'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the image</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span> 
</pre>


<img src="images/test-image-001.jpg" width = "1000"/>

### 4.3. Step 3: Detect Skin based on HSV Color space segmentation:

* The RGB image value is converted to HSV color space: 
    * Potential skin pixels are detected based on a simple thresholding operation
    * HSV color space: 0<=H<=17 and 15<=S<=170 and 0<=V<=255 


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 3.1) Convert the input image from BGR to HSV color </span>
<span style="color:#595979; ">#      space</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
img_HSV <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 3.2) Apply the thresholding operations on the HSV color </span>
<span style="color:#595979; ">#      channels as specified above</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
HSV_mask <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>inRange<span style="color:#308080; ">(</span>img_HSV<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>H_MIN<span style="color:#308080; ">,</span> S_MIN<span style="color:#308080; ">,</span> V_MIN<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>H_MAX<span style="color:#308080; ">,</span> S_MAX<span style="color:#308080; ">,</span> V_MAX<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> 

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 3.3) Apply morphological operations to remove small </span>
<span style="color:#595979; ">#      disconnected detections</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
HSV_mask <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>morphologyEx<span style="color:#308080; ">(</span>HSV_mask<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>MORPH_OPEN<span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>ones<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 3.4) Visualize the potential skin-segemntation based</span>
<span style="color:#595979; ">#      on the HSV color space</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># create a figure</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># visualize the original image</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">121</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Original image"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># The HSV skin detection</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">122</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"HSV color space: Skin detection"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>HSV_mask<span style="color:#308080; ">,</span>  cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">,</span> vmin<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> vmax<span style="color:#308080; ">=</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>

<img src="images/HSV-segmentation-results.PNG" width = "1000"/>


### 4.4. Step 4: Detect Skin based on YCrCb Color space segmentation:
* The RGB image value is converted to YCrCb color space: 
    * Potential skin pixels are detected based on a simple thresholding operation
    * YCbCr color space: 0<=Y<=255 and 135<=Cr<=180 and 85<=Cb<=135 
    

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.1) Convert the input image from BGR to YCrCb color </span>
<span style="color:#595979; ">#      space</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
img_YCrCb <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2YCrCb<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.2) Apply the thresholding operations on the YCrCb color </span>
<span style="color:#595979; ">#      channels as specified above</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
YCrCb_mask <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>inRange<span style="color:#308080; ">(</span>img_YCrCb<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>Y_MIN<span style="color:#308080; ">,</span> Cr_MIN<span style="color:#308080; ">,</span> Cb_MIN<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>Y_MAX<span style="color:#308080; ">,</span> Cr_MAX<span style="color:#308080; ">,</span> Cb_MAX<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> 

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.3) Apply morphological operations to remove small </span>
<span style="color:#595979; ">#      disconnected detections</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
YCrCb_mask <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>morphologyEx<span style="color:#308080; ">(</span>YCrCb_mask<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>MORPH_OPEN<span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>ones<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.4) Visualize the potential skin-segemntation based</span>
<span style="color:#595979; ">#      on the YCrCb color space</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># create a figure</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># visualize the original image</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">121</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Original image"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># The HSV skin detection</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">122</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"YCrCb color space: Skin detection"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>YCrCb_mask<span style="color:#308080; ">,</span>  cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">,</span> vmin<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> vmax<span style="color:#308080; ">=</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>

<img src="images/YCrCb-segmentation-results.PNG" width = "1000"/>



#### 4.5. Step 5: Combine the HSV and YCrCb Color space segmentations:

* Combine the HSV and YCrCb skin segmentation masks to obtain a final merged skin segmentation mask:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 5.1) Merge the HSV and YCrCb skin masks together:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
merged_mask<span style="color:#308080; ">=</span>cv2<span style="color:#308080; ">.</span>bitwise_and<span style="color:#308080; ">(</span>YCrCb_mask<span style="color:#308080; ">,</span>HSV_mask<span style="color:#308080; ">)</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 5.2) Post-process the moerged mask to reduce noise</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
merged_mask<span style="color:#308080; ">=</span>cv2<span style="color:#308080; ">.</span>medianBlur<span style="color:#308080; ">(</span>merged_mask<span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 5.3) Apply morphological operations to remove small </span>
<span style="color:#595979; ">#      disconnected detections</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
merged_mask <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>morphologyEx<span style="color:#308080; ">(</span>merged_mask<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>MORPH_OPEN<span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>ones<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">4</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">4</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 5.4) Visualize the final merged skin detection results</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># create a figure</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># visualize the original image</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">121</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Original image"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># The HSV skin detection</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">122</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Merged HSV and YCrCb color spaces: Skin detection"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>merged_mask<span style="color:#308080; ">,</span>  cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">,</span> vmin<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> vmax<span style="color:#308080; ">=</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>


<img src="images/HSV+YCrCb-segmentation-results.PNG" width = "1000"/>


#### 4.6. Step 6: Display a successful execution message


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">14</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">32</span><span style="color:#308080; ">:</span><span style="color:#008000; ">54.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In view of the final image segmentation results, we make the following observations:
    * The HSV color space appears to be more sensitive to skin tone than the YCrCb color space
    * In some cases the HSV color space seems to under-segment.
    * The YCrCb color space yields consistently good skin segmentation results for the different ethnicities and skin tones
    * In some cases the YCrCb color space seems to over-segment.
    * The combined skin segmentation results appear to be negatively impacted by the HSV segmentation for darker skin tones
    * The YCrCb may be the preferred skin segmentation method as it yields consistently good skin detection results for the different ethnicities and skin tones.


## 6. Future Work

* We proposed to explore the following related issues:
  * To further explore the YCrCb color space skin segmentation
  * To experiment with different combinations of color range thresholds
  * To explore the sensitivity of the YCrCb color space skin segmentation to different factors, including:
    * Illumination conditions
    * Individual characteristics such as age, sex and body parts.
    * Varying skin tone with respect to different races. 
    * Other factors such as background colors, shadows and motion blur.

## 7. References

1. Adrian Rosebrock. Skin Detection: A Step-by-Step Example using Python and OpenCV. https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/ 
 2. S. Kolkur, et al. Human Skin Detection Using RGB, HSV and YCbCr Color Models. https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf 
 3. Nalin Chhibber. Skin Detection Using OpenCV Python. https://nalinc.github.io/blog/2018/skin-detection-python-opencv/ 
 4. Kseniia Nikolskaia, et al. Skin Detection Technique Based on HSV Color Model and SLIC Segmentation Method. http://ceur-ws.org/Vol-2281/paper-13.pdf 
 5. Rahul Singh. Skin Detection Using OpenCV in Python. https://www.codespeedy.com/skin-detection-using-opencv-in-python/ 
 6. Linda.com. Skin detection. https://www.lynda.com/Python-tutorials/Skin-detection/601786/660485-4.html
 
 





