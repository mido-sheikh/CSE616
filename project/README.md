# Traffic Sign Classification
**In this project, I used Python and TensorFlow to classify traffic signs.**

**Dataset used: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
This dataset has more than 50,000 images of 43 classes.**
Download the dataset from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). 
This is a pickled dataset in which we've already resized the images to 32x32.

## Architecture:
- **plot.py**
	We use matplotlib plot sample images from each subset.
	and numpy to plot a histogram of the count of images in each unique class.
- **preprocess.py**.
	We apply several preprocessing steps to the input images to achieve the best possible results
	using the following preprocessing techniques:
    - Shuffling.
    - Grayscaling.
    - Local Histogram Equalization.
    - Normalization.
- **VGGnet.py**
	We design and implement a deep learning model that learns to recognize traffic signs from our dataset
	Using VGGNet, we've been able to reach a very high accuracy rate. We can observe that the models saturate after nearly 10 epochs, so we can save some computational resources and reduce the number of epochs to 10.
	We can also try other preprocessing techniques to further improve the model's accuracy..
- **main.py**
	Responsible for starting the training and testing