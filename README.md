# Projects on Deep Learning 
## Project 1 - Stock Price Prediction 
* The Dataset is taken from yahoo finance using yfinance library.
* Using the past data of the stock price, we will try to predict the price of the stock in the future. 
* We could totally wrong in predicting a stock's movement since, movement of a stock is independent of its movement in the past.


# Project 2 - Fashion MNIST (NN Classification)
* The Fashion MNIST dataset consists of 70,000 grayscale images of 28x28 pixels each, with 60,000 images for training and 10,000 images for testing.
* Class_names: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
### Building a mulit-class Classification Model

For our multi-class classification model, we can use a similar architecture to our binary classifiers, however, things:
* Input Shape = 28 * 28 (The shape of one image)
* Output Shape = 10 (One per class of clothing)
* Loss Function = tf.keras.losses.CategoricalCrossEntropy()
    * If your labels are one-hot encoded, use CategoricalCrossEntropy()
    * If your labels are integer form use SparseCategoricalCrossEntropy()
* Output layer activation = Softmax(not sigmoid)


# Project 3 - Food Images Classification (CNN)
### Building a CNN to find patterns in our images, more specifically we need a way to:

* Load our images
* Preprocess our images
* Data Augmentation 
* Build a CNN to find patterns in our images
* Compile our CNN
* Fit the CNN to our training data

 Adjust the model hyperparameters (to beat the baseline/ reduce overfitting)
Fixing overfitting by....
> **Getting more data** - Having more data gives a model more opportunity to learn diverse patterns...

> **Simplify the model** - If our current model is overfitting the data, it may be too complicated of a model, one way to simplify a model is to: reduce number of layers or reduce number of hidden units in layers

> **Use data augmentation** - data augmentation manipulates the training data in such a way to add more diversity to it (without altering the orginal data)

> **Use transfer learning** - Transfer Learning leverages the patterns another model has learned on similar data to your own and allows you to use those patterns on your own dataset. 
