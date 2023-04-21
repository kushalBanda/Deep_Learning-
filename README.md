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


