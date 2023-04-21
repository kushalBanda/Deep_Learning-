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


# Project 4 - Diaster Text Classification (NLP)
In NLP, there are two main concepts for turning text into numbers:

* Tokenization - A straight mapping from word or character or sub-word to a numerical value. There are three main levels of tokenization:

1. Using word-level tokenization with the sentence "I love TensorFlow" might result in "I" being 0, "love" being 1 and "TensorFlow" being 2. In this case, every word in a sequence considered a single token.

2. Character-level tokenization, such as converting the letters A-Z to values 1-26. In this case, every character in a sequence considered a single token.

3. Sub-word tokenization is in between word-level and character-level tokenization. It involves breaking individual words into smaller parts and then converting those smaller parts into numbers. For example, "my favorite food is pineapple pizza" might become "my, fav, avour, rite, fo, oo, od, is, pin, ine, app, le, piz, za". After doing this, these sub-words would then be mapped to a numerical value. In this case, every word could be considered multiple tokens.

* Embeddings - An embedding is a representation of natural language which can be learned. Representation comes in the form of a feature vector. For example, the word "dance" could be represented by the 5-dimensional vector [-0.8547, 0.4559, -0.3332, 0.9877, 0.1112]. It's important to note here, the size of the feature vector is tuneable. There are two ways to use embeddings:

1. Create your own embedding - Once your text has been turned into numbers (required for an embedding), you can put them through an embedding layer (such as tf.keras.layers.Embedding) and an embedding representation will be learned during model training.

2. Reuse a pre-learned embedding - Many pre-trained embeddings exist online. These pre-trained embeddings have often been learned on large corpuses of text (such as all of Wikipedia) and thus have a good underlying representation of natural language. You can use a pre-trained embedding to initialize your model and fine-tune it to your own specific task.

### Experiments
* Model 0: Naive Bayes with TF-IDF encoder (baseline)
* Model 1: Feed-forward neural network (dense model)
* Model 2: LSTM (RNN)
* Model 3: GRU (RNN)
* Model 4: Bidirectional-LSTM (RNN)
* Model 5: 1D Convolutional Neural Network
* Model 6: TensorFlow Hub Pretrained Feature Extractor
* Model 7: TensorFlow Hub Pretrained Feature Extractor
(10% of data) 

Approach to all of these models
USe the standard steps in modelling with tensorflow
* Create a model
* Build a model
* Fit a model
* Evaluate our model
