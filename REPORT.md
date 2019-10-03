## Goals
1. Predict what star-rating (out of 5) Yelp users gave to restaurants from the review text.
2. Analyze the performance of different approaches as well as a novel method.

## Approach
We decided to follow an incremental approach in order to measure success definitively. We started by first establishing a baseline using a simple bag of words and linear model method. The poor performance of this model acted as a sanity check for our following endeavors. Later, we experimented with preprocessing approaches. Our preprocessing included fixing common typos, removing the HTML artifacts present in the dataset, replacing emojis with the closest English phrase. Our preprocessing approaches increased the performance of our baseline model, so we decided to keep them for future models as well.

Next, we began experimenting with different models. We started using the GLoVe pretrained vectors to enrich our word representations. We first experimented with two models, a CNN model that reported good performance in a similar paper and a LSTM model that we knew from our studies is well adapted for the task. Both of them achieved respectable results, in the ballpark of what previous papers had achieved. Next up, we decided to combine the two approaches, forcing the model to simultaneously learn via its convolutional and recurrent parts. Since the dataset we were using consisted of around 2 million reviews, we did all our training and dataset preparation using the powerful Tensorflow library. This allowed us to think more about the important decisions regarding the data science behind the task rather than the implementation.

## Outcome
We beat both the benchmarks we created for ourselves as well as any prior research found on the task. Moreover, I gained confidence working with datasets larger than the amount of RAM available by leveraging and becoming very accustomed to the Tensorflow machine learning library.

Please have a look at our [Project report](http://bit.ly/yelp-nlu-report).