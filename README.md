# LSTM code clone dection

This Python script is a simple implementation of a LSTM (Long Short-Term Memory) model for sentence similarity detection. The model is trained on pairs of sentences and learns to predict whether two sentences are similar or not.

We currently do not have any article to prove its accuracy, but its accuracy is quite high.

## How it works

1. **Data Preparation**: The `prepare_sequence` function is used to convert a sentence into a tensor of word indices based on a given word dictionary.

2. **Model Definition**: The `MyModule` class defines the LSTM model. It includes an embedding layer for word representation, a LSTM layer for sequence processing, and a softmax layer for outputting similarity scores.

3. **Training**: The `train` function is used to train the model. It iterates over the training data, computes the loss between the predicted and actual labels, and updates the model parameters. The loss function used is `HingeEmbeddingLoss`, and the optimizer is `SGD`. The model's state is saved after each epoch.

4. **Evaluation**: The `evaluate` function is used to evaluate the model's performance on test data. It computes the pairwise distance between the output vectors of two sentences, and predicts that the sentences are similar if the distance is less than 0.5. The accuracy of the model is then calculated based on the number of correct predictions.

## Usage

To use this script, you need to have a dataset of sentence pairs and their corresponding labels. The labels should be 1 for similar sentences and -1 for dissimilar sentences. You also need to have a word dictionary that maps each word in your dataset to a unique index.

You can run the script using the following command:

```bash
python Fake_CDLH_Main.py
```

By default, the script will load the data, train the model, and evaluate its performance. You can comment out the `train` or `evaluate` function calls in the `if __name__ == '__main__':` block to skip training or evaluation.

## Author's idle talk

We comeplete it while doing our student project of Harbing Institution of Technology.

In our origin project, we wish to implement an CDLH base code clone dection AI model according to [this paper](https://dl.acm.org/doi/10.5555/3172077.3172312)

We wrote this semi finished model, and surprisingly found it gets 79% accuracy.

However, we have to refactor our code for the sake of our wrong implement.

In fact, this model runs faster, but only loses a small amount of accuracy. So we reserve this code, and separate it into a new repo.

## Notice

+ We used some code in [this repo](https://github.com/milkfan/CDLHDetector), thanks for his work.Unfortunately, despite the name of this repo, it is not really a CDLH.like the version I first uploaded.
+ We used OJ Clone data set.