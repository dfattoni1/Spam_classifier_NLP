# Spam Classifier (NLP)

**Using a Naive Bayes Classifier for Natural Language Processing in order to determine whether emails are spam.**

## Project overview
Email communication remains a primary vector for digital productivity, yet it is constantly threatened by spam. This project focuses on building a Spam Filter using a Multinomial Naive Bayes Classifier. By analyzing word frequencies and patterns, the model identifies the probability that an incoming message is spam.

## The data
The data for this Project was obtained through a publicly available Kaggle dataset on spam vs ham emails. The dataset can be accessed [here](https://www.kaggle.com/datasets/suvidyasonawane/spam-vs-ham-emails). All the data is contained within a single file, email_spam_dataset.csv, which can be found in the data folder of this project. The dataset included 2 columns:

| Feature | Description | Data type |
|:--- | :--- | :--- |
| email_text | Contents of the email | String |
| label | Label indicating whether the email is "spam" or "ham" | String |

## Development of the project
There were two important sections to this project, the building of the model and the evaluation of the model. Both Will be explained in this section

### Building the model
Since the main objective with this dataset was to classify whether emails were spam or not, it was necessary to make use of NLP, so a Naive Bayes Classifier was implemented. The model was built and tuned for the n-grams and level of alpha that could yield the highest F1-Score. The different n-grams and values of alpha were tested using a grid search with cross validation. Also, since it was necessary to calculate the number of occurrences of each Word in the training set for each time the test fold changed, a pipeline was set up to manage the counting of words and fitting of the model. The best parameters were found to be unigrams and no smoothing (alpha = 0), meaning the training data was fairly simple.

### Evaluating the model
The model found by the grid search yielded an F1-Score of 1, meaning perfect recall and perfect precision. This was true for the test data and the validation data not included in the grid search. This clearly sparked doubts about the model's validity, as such a high score could have indicated some sort of data leakage along the process or a different type of mistake.

Examining the results of the grid search, it became apparent that the model was obtaining this perfect score for all orders of n-grams and all values of alpha, even for the accuracy. Given that the dataset only contained 320 rows (extremely limited) and the messages in each email were not particularly long, it was possible that the dataset was simply perfectly separable, which would be more likely given the low volume of data. However, in order to make sure this was the case and not an error with the model, a data frame was created in which it was possible to visualize each unique word in the dataset alongside the log of the probability of it belonging to the "spam" or "ham" categories.

After careful examination, it became apparent that there was practically no overlap between the words that made up the spam emails and the rest. Besides a few generic words such as "to", "for", "are", etc., most of the words had a probability of practically zero of belonging to either one or the other category. An additional column containing the difference in the log of probabilities was included to see how much likely one word is to appear in one case or the other, and the values were infinity or negative infinity in all but those few overlapping words.

### Final considerations
Given the findings of the analysis, it became clear that the model was possibly obtaining perfect results given the reduced and perfectly separable nature of the dataset at hand. This is also probably the reason why it was possible to achieve this even while only using unigrams and no smoothing whatsoever. It would be possible to assume that the model in this project is misleadingly accurate. While it performs perfectly with this dataset, being trained on such a reduced vocabulary (only 71 unique words in total) and with no overlap between categories suggests that it would most likely not perform as well when exposed to more complex data.

The full analysis, including the evaluation of the model, can be found [here](spam_classifier.ipynb).

## How to explore this repository
**Analysis:** View the full the analysis in the [Jupyter Notebook](spam_classifier.ipynb) for full EDA and modelling.

**Reproduction:** Run `pip install -r requirements.txt` to set up the environment.