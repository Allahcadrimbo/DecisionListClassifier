# DecisionListClassifier
Implements a decision list classifier that performs sentiment analysis. This project is broken up into three sections (Train, Test, and Evaluate). Pang-Lee-PA3.zip contains data that can be used to train the decision list. The data is from Bo Pang and Lillian Lee.

# decision-list-train.py

### Description:
In this PA we are attempting to implement a decision list classifier that will perform sentiment analysis. A decision
list is a list of rules that are learned from some training set that will be used on a training set to make some
classification. In our case we are try to perform sentiment analysis on movie reviews. When we are analysing sentiment
in movie reviews we are trying to determine if the review is positive or negative. The decision list implementation we
will be doing in this PA is based off of the method originally described by Yarowsky. This method goes through the
features in training data and calculates the log likelihood of that feature. This is the formula for the log likelihood
ratio log(Pr(SenseA | Collocationi)/Pr(SenseB | Collocationi)). This ratio will be used as the confidence factor for the
classifier. The larger this number is then the more likely this feature is correlated with the sense it is paired with.
Decisions in the decision list will look like the following [feature, sense, log-likelihood] where feature is a unigram
or bigram. In this context a unigram will be one token and a bigram will be two tokens that occur next to each other in
the training data. The sense for each feature in the training data is determine by looking at the beginning of the
review where there will be either a '0' or '1' indicating whether it is positive or negative value. When we are creating
the unigrams from the training data if a not or n't is encountered in the sentence then every word after that up until
the next sentence boundary has a not_ prepended to it. This is done in an effort to better capture the sentiment of
words that follow a negation. The following is an example of this processing: \
    &emsp;&emsp;Before: I did not like this movie at all.   &emsp;      After: I did not not_like not_this not_movie not_at not_all. \
This is not done for bigrams because we consider them to be an orthogonal way to incorporate context.

### Example Input and Output:
input: `./decision-list-train sentiment-train.txt  > sentiment-decision-list.txt` \
    &emsp; Where sentiment-train.txt is the training data and sentiment-decision-list.txt is where the decision list will be sent.

output: The decision list will be written to the file sentiment-decision-list.txt

### Algorithm
Algorithm: \
 &emsp;&emsp;- Grab all of the command line variables and assign them to variables \
 &emsp;&emsp;- Process the training data creating a list of unigram and bigrams \
    &emsp;&emsp;&emsp;&emsp;- Go through the data line by line or in this case review by review \
    &emsp;&emsp;&emsp;&emsp;- If we are making the list of unigrams do the above mentioned not processing on each line \
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;- The list values will be of the format [pos_count, neg_count, log-likelihood] \
    &emsp;&emsp;&emsp;&emsp;- Check if the review is a positive or negative review \
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;- If that feature is already in the list \
            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;- For that feature either iterate the pos_count or neg_count up by on \
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-If that feature isn't in the list \
            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;- Create a new value whit the correct base counts. Start everything at 1 to avoid the issue of dividing by 0 later on. \
    &emsp;&emsp;&emsp;&emsp;- Return the bi/unigram list \
&emsp;&emsp;- Calculate the log-likelihood of feature in both the bigram and unigram list and then combine the two lists \
    &emsp;&emsp;&emsp;&emsp;- Sort this combined list by the log-likelihood and this will then be the decision list \
    &emsp;&emsp;&emsp;&emsp;- Return the decision list

# decision-list-test.py

### Description: 
In the decision-list-train.py program we created a decision list on some training data. Now in this program we will
get the decision list from a file and then test it on some training data. When we test it on the testing data it will
classify the sentiment of each review and then add that to a list of answers.

### Example input and output:
input:  `./decision-list-test  sentiment-decision-list.txt sentiment-test.txt  > sentiment-system-answers.txt` \
&emsp;&emsp;Where sentiment-decision-list.txt is the decision list, sentiment-test.txt is the test data, and sentiment-system-answers.txt is where the answers will be outputted. 

output: The answers will be outputted to sentiment-system-answers.txt

### Algorithm:
Algorithm:\
    &emsp;&emsp;- Grab all of the command line variables and assign them to variables \
    &emsp;&emsp;- Read in the decision list from the inputted file \
    &emsp;&emsp;- Calculate the most common sense \
    &emsp;&emsp;- Go through each review in the test data and classify it with the passed in decision list \
       &emsp;&emsp;&emsp;&emsp; - Write the answers to a list \
    &emsp;&emsp;- Write the list to the specified file
    
# decision-list-eval.py

### Description: 
In this program we are going to read in the answers from the file created by decision-list-test.py and compare it to
the file sentiment-gold.txt which has the correct classification of each review to evaluate how well our classifier
worked.

### Example input and output:
input:  `./decision-list-eval sentiment-gold.txt sentiment-system-answers.txt` \
Where sentiment-gold.txt is the file containing the correct classifications and sentiment-system-answers.txt contains the classifications made from the test program. \

output: It will output a confusion matrix containing raw counts and percentages and the accurracy of the answers \
------------------------------------RESULTS----------------------------------- \
Accuracy: 51.5% \
True Positives: 12(6.0) \
False Positives: 9(4.5) \
True Negatives: 91(45.5) \
False Negatives: 88(44.0) \

### Algorithm:
Algorithm: \
    &emsp;&emsp;- Grab all of the command line variables and assign them to variables \
    &emsp;&emsp;- Read in the gold sense file into a list \
    &emsp;&emsp;- Read the answer sense file into a list \
    &emsp;&emsp;- Compare the two lists and calculate the confusion matrix
