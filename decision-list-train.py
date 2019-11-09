"""
Author: Grant Mitchell
Date: 10/23/19
PA #3 NLP

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
words that follow a negation. The following is an example of this processing:
    Before: I did not like this movie at all.           After: I did not not_like not_this not_movie not_at not_all .
This is not done for bigrams because we consider them to be an orthogonal way to incorporate context.

This particular file will handle the training portion of the PA. That is it will analyse the training data and will
create a decision list classifier. The following is example input and output:

input: ./decision-list-train sentiment-train.txt  > sentiment-decision-list.txt
    Where sentiment-train.txt is the training data and sentiment-decision-list.txt is where the decision list will be
    sent.

output: The decision list will be written to the file sentiment-decision-list.txt

Algorithm:
 - Grab all of the command line variables and assign them to variables
 - Process the training data creating a list of unigram and bigrams
    - Go through the data line by line or in this case review by review
    - If we are making the list of unigrams do the above mentioned not processing on each line
        - The list values will be of the format [pos_count, neg_count, log-likelihood]
    - Check if the review is a positive or negative review
        - If that feature is already in the list
            - For that feature either iterate the pos_count or neg_count up by on
        -If that feature isn't in the list
            - Create a new value whit the correct base counts. Start everything at 1 to avoid the issue of dividing by
                0 later on.
    - Return the bi/unigram list
- Calculate the log-likelihood of feature in both the bigram and unigram list and then combine the two lists
    - Sort this combined list by the log-likelihood and this will then be the decision list
    - Return the decision list
"""

import sys
import re
import math


#  This function goes through each review in the training data and creates either a bigram or unigram list, specified by
#  is_unigram. Each value in the list is of the form [pos_count, neg_count, log-likelihood]. When creating the list it
#  will increment either the pos_count or neg_count depending on the review classification
#  @param train_filename  The file name of the training data
#  @param is_unigram  Determines whether to make a list of unigrams or bigrams
#  @return Either a list of unigrams or bigrams
def analyse_text(train_filename, is_unigram):
    gram_list = dict()

    # Opens the training data
    with open(train_filename, "r", encoding="unicode_escape") as text:
        for line in text:
            # We wan't to check if we are doing the unigrams of bigrams currently
            # because we will only do the not prepending on unigrams. We do not do it on the bigrams because we consider
            # them to be an orthogonal way to incorporate context
            if is_unigram:
                processed_line = prepend_not_processing(line)

                # Check to make sure the processed_line isn't None because it will return None if there isn't any not
                # processing to be done
                if processed_line is not None:
                    line = processed_line

                # Check if the current review is of positive or negative sentiment and then updated the sense counter
                if re.search(r'.txt\s(0)\s.*', line):  # negative reviews
                    update_sense_counter(line.split(), gram_list, 1)
                elif re.search(r'.txt\s(1)\s.*', line):  # positive reviews
                    update_sense_counter(line.split(), gram_list, 0)
            else:
                if re.search(r'.txt\s(0)\s.*', line):  # negative reviews
                    gram_list = process_model(create_ngram_model(2, line.split()), 0)
                elif re.search(r'.txt\s(1)\s.*', line):  # positive reviews
                    gram_list = process_model(create_ngram_model(2, line.split()), 1)

        return gram_list


# Creates a ngram model dictated by the integer passed in as n and trains the model on the list of token passed in
def create_ngram_model(n, tokens):
    ngram_result = []  # New list to hold the ngram

    # For every token in the list we will grab the token plus the next n tokens and then append that gram to the model
    # We can grab the next n tokens because it is essentially the same thing as grabbing the previous n tokens. They
    # both will yield the same result. We stop at len(tokens)-n because anything past that won't have n tokens ahead of
    # it.
    for j in range(len(tokens) - n):
        ngram_result.append(tokens[j:j + n])

    return ngram_result


# This function processes the model that was just created. The model will be a list of lists that looks something like
# [["x", "y", "z"],...] but we want it to look like [["x y z"],...] so this method does that for us.
def process_model(model, sense):
    processed_model = []  # New list to hold the processed gram
    processed_model_with_sense = dict()

    # For every sublist in the list model we will join each element with a space in between
    for gram in model:
        processed_model.append(' '.join(gram))

    for gram in processed_model:
        if gram in processed_model_with_sense:
            # Increment the appropriate sense counter
            processed_model_with_sense[gram][sense] = processed_model_with_sense[gram][sense] + 1
        else:
            if sense == 0:
                processed_model_with_sense[gram] = [2, 1, 0]  # Initialize values at 1 to avoid divide by 0 issues later
            else:
                processed_model_with_sense[gram] = [1, 2, 0]

    return processed_model_with_sense


# This function will go through each word in a review and increment it's sense counter in the list
# @param words The words in the current review
# @param gram_list List of unigrams
# @param sense The sentiment of the current review
def update_sense_counter(words, gram_list, sense):
    for word in words:
        if word in gram_list:
            # Increment the appropriate sense counter
            gram_list[word][sense] = gram_list[word][sense] + 1
        else:
            if sense == 0:
                gram_list[word] = [2, 1, 0]  # Initialize values at 1 to avoid having divide by 0 issues later on
            else:
                gram_list[word] = [1, 2, 0]


# Will take a review and append not_ to every word after not or n't until the next sentence boundary
# @param text A review in the training data
def prepend_not_processing(text):
    # Check to see if there is a not or n't in the text
    if re.search(r'not[^.?!]+|n\'t[^.?!]+', text):
        # Get the text after the occurrence of not or n't
        text_after_not = re.findall(r'not[^.?!]+|n\'t[^.?!]+', text)
        # Findall will return it as a list so we need turn it into a string
        text_after_not = text_after_not[0]
        # Splits the string on whitespace so that we can iterate through the wors
        text_after_not = text_after_not.split()
        text_new = ""
        # For each word if it isn't not or doesn't contain n't it prepends not_ onto it
        for word in text_after_not:
            if word == "not":
                text_new = text_new + word + " "
            elif re.search(r'.*n\'t.*', word):
                text_new = text_new + word + " "
            else:
                text_new = text_new + "not_" + word + " "

        # Get the original ending sentence boundary
        punctuation = re.sub(r'.*not[^.?!]+|.*n\'t[^.?!]+', '', text)

        # Get the text before the not or n't
        before_not = re.sub(r'not.*|n\'t.*', '', text)

        # Concatenate the new sentence
        return before_not + text_new + punctuation  # Returns the processed text or None if it wasn't processed


# Calculates the log likelihood of a feature
# @param model The dictionary of processed features with sense
def calc_log_likelihood(model):
    for feature, sense_values in model.items():
        sense_total = float(sense_values[0] + sense_values[1])
        pr_s1_ci = float(sense_values[0] / sense_total)
        pr_s2_ci = float(sense_values[1] / sense_total)
        sense_values[2] = abs(math.log(pr_s1_ci / pr_s2_ci))

    return model


# Given a bigram and unigram dict it creates a decision list with the format [feature, sense, log-likelihood]
def make_decision_list(unigram_dict, bigram_dict):
    l1 = []
    l2 = []

    # For all of the unigram features assign a sentiment based on whether the pos_count or the neg_count is larger
    for key, value in unigram_dict.items():
        if value[0] > value[1]:
            sense = 1
        else:
            sense = 0
        # Add the feature, it's sense, and likelihood to the list and prepend unigram to assist processing later
        l1.append(["unigram:", key, sense, value[2]])

    # For all of the biigram features assign a sentiment based on whether the pos_count or the neg_count is larger
    for key, value in bigram_dict.items():
        if value[0] > value[1]:
            sense = 1
        else:
            sense = 0
        # Add the feature, it's sense, and likelihood to the list and prepend bigram to assist processing later
        l2.append(["bigram:", key, sense, value[2]])

    # Combine the two lists
    l1.extend(l2)

    # Sort the decision list by log-likelihood
    l1.sort(key=lambda tup: tup[3], reverse=True)

    return l1


# Sends the decision list (dcl) to the specified file
def send_to_file(dcl):
        for decision in dcl:
            rule = ""
            for value in decision:
                rule = rule + str(value) + " "
            print(rule + '\n')


if __name__ == "__main__":
    train_file = sys.argv[1]  # Gets the training data filename

    # Creates the unigram and bigram decision lists
    unigram_dic = analyse_text(train_file, True)
    bigram_dic = analyse_text(train_file, False)

    # Makes the decision list
    decision_list = make_decision_list(calc_log_likelihood(unigram_dic), calc_log_likelihood(bigram_dic))

    # Sends the decision list to the specified file
    send_to_file(decision_list)
