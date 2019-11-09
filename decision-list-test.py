"""
Author: Grant Mitchell
Date: 10/23/19
PA #3 NLP

In the decision-list-train.py program we created a decision list on some training data. Now in this program we will
get the decision list from a file and then test it on some training data. When we test it on the testing data it will
classify the sentiment of each review and then add that to a list of answers.

input:  ./decision-list-test  sentiment-decision-list.txt sentiment-test.txt  > sentiment-system-answers.txt

    Where sentiment-decision-list.txt is the decision list, sentiment-test.txt is the test data, and
    sentiment-system-answers.txt is where the answers will be outputted.

output: The answers will be outputted to sentiment-system-answers.txt

Algorithm:
    - Grab all of the command line variables and assign them to variables
    - Read in the decision list from the inputted file
    - Calculate the most common sense
    - Go through each review in the test data and classify it with the passed in decision list
        - Write the answers to a list
    - Write the list to the specified file
"""

import re
import sys


# This function reads the decision list that was created in the first program from a specified file
# @param filename The file name of the decision list
def read_in_decision_list(filename):
    decision_list = []
    with open(filename, "r", encoding="unicode_escape") as dcl:
        # For every line check if the rule is for a unigram or bigram feature using our text we prepended in the last
        # program. Then add the rule to the list so the values are of the form [feature, sense, log-likelihood]
        for line in dcl:
            if re.search(r'^unigram: .*', line):  # if it is a unigram rule
                line = re.sub(r'^unigram:\s*', r'', line)  # remove the helper string

                val = read_helper(line)

                # Append the rule to the decision list
                decision_list.append([val[0], int(val[1]), float(val[2])])

            elif re.search(r'^bigram: .*', line):  # if it is a bigram rule
                line = re.sub(r'^bigram: ', r'', line)  # remove the helper string

                val = read_helper(line)

                # Append the rule to the decision list
                decision_list.append([str(val[0]) + " " + str(val[1]), int(val[2]), float(val[3])])

    return decision_list


# Takes in a line and reads each word into a list
def read_helper(line):
    val = []
    i = 0
    for value in line.split():
        val.append(value)
        i = i + 1
    return val


# Read in the test data and for each rule in the decision list check if the rule is in the review. If it is assign
# the review the sense of said rule.
def test_test_data(test_filename, dcl, def_sense):
    answers = []
    with open(test_filename, "r", encoding="unicode_escape") as test_data:
        # For each review get the review id and compare it on the decision list
        for line in test_data:
            reviewid = get_review_id(line)  # Review id of the review

            matched_pattern = False  # Used to check if it ends up matching a rule

            # For every rule in the decision list check if the feature is in the review. If it is then add the
            # classification to the answer list with the reviews corresponding id
            for decision in dcl:
                if decision[0] in line:
                    answers.append([reviewid, decision[1]])
                    matched_pattern = True
                    break

            # If the review doesn't match any of the rules then assign is the most common sense
            if not matched_pattern:
                answers.append([reviewid, def_sense])
    return answers


# Gets the first substring from the line
def get_review_id(line):
    rid = re.sub(r'(.*.txt)(.*)', r'\1', line)
    return rid.strip("\n")


# Given a decision list determing which sense is the most common
def get_most_common_sense(dcl):
    pos_sense = 0
    neg_sense = 0
    for decision in dcl:
        if decision[1] == 0:
            neg_sense = neg_sense + 1  # Count the number of negatives senses
        else:
            pos_sense = pos_sense + 1  # Count the number of positive senses

    # Determing which sense is more common and return it
    if pos_sense > neg_sense:
        return 1
    else:
        return 0


# Given a filename and a list of answers write the list to the specified file
def write_answers(al):
        for answer in al:
            ans = ""
            for value in answer:
                ans = ans + str(value) + " "
            print(ans + '\n')


if __name__ == "__main__":
    decision_list_filename = sys.argv[1]  # Decision list file name
    test_data_file_name = sys.argv[2]  # Test data filename

    # Return the decision list from the inputted file
    dec_list = read_in_decision_list(decision_list_filename)

    # Calculate the most common sense
    mcs = get_most_common_sense(dec_list)

    # Classify the test data with the decision list and create a list of answers
    answers_list = test_test_data(test_data_file_name, dec_list, mcs)

    # Write the answers to specified answer
    write_answers(answers_list)
