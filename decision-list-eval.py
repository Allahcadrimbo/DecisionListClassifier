"""
Author: Grant Mitchell
Date: 10/23/19
PA #3 NLP

In this program we are going to read in the answers from the file created by decision-list-test.py and compare it to
the file sentiment-gold.txt which has the correct classification of each review to evaluate how well our classifier
worked.

input:  ./decision-list-eval sentiment-gold.txt sentiment-system-answers.txt

    Where sentiment-gold.txt is the file containing the correct classifications and sentiment-system-answers.txt
    contains the classifications made from the test program.

output: It will output a confusion matrix containing raw counts and percentages and the accurracy of the answers
------------------------------------RESULTS-----------------------------------
Accuracy: 51.5%
True Positives: 12(6.0)
False Positives: 9(4.5)
True Negatives: 91(45.5)
False Negatives: 88(44.0)

Algorithm:
    - Grab all of the command line variables and assign them to variables
    - Read in the gold sense file into a list
    - Read the answer sense file into a list
"""
import sys


# Reads in the sense from a specified file
def read_in_sense(filename):
    sense_list = []
    with open(filename, "r") as file:
        # Both files reviews are in the same order so we can just make a list of the senses
        for line in file:
            i = 0
            # for each line grab just the sense and add it to the sense_list
            for word in line.split():
                if i != 0:
                    sense_list.append(word)
                i += 1
    return sense_list


# Calculate the confusion matrix with the gold senses and the senses we answered
def calculate_matrix(gold, answer):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # For all of the values in the list check if it is a true positive, true negative, false positve, false negative
    for x in range(len(gold)):
        if gold[x] == '0' and answer[x] == '0':  # True Negative
            tn = tn + 1
        elif gold[x] == '0' and answer[x] == '1':  # False Positives
            fp = fp + 1
        elif gold[x] == '1' and answer[x] == '1':  # True Positives
            tp = tp + 1
        elif gold[x] == '1' and answer[x] == '0':  # False Negative
            fn = fn + 1

    # Print the information
    print_confusion_matrix(tp, fp, tn, fn, (tp + fp + tn + fn))


# Prints the confusion matrix info and accuracy
def print_confusion_matrix(tp, fp, tn, fn, total):
    accuracy = (tp + tn)/total * 100
    print("------------------------------------RESULTS-----------------------------------")
    print("Accuracy: " + str(accuracy) + "%")
    print("True Positives: " + str(tp) + "(" + str(tp/total*100) + ")")
    print("False Positives: " + str(fp) + "(" + str(fp / total * 100) + ")")
    print("True Negatives: " + str(tn) + "(" + str(tn / total * 100) + ")")
    print("False Negatives: " + str(fn) + "(" + str(fn / total * 100) + ")")


if __name__ == "__main__":
    gold_filename = sys.argv[1]  # The file name of the correct classifications
    answers_file_name = sys.argv[2]  # The file of the classifications made from the test program

    gold_sense_list = read_in_sense(gold_filename)  # Read the correct classifications into a list
    answers_sense_list = read_in_sense(answers_file_name)  # Read the classifications we made into a list

    # Calculate the accuracy and the confusion matrix
    calculate_matrix(gold_sense_list, answers_sense_list)
