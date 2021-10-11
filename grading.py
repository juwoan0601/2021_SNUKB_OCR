import csv
import numpy as np
def compare_csv(source_path, target_path):
    source_file = open(source_path, 'r', encoding='utf-8')
    target_file = open(target_path, 'r', encoding='utf-8')
    source_reader = csv.reader(source_file)
    target_reader = csv.reader(target_file)
    source_data = {}
    target_data = {}
    for line in source_reader:
        if not line == '\n': source_data[line[0]] = line[1]
    for line in target_reader: 
        if not line == '\n': target_data[line[0]] = line[1]
    nunber_of_data = len(source_data)
    correct_word = {}
    correct_char = {}
    for key, value in source_data.items():
        if source_data[key] == target_data[key]: correct_word[key] = 1
        else: correct_word[key] = 0
        len_char = len(source_data[key])
        if len_char == 0:
            correct_char[key] = 0
        else:
            nc_char = 0
            len_comp = len(source_data[key]) if len(source_data[key]) < len(target_data[key]) else len(target_data[key])
            for cn in range(len_comp): 
                if source_data[key][cn] == target_data[key][cn]: nc_char+=1
            correct_char[key] = nc_char/len_char
    
    sum_correct_word = 0
    sum_correct_char = 0
    for key, value in correct_word.items(): sum_correct_word += value
    for key, value in correct_char.items(): sum_correct_char += value

    print("[GRADE] Source File: {0}".format(source_path))
    print("[GRADE] Target File: {0}".format(target_path))
    print("[GRADE] Word Correct: {0}/{1} - {2}%".format(sum_correct_word, len(correct_word), (sum_correct_word/len(correct_word))*100))
    print("[GRADE] Average Character Correct: {0}%".format((sum_correct_char/len(correct_word))*100))

if __name__ == "__main__":
    true_path = "./images/valid_1000.csv"
    predict_path = "./result/valid_1000_STEP6_40/result.csv"
    compare_csv(true_path, predict_path)
        
