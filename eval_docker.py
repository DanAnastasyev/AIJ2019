import json
import os
import numpy
import pickle
import requests
from collections import defaultdict
import attr

@attr.s
class Score(object):
    score = attr.ib(default=0)
    max_score = attr.ib(default=0)

def zero_if_exception(scorer):
    def new_scorer(*args, **kwargs):
        try:
            return scorer(*args, **kwargs)
        except KeyboardInterrupt:
            return 0
    return new_scorer

def get_score(y_true, prediction):
    if "correct" in y_true:
        if y_true["correct"] == prediction:
            return 1
    elif "correct_variants" in y_true and isinstance(y_true["correct_variants"][0], str):
        if  prediction in y_true["correct_variants"]:
            return 1
    elif "correct_variants" in y_true and isinstance(y_true["correct_variants"][0], list):
        y_true = set(y_true["correct_variants"][0])
        y_pred = set(prediction)
        return int(len(set.intersection(y_true, y_pred)) == len(y_true) == len(y_pred))
    return 0

# для 8 и 26
def get_matching_score(y_true, pred):
    score = 0
    y_true = y_true["correct"]
    if len(y_true) != len(pred):
        return 0
    for y in y_true:
        if y_true[y] == pred[y]:
            score += 1
    return score

# для 16 задания
def get_multiple_score(y_true, y_pred):
    y_true = y_true["correct_variants"][0] if "correct_variants" in y_true else y_true["correct"]
    while len(y_pred) < len(y_true):
        y_pred.append(-1)
    return max(0, len(set.intersection(set(y_true), set(y_pred))) - len(y_pred) + len(y_true))

def run_tasks(dir_path, task_num=None):
    tasks, filenames = [], [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    task_scores = defaultdict(Score)
    test_scores = []
    for filename in filenames:
        if filename.endswith(".json"):
            predictions = []
            with open(filename) as f:
                data = json.load(f)
                resp = requests.post('http://localhost:8000/take_exam', json={'tasks':data})

                answers = resp.json()['answers']
                for i, task in enumerate(data):
                    task_index, task_type = int(task['id']), task["question"]["type"]
                    y_true = task["solution"]
                    prediction = answers[task['id']]
    
                    if task_type == "matching":
                        score = get_matching_score(y_true, prediction)
    
                        if "correct_variants" in y_true:
                            correct_values = y_true["correct_variants"][0]
                        else:
                            correct_values = y_true["correct"]
    
                        max_score = len(correct_values)
                    elif task_index == 16:
                        score = get_multiple_score(y_true, prediction)
    
                        if "correct_variants" in y_true:
                            correct_values = y_true["correct_variants"][0]
                        else:
                            correct_values = y_true["correct"]
    
                        max_score = len(correct_values)
                    else:
                        score = get_score(y_true, prediction)
                        max_score = 1
    
                    task_scores[task_index].score += score
                    task_scores[task_index].max_score += max_score
    
                    print("Score: {} / {}\nCorrect: {}\nPrediction: {}\n".format(score, max_score, y_true, prediction))
                    predictions.append(score)
                test_scores.append(predictions)
    print("Results per task:")
    for task_id in sorted(task_scores, key=lambda id: int(id)):
        print("{}\t{:.3%}".format(task_id,
            float(task_scores[task_id].score) / task_scores[task_id].max_score
            if task_scores[task_id].max_score != 0 else 0.))
    for task_id in sorted(task_scores, key=lambda id: int(id)):
        print("{}".format(
            float(task_scores[task_id].score) / task_scores[task_id].max_score
            if task_scores[task_id].max_score != 0 else 0.) * 100)

#tasks = load_tasks('public_set/train', task_num=8)




print(requests.get('http://localhost:8000/ready'))


#resp = requests.post('http://localhost:8000/take_exam', json={'tasks':tasks[:30]})

#print(resp.json())

run_tasks('dataset/test')
