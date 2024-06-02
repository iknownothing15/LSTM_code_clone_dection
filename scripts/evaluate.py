from scripts.model import MyModule,EMBEDDING_DIM,HIDDEN_DIM,prepare_sequence
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from scripts.preprocess import parser_single
def evaluate(test_pairs,word_dict):
    model=torch.load('model.pt')
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for pair, label in test_pairs:
            model.hidden=model.init_hidden()
            sentence_1, sentence_2 = pair
            sentence_in_1 = prepare_sequence(sentence_1, word_dict)
            sentence_in_2 = prepare_sequence(sentence_2, word_dict)
            min_size = min(len(sentence_1), len(sentence_2))
            index = - min_size
            tag_scores_1 = model(sentence_in_1)[index:]
            tag_scores_2 = model(sentence_in_2)[index:]
            distance = functional.pairwise_distance(tag_scores_1.view(1, -1), tag_scores_2.view(1, -1), p=1)
            predicted = 1.0 if distance.item() < 0.5 else -1.0  # 这里我们假设距离小于0.5的两个句子是相似的
            # print('id=%d distance=%f' % (id, distance.item()))
            if predicted == label:
                correct += 1
            total += 1
    print('Accuracy of the model on the test pairs: {}%'.format(100 * correct / total))

def evaluate_single(file1,file2,word_dict):
    model=torch.load('model.pt')
    sentence_1=parser_single(file1)
    sentence_2=parser_single(file2)
    sentence_in_1=prepare_sequence(sentence_1,word_dict)
    sentence_in_2=prepare_sequence(sentence_2,word_dict)
    min_size=min(len(sentence_1),len(sentence_2))
    index=-min_size
    tag_scores_1=model(sentence_in_1)[index:]
    tag_scores_2=model(sentence_in_2)[index:]
    distance=functional.pairwise_distance(tag_scores_1.view(1,-1),tag_scores_2.view(1,-1),p=1)
    re_distance=distance.item() if distance.item()>0 else 0
    predict=100*(1-re_distance)
    print('Similarity between %s and %s is %f' % (file1,file2,predict))
    return predict