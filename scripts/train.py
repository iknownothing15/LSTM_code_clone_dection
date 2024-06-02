from scripts.model import MyModule,EMBEDDING_DIM,HIDDEN_DIM,prepare_sequence
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from scripts.preprocess import parser_single
def train(training_data,word_dict,EPOCH=20):
    model=MyModule(EMBEDDING_DIM,HIDDEN_DIM,len(word_dict),HIDDEN_DIM)
    # model=torch.load('model.pt')
    loss_function=nn.HingeEmbeddingLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1)
    for epoch in range(EPOCH):
        i = 0
        epoch_loss = 0
        running_loss = 0
        # 遍历训练数据中的每一对句子和对应的标签
        for pair, label in training_data:
            # 清零模型的梯度
            model.zero_grad()
            # 初始化模型的隐状态
            model.hidden = model.init_hidden()
            sentence_1, sentence_2 = pair
            # 将句子转换为张量
            sentence_in_1 = prepare_sequence(sentence_1, word_dict)
            sentence_in_2 = prepare_sequence(sentence_2, word_dict)
            # 计算两个句子的最小长度
            min_size = min(len(sentence_1), len(sentence_2))
            index = - min_size

            # 前向传播，得到标签分数
            tag_scores_1 = model(sentence_in_1)[index:]
            tag_scores_2 = model(sentence_in_2)[index:]
            # 计算两个标签分数的距离
            distance = functional.pairwise_distance(tag_scores_1.view(1, -1), tag_scores_2.view(1, -1), p=1)
            # 计算损失
            loss = loss_function(distance, torch.tensor([label]))
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            epoch_loss += loss.item()
            i += 1
            if i % 500 == 499:
                print(epoch, i + 1, "running loss: ", running_loss / 499)
                running_loss = 0
        print('epoch %d: finish to train different codes' % epoch)
        print('average loss of epoch %d: %f' % (epoch, epoch_loss / len(training_data)))
        print('save model to file')
        # 保存模型
        torch.save(model, 'model.pt')
        epoch_loss = 0