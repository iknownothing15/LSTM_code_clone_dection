import os
import random
import pickle
from pycparser import parse_file
from scripts.myVisitor import MyVisitor
random.seed(114514)
def init_pairs(data,name_to_id,pairs):
    data_pairs=[]
    for pair,label in pairs:
        sentence_1=data[name_to_id[str(pair[0])]]
        sentence_2=data[name_to_id[str(pair[1])]]
        data_pairs.append(((sentence_1,sentence_2),label))
    return data_pairs

def join_dir(path):
    data=[]
    name_to_id={}
    for dirpath,dirnames,filenames in os.walk(path+'oj_clone_programs/'):
        for filename in filenames:
            if(filename.endswith('.cpp')):
                filepath=os.path.join(dirpath,filename)
                ast=parse_file(filepath,use_cpp=False)
                visitor=MyVisitor()
                visitor.visit(ast)
                data.append(visitor.values)
                name=filename.split('.')[0]
                #name is a string
                name_to_id[name]=len(data)-1
    pairs=[]
    print('data initialized with length '+str(len(data)))
    with open(path+'oj_clone_mapping.pkl','rb') as file_mapping:
        mapping=pickle.load(file_mapping)
    pairs=init_pairs(data,name_to_id,mapping)
    return data,pairs

def init():
    training_data,training_pairs=join_dir('data/training/')
    with open('temp/training_data.pkl','wb') as file_training_data:
        pickle.dump(training_data,file_training_data)
    with open('temp/training_pairs.pkl','wb') as file_training_pairs:
        pickle.dump(training_pairs,file_training_pairs)

    test_data,test_pairs=join_dir('data/test/')
    with open('temp/test_data.pkl','wb') as file_test_data:
        pickle.dump(test_data,file_test_data)
    with open('temp/test_pairs.pkl','wb') as file_test_pairs:
        pickle.dump(test_pairs,file_test_pairs)
    
    word_dict={}
    for sentence in training_data:
        for word in sentence:
            if(word not in word_dict):
                word_dict[word]=len(word_dict)
    for sentence in test_data:
        for word in sentence:
            if(word not in word_dict):
                word_dict[word]=len(word_dict)
    with open('temp/word_dict.pkl','wb') as file_word_dict:
        pickle.dump(word_dict,file_word_dict)


def read_data():
    with open('temp/training_data.pkl','rb') as file_training_data:
        training_data=pickle.load(file_training_data)
    with open('temp/training_pairs.pkl','rb') as file_training_pairs:
        training_pairs=pickle.load(file_training_pairs)    
    with open('temp/test_data.pkl','rb') as file_test_data:
        test_data=pickle.load(file_test_data)
    with open('temp/test_pairs.pkl','rb') as file_test_pairs:
        test_pairs=pickle.load(file_test_pairs)
    with open('temp/word_dict.pkl','rb') as file_word_dict:
        word_dict=pickle.load(file_word_dict)
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)
    return training_pairs,test_pairs,word_dict

if __name__ == '__main__':
    init()