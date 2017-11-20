import numpy as np
import time
import random

def my_train(my_train_matrix,train_w,train_row,train_col):
    insert_one = [1]* train_row
    train_y = my_train_matrix[:,-1]
    #截取训练矩阵最后一列，正确率
    my_train_matrix=np.column_stack((insert_one, my_train_matrix))
    #在训练集前方插一列1
    my_train_matrix=np.delete(my_train_matrix,train_col,axis = 1)
    #删除最后一列
    cost_w = [1.0] * train_col
    de_w = train_w
    count = 0
    learn_rate = 1#0.00001
    while(True):
        s_array = np.dot(my_train_matrix, de_w)
        #print(len(s_array),s_array)
        hx_array = (1/(1+np.e**(-s_array))-train_y)
        #print(len(hx_array),hx_array)
        #print(len(de_w), de_w)
        for i in range(0,train_col):
            cost_w[i] = learn_rate* np.dot(hx_array , my_train_matrix[:, i])
            #print(len(cost_w) , cost_w)
        #print(len(train_w), train_w)
        ret = random.uniform(0.5, 1.2)
        learn_rate = learn_rate * ret
        de_w = np.array(de_w) - np.array(cost_w)
        count += 1
        #print(count,de_w)
        if count >= train_row:
            break
    #print(de_w)
    return de_w

def my_val(train_w):
    my_val_matrix = np.loadtxt(val_set,delimiter=",")
    val_row = my_val_matrix.shape[0]
    val_col = my_val_matrix.shape[1]
    insert_one = np.ones((val_row))
    answer_y = my_val_matrix[:,-1]
    my_val_matrix=np.column_stack((insert_one, my_val_matrix))
    my_val_matrix=np.delete(my_val_matrix,val_col,axis = 1)
    val_p = [1] * val_row
    pre_val = [1] * val_row
    for i in range (0,val_row):
        #print("val_p",np.e**(-np.dot(train_w,my_val_matrix[i])))
        val_p[i] = 1/(1+np.e**(-np.dot(train_w,my_val_matrix[i])))
        #print(i, train_w, my_val_matrix[i])
        #print(i,val_p[i])
        if val_p[i] > 0.5:
            pre_val[i] = 1
        else:
            pre_val[i] = 0
        print(i, pre_val[i])
    de_right = 0
    for (i1, i2) in zip(pre_val, answer_y):
        if i1 == i2:
            de_right += 1

    print("正确率为：",de_right/val_row)

def my_test():
    my_test_matrix = np.loadtxt(test_set,delimiter=",")
    test_row = my_test_matrix.shape[0]
    test_col = my_test_matrix.shape[1]
    insert_one = np.ones((test_row))
    test_y = np.ones((test_row))
    answer_y = my_test_matrix[:,-1]
    my_test_matrix=np.column_stack((insert_one, my_test_matrix))
    my_test_matrix=np.delete(my_test_matrix,test_col,axis = 1)


if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/lab5/train.csv'
    val_set = 'E:/B,B,B,BBox/大三上/人工智能/lab5/val.csv'
    test_set = 'E:/B,B,B,BBox/大三上/人工智能/lab5/test.csv'
    my_train_matrix = np.loadtxt(train_set,delimiter=",")
    train_row = my_train_matrix.shape[0]
    train_col = my_train_matrix.shape[1]
    train_w = [1]* train_col
    train_w = my_train(my_train_matrix, train_w, train_row,train_col)

    print(train_w)
    #my_test()
    my_val(train_w)
    end_time = time.time()
    print(end_time - start_time)
