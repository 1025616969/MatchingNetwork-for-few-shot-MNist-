#-*- coding:utf-8 -*-
import argparse

def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_way',type=int,default=5)
    parser.add_argument('--data_shot',type=int,default=1)
    parser.add_argument('--data_query',type=int,default=5)
    parser.add_argument('--data_test_way',type=int,default=5)
    parser.add_argument('--data_test_shot',type=int,default=1)
    parser.add_argument('--data_test_query',type=int,default=15)
    parser.add_argument('--data_train_episodes',type=int,default=100)
    parser.add_argument('--data_test_episodes',type=int,default=100)

    parser.add_argument('--train_epoches',type=int,default=10)
    parser.add_argument('--train_optim_method',type=str,default='Adam')
    parser.add_argument('--train_learningrate',type=float,default=0.001)
    parser.add_argument('--train_decay_every',type=int,default=20,help='number of epoches after which to decay the lr')

    parser.add_argument('--data_root',type=str,default='data/omniglot')
    parser.add_argument('--data_cuda',action='store_true')
    parser.add_argument('--log_file',type=str,default='results')

    parser.add_argument('--run_mode',type=str,default='train_val',
                        help='train_val代表train+val, train代表只有train， test 代表 测试')

    return parser