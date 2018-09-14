import os
from os.path import join
import joblib

listwt = []
word_tag = {}

word1 =[]
word2 =[]
word3 = []
cnt1 ={}
cnt2 ={}
cnt3 = {}

_qGram = {}
_PI_y  = {}

IS_THE_FIRST_TIME_READ_FILE = True
ROOT_FOLDER = "brown"
RAW_SENTENCES = []
SAVE_FILE = 'hmm_sentences.n'
_alpha1= {}
_alpha2= {}
_sumQG1= {}

def readFile_first_time():
    for fileName in os.listdir(ROOT_FOLDER):
        with open(join(ROOT_FOLDER, fileName), 'r') as fi:
            for index, line in enumerate(fi):
                line = line.replace("\n", "").strip()
                if line != '' and len(line) > 0:
                    # print(line)
                    line = line.lower()
                    tokens = line.split(' ')
                    token = [item for item in tokens if item!= '' and item.count('/')>0]
                    # print(token)
                    RAW_SENTENCES.append(token)
    print("READING TOTAL :{} SENTENCES".format(len(RAW_SENTENCES)))
    joblib.dump(RAW_SENTENCES, SAVE_FILE )


def readFile():

    print("READ FILE ... ")
    if IS_THE_FIRST_TIME_READ_FILE:
        readFile_first_time()
    newsen = joblib.load(SAVE_FILE)
    for w in newsen:
        RAW_SENTENCES.append(w)
    print(len(RAW_SENTENCES))
    #print(RAW_SENTENCES)
    print("READ FILE complete")

def init_all_word():
    for ss in RAW_SENTENCES:
        se = []
        se.append('START/START')
        se.append('START/START')
        for w in ss :
            se.append(w)
        se.append('STOP/STOP')
        for i in range(0, len(se)):
            listwt.append(se[i])
        tag_list = [item.split('/')[1] for item in se ]
        for i in range (0 , len(tag_list)):
            word1.append(tag_list[i])
        for i in range (0, len(tag_list) - 1):
            word2.append((tag_list[i],  tag_list[i+1] ))
        for i in range (0, len(tag_list) - 2):
            word3.append((tag_list[i] , tag_list[i+1] , tag_list[i+2]))

    print('1 word   :',len(word1))
    print('2 words  :',len(word2))
    print('3 words  :',len(word3))
    init_cnt_wt()
    init_cnt1()
    init_cnt2()
    init_cnt3()
    print('Count UniGram = ',len(cnt1))
    print('Count BiGram  = ',len(cnt2))
    print('Count TriGram = ',len(cnt3))
    init_alpha1()
    init_alpha2()
    #init_sumQG()
    print("Init Completed !")
    word1.sort()
    word2.sort()
    word3.sort()

    return 0
    ############################################################################

def init_cnt_wt():
    for w in listwt: word_tag[w] = 0
    for w in listwt: word_tag[w] += 1

def init_cnt1():
    for w in word1: cnt1[w] = 0
    for w in word1: cnt1[w] += 1
def init_cnt2():
    for w in word2: cnt2[w] = 0
    for w in word2: cnt2[w] += 1
def init_cnt3():
    for w in word3: cnt3[w] = 0
    for w in word3: cnt3[w] += 1

def isInWordX(a , w):
    f = 0
    l = len(a) - 1
    while (f<=l):
        m = (int)((f+l) / 2)
        if (a[m] == w):
            return True
        if (a[m] > w):
            l = m-1
        else :
            f = m+1
    return False

def init_alpha1():
    for w,v in cnt2.items():
        _alpha1[w[0]] = 1
        _alpha2[w[1]] = 1
    for w,v in cnt2.items():
        #w[0] w[1]
        _alpha1[w[0]] -= (v - 0.5)/ cnt1[ w[0]]


def init_alpha2():
    for w,v in cnt3.items():
        _alpha2[(w[0],w[1])] = 1
        _alpha2[(w[1],w[2])] = 1
    for w,v in cnt3.items():
        # w[0] w[1] w[2]
        _alpha2[(w[0], w[1])] -= (v - 0.5) / cnt2 [(w[0] , w[1])]
    print("Init Alpha completed")


def qGram1(w1):
    if ~isInWordX(word1 , w1):
        return 0
    return cnt1[w1] / len(word1)

def alpha1(w1):
    if ~isInWordX(word1 , w1):
        return 1
    return _alpha1[w1]


def alpha2(w1,w2):
    if ~isInWordX(word2, (w1,w2)):
        return 1
    return _alpha2[(w1,w2)]

def init_sumQG1():
    for w,v in cnt2.items():
        _sumQG1[w[0]] = 1
    for w,v in cnt2.items():
        _sumQG1[w[0]] -= qGram(w[1])

def sumQGram1(w1):
    if (w1 not in _sumQG1.keys()):
        return 1
    return _sumQG1[w1]

def qGram2(w1, w2):
    if isInWordX(word2 , (w1,w2)):
        #print('in dict2')
        return (cnt2[(w1, w2)] - 0.5) / cnt1[w1]
    #print('not int dict2')
    xx = qGram1(w2)
    #print(xx)
    yy = sumQGram1(w1)
    #print(yy)
    return alpha1(w1) * xx / yy

def sumQGram2(w1, w2):
    if (w1,w2) not in cnt2.keys():
        return 1
    sum = 0
    for w,v in cnt1.items():
        if (w1,w2,w) not in cnt3.keys():
            sum += qGram2(w, w2)
    return sum


def qGram(w1, w2, w3):
    if isInWordX(word3, (w1,w2,w3)):
        # print('in dict3')
        return (cnt3[(w1, w2, w3)] - 0.5) / cnt2[(w1, w2)]
    # print('not in ditc3')
    return alpha2(w1, w2) * qGram2(w2, w3) / sumQGram2(w1, w2)


def triGram(w1, w2, w3 , _sum):
    if isInWordX(word3, (w1,w2,w3)):
        # print('in dict3')
        return (cnt3[(w1, w2, w3)] - 0.5) / cnt2[(w1, w2)]
    # print('not in ditc3')
    return alpha2(w1, w2) * qGram2(w2, w3) / _sum

def emission_probability(tag , word):
    tmp = word+'/'+tag
    if tmp not in word_tag.keys() :
        return  0
    return word_tag[tmp] / cnt1[tag]

def range_down(start, end):
    while start >= end:
        yield start
        start -= 1

def init_triGram():
    # O (N^3)
    cc = 1
    for w1 in cnt1.keys():
        for w2 in cnt1.keys():
            sum = sumQGram2(w1,w2)
            for w3 in cnt1.keys():
                if (cc%1000000 == 0): print(cc)
                cc+=1
                _qGram[(w1,w2,w3)] = triGram(w1,w2,w3,sum)
    print("Init trigram completed !")
# main
# readFile
readFile()
# init word
init_all_word()
init_triGram()

#run run run
input = 'The man saw the dog with the telescope'
input = input.lower()
sentences = 'START START '+input+' STOP'

token = sentences.split(' ')
tag_ans = ['START' , 'START']

n = len(token)
#init _PI_y
_PI_y[(1,'START','START')] = (1,'START')
for k in range(2,n):
    print("OK " , k)
    for u in cnt1.keys():
        for v in cnt1.keys():
            ma = -1
            ww = 'NONE'
            for w in cnt1.keys():
                if (k-1,w,u) not in _PI_y.keys(): continue
                if (w,u,v) not in word_tag.keys():continue
                pi = _PI_y[(k-1,w,u)]*_qGram[(w,u,v)] [0] *emission_probability(v,token[k])
                if pi > ma :
                    ma = pi
                    ww = w
            _PI_y[(k,u,v)] = (ma,ww)
print('Init _PI_y completed !...')
# #####
print('Solve ...')

#init yn-1, yn-2. yn-3
ma = -1
w1 = 'NONE'
w2 = 'NONE'
w3 = 'STOP'
for u in cnt1.keys():
    for v in cnt1.keys():
        if (n-1, u ,v ) not in _PI_y.keys(): continue
        pi = _PI_y[(n-1,u,v)]*_qGram(u,v,w3)
        if (pi > ma):
            ma = pi
            w1,w2 = u,v
# set yn-2 yn-3 ... yn-1 = 'STOP'
tag_ans = [w1,w2,w3]

for k in range_down(n-4,1):
    pi , yk = _PI_y(k+2, tag_ans[0] , tag_ans[1])
    tag_ans.insert(0,yk)

ans = []
for i in range(0, len(token)):
    ans.append( token[i] + "/" + tag_ans[i] )

print(ans)



# run
#
# input = 'The man saw the dog with the telescope'
# input = input.lower()
# sentences = 'START START '+input+' STOP'
#
# token = sentences.split(' ')
# tag_ans = ['START' , 'START']
# for i in range (0,len(token)-2):
#     # i , i+1, i+2 ... calc tag i+2
#     w1 = tag_ans[i]
#     w2 = tag_ans[i+1]
#     ma = -1
#     next_tag = "NONE"
#     for w3,v in cnt1.items():
#         ee = emission_probability(w3 , token[i+2])
#         if ee == 0 :
#             continue
#         qq = qGram(w1,w2,w3)
#         if (qq*ee > ma):
#             ma = qq
#             next_tag = w3
#     tag_ans.append(next_tag)
#
# ans = []
# for i in range(0, len(token)):
#     ans.append( token[i] + "/" + tag_ans[i] )
#
# print(ans)