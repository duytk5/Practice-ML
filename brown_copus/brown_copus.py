import os
from os.path import join
import joblib

word1 = word2 = word3 = []
cnt1 = cnt2 = cnt3 = {}

ROOT_FOLDER = "brown"
RAW_SENTENCES = []


def readFile_first_time():
    for fileName in os.listdir(ROOT_FOLDER):
        with open(join(ROOT_FOLDER, fileName), 'r') as fi:
            for index, line in enumerate(fi):
                line = line.replace("\n", "").strip()
                line = line.replace("./.", "").strip()
                line = line.replace(",/,", "").strip()
                line = line.replace("''/''", "").strip()
                line = line.replace("'/'", "").strip()
                line = line.replace("``/``", "").strip()
                if line != '' and len(line) > 0:
                    # print(line)
                    line = line.lower()
                    tokens = line.split(' ')
                    token = [item.split('/')[0] for item in tokens if item.split('/')[0] != '']
                    # print(token)
                    RAW_SENTENCES.append(token)
    print("READING TOTAL :{} SENTENCES".format(len(RAW_SENTENCES)))
    joblib.dump(RAW_SENTENCES, 'sentences.n')


def readFile():

    print("READ FILE ... ")
    #readFile_first_time()
    newsen = joblib.load('sentences.n')
    for w in newsen:
        RAW_SENTENCES.append(w)
    #print(RAW_SENTENCES)
    print("READ FILE complete")

def init_all_word():

    word1.append('START')
    word1.append('END')
    word2.append(('START' , 'END'))
    word2.append(('START' , 'START'))
    word3.append(('START' , 'START' , 'END'))
    for se in RAW_SENTENCES:
        for w in se :
            word1.append(w)


        n = len(se)
        for i in range ( 0 , n-1 ):
            word2.append((se[i] , se[i+1]))
            word3.append(('START', se[i] , se[i+1]))
            word3.append((se[i] , se[i+1] , 'END'))

        for w in se :
            word2.append(('START', w))
            word2.append((w, 'END'))
            word3.append(('START', 'START' , w))
            word3.append(('START' , w , 'END'))

        for i in range (0 , n-2):
            word3.append((se[i] , se[i+1] , se[i+2]))
        word3.append(('START' , 'START' , 'END'))
    init_cnt1()
    init_cnt2()
    init_cnt3()
    print("Init Complete !")

    return 0
    ############################################################################


def init_cnt1():
    for w in word1: cnt1[w] = 0
    for w in word1: cnt1[w] += 1


def init_cnt2():
    for w in word2: cnt2[w] = 0
    for w in word2: cnt2[w] += 1


def init_cnt3():
    for w in word3: cnt3[w] = 0
    for w in word3: cnt3[w] += 1

def alpha(w1):
    sum = 0
    for w in word1:
        sum += (cnt2[(w1, w)] - 0.5) / cnt1[w1]
    return 1 - sum

def alpha(w1,w2):
    sum = 0
    for w in word1:
        if (w1,w2,w) in word3:
            sum += (cnt3[(w1,w2,w)] - 0.5) / cnt2[(w1,w2)]
    return 1 - sum

def qGram(w1):
    return cnt1[w1] / len(word1)


def sumQGram1(w1):
    sum = 0
    for w in word1:
        if cnt2[(w1, w)] == 0:
            sum += qGram(w)
    return sum


def qGram(w1, w2):
    if (w1,w2) in cnt2 :
        return (cnt2[(w1, w2)] - 0.5) / cnt1[w1]
    return alpha(w1) * qGram(w2) / sumQGram1(w1)


def sumQGram2(w1, w2):
    sum = 0
    for w in word1:
        if cnt3[(w1, w2, w)] == 0:
            sum += qGram(w, w2)
    return sum


def qGram(w1, w2, w3):
    if (w1,w2,w3) in cnt3:
        return (cnt3[(w1, w2, w3)] - 0.5) / cnt2[(w1, w2)]
    return alpha(w1, w2) * qGram(w3, w2) / sumQGram2(w1, w2)


def N_Gram_Sentence(s):
    ans = 1
    for i in range(0, len(s) - 2):
        tmp = qGram(s[i], s[i + 1], s[i + 2])
        ans *= tmp
        print(tmp)
    return ans


# main

# readFile
readFile()

# init word
init_all_word()
print("WORD 3 :")
# checkNGram-aSentence
sentence = ['START' , 'START' , 'the' ,  'sun' ,  'rises' ,  'in' , 'the' ,  'east' , 'and' , 'sets' , 'in' ,  'the' ,  'west' , 'END']
print(sentence)
print("N-Gram-Sentence : " , N_Gram_Sentence(sentence))

# randomSentence-length-m