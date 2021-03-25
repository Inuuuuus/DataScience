####
Author zwz
Createtime 20210325


# PART1 calculate words, tags
tag2id, id2tag = {}, {}
word2id, id2word = {}, {}

for line in open('./traindata.txt'):
    items = line.split('/')
    word, tag = items[0], items[1].rstrip()

    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

M = len(word2id)  # 词典的大小
N = len(tag2id)   # 词性的种类个数
#print(M,N)

# PART2 Calculate pi, A, B
import numpy as np

pi = np.zeros(N) # 每个词性出现在句子中第一个位置的概率 N: # of tags pi[i]: tag i出现在句子中第一个位置的概率
A = np.zeros((N, M)) # A[i][j]: 给定tag i 出现单词j的概率
B = np.zeros((N, N)) # B[i][j]: 之前的状态是i,之后转换成状态j的概率

prev_tag = ""
for line in open('./traindata.txt'):
    items = line.split('/')
    wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
    if prev_tag == "": # 意味着句子开始
        pi[tagId] += 1
        A[tagId][wordId] += 1
    else:   # 如果不是句子开始
        A[tagId][wordId] += 1
        B[tag2id[prev_tag]][tagId] += 1
    if items[0] == ".": # 意味着句子结束
        prev_tag = ""
    else:
        prev_tag = items[1].rstrip()

# normalize
pi = pi/sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

def log(v):
    if v == 0:
        return np.log(v+0.00000001)
    else:
        return np.log(v)

# PART3 Calculate Viterbi Path
def viterbi(x,pi,A,B):
    """
    	x: user input string/sentence
    	pi: Initial probabilities of tags
    	a: 给定tag，每个单词出现的概率
    	b: tag之间的转移概率
    """
    x = [word2id[word] for word in x.split(" ")]
    T = len(x)

    dp = np.zeros((T,N))
    ptr = np.array([[0 for x in range(N)] for y in range(T)])

    for j in range(N):
        dp[0][j] = log(pi[j])+log(A[j][x[0]])

    for i in range(1,T):
        for j in range(N):
            dp[i][j] = -9999
            for k in range(N):
                score = dp[i-1][k]+log(B[k][j])+log(A[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k
    # decoding 把最好的tag sequence打印出来
    best_seq = [0]*T
    # step1: 找出对应于最后一个单词的词性
    best_seq[T-1] = np.argmax(dp[T-1])
    # step2: 通过从后到前的循环依次求出每个单词的词性
    for i in range(T-2,-1,-1):
        best_seq[i] = ptr[i+1][best_seq[i+1]]

    #到目前为止，best_seq存放了对应于x的词性序列
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])

if __name__ == '__main__':
    x = "Social Security number , passport number and details about the services provided for the payment"
    viterbi(x,pi,A,B)
