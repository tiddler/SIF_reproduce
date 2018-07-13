from SIF import utils
import SIF.embedding


STS14 = [
    'data/sts2014/deft-forum.test.tsv',
    'data/sts2014/deft-news.test.tsv',
    'data/sts2014/headlines.test.tsv',
    'data/sts2014/images.test.tsv',
    'data/sts2014/OnWN.test.tsv',
    'data/sts2014/tweet-news.test.tsv'
]

STS15 = [
    'data/sts2015/answers-forums.test.tsv',
    'data/sts2015/answers-students.test.tsv',
    'data/sts2015/belief.test.tsv',
    'data/sts2015/headlines.test.tsv',
    'data/sts2015/images.test.tsv'
]

GLOVE_PATH = './resources/glove.840B.300d.txt'
PSL_PATH = './resources/paragram_300_sl999.txt'

if __name__ == '__main__':
    glove = utils.WordToWeight(GLOVE_PATH)
    psl = utils.WordToWeight(PSL_PATH)

    for embedding_method, name in [
        (SIF.embedding.AVG_embedding, 'average weighted'),
        (SIF.embedding.W_embedding, 'freq weighted'),
        (SIF.embedding.WR_embedding, 'freq weighted + SVD')]:
        scores = []
        for data in STS14:
            res = utils.evaluate(data, embedding_method, glove)
            scores.append(res)
            print(name, data, res)
        print(name, 'STS14 average: ', sum(scores) / len(scores))

        scores = []
        for data in STS15:
            res = utils.evaluate(data, embedding_method, glove)
            scores.append(res)
            print(name, data, res)
        print(name, 'STS15 average: ', sum(scores) / len(scores))

    for embedding_method, name in [
        (SIF.embedding.AVG_embedding, 'average weighted'),
        (SIF.embedding.W_embedding, 'freq weighted'),
        (SIF.embedding.WR_embedding, 'freq weighted + SVD')]:
        scores = []
        for data in STS14:
            res = utils.evaluate(data, embedding_method, psl)
            scores.append(res)
            print(name, data, res)
        print(name, 'STS14 average: ', sum(scores) / len(scores))

        scores = []
        for data in STS15:
            res = utils.evaluate(data, embedding_method, psl)
            scores.append(res)
            print(name, data, res)
        print(name, 'STS15 average: ', sum(scores) / len(scores))
