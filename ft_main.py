import fasttext
import aux_data

def write_data( fname, txts, ys ):
    with open( fname, 'w', encoding='utf-8') as fout:
        for txt,y  in zip( txts, ys):
            #TODO tokenize first?
            txt = txt.replace('\n', ' ')
            label = '__label__{}'.format(y)
            fout.write(label+' '+txt+'\n')


def convert_data( dir = './data/ag_news_csv/', txt_cols = (1,2)):
    train_file = dir + 'train.csv'
    test_file = dir + 'test.csv'
    txt_train, y_train = aux_data.load_csv(train_file, txt_cols)
    txt_test, y_test = aux_data.load_csv(test_file, txt_cols)
    write_data( 'data/train.txt', txt_train, y_train )
    write_data('data/test.txt', txt_test, y_test)



if __name__ == '__main__':
    convert_data()
    print('convert data done.')
    classifier = fasttext.supervised('data/train.txt', 'model')
    # classifier = fasttext.supervised('data/train.txt', 'model', label_prefix='__label__')
    result = classifier.test('data/test.txt')
    # print('acc:', result.accuracy)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)