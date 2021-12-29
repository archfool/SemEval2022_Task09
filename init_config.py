import os

if os.path.exists(u'D:'):
    root_dir = u'D:\\'
elif os.path.exists(u'/media/archfool/data'):
    root_dir = u'/media/archfool/data/'
elif os.path.exists(u'/mnt/hgfs/'):
    root_dir = u'/mnt/hgfs/'
else:
    root_dir = None
    print("root_dir is invalid !!!")

src_root_dir = os.path.join(root_dir, u'src')
data_root_dir = os.path.join(root_dir, u'data')

src_dir = os.path.join(src_root_dir, 'SemEval2022_Task09')
data_dir = os.path.join(data_root_dir, 'SemEval-2022', 'task9')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if __name__ == "__main__":
    import pickle
    from util_tools import logger

    logger.info("BEGIN")
    corpus_list = []
    f = open(corpus_embd_file_path, "rb")
    i = 0
    while True:
        try:
            corpus_list.append(pickle.load(f))
            i += 1
            # print(i)
        except:
            f.close()
            break
    logger.info("END")
