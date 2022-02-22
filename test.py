import os
from data_process import data_process
from datasets import load_dataset, load_metric, Dataset
from trainer import extract_qa_manager

if __name__ == '__main__':
    # 准备数据
    # if False:
    if os.path.exists(u'D:'):
        dataset_model_vali, dataset_rule_vali = data_process('vali')
        dataset_model_vali = {key: value[:2] for key, value in dataset_model_vali.items()}
        dataset_model_vali = Dataset.from_dict(dataset_model_vali)
        # dataset_model_test = dataset_model_vali
        dataset_model_test, dataset_rule_test = data_process('test')
        dataset_model_test = {key: value[:2] for key, value in dataset_model_test.items()}
        dataset_model_test = Dataset.from_dict(dataset_model_test)
        datasets_model = {'train': dataset_model_vali, 'validation': dataset_model_vali, 'test': dataset_model_vali}
    else:
        dataset_model_train, dataset_rule_train = data_process('train')
        dataset_model_train = Dataset.from_dict(dataset_model_train)
        dataset_model_vali, dataset_rule_vali = data_process('vali')
        dataset_model_vali = Dataset.from_dict(dataset_model_vali)
        dataset_model_test, dataset_rule_test = data_process('test')
        dataset_model_test = Dataset.from_dict(dataset_model_test)
        datasets_model = {'train': dataset_model_train, 'validation': dataset_model_vali, 'test': dataset_model_test}

    # 获取模型结果
    model_pred_result, output_dir = extract_qa_manager(datasets_model)

