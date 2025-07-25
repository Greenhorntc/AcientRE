from datasets import Dataset,DatasetDict
import numpy as np

def get_fewshot_dataset(dataset: Dataset, k: int, label_column: str = "label", seed: int = 42) -> Dataset:
    """
    从数据集中按类别采样k个样本
    :param dataset: Hugging Face Dataset 对象
    :param k: 每个类别的采样数量
    :param label_column: 数据集中标签列的名称（默认是"label"）
    :param seed: 随机种子
    :return: 采样后的子集 Dataset
    """
    # 获取所有标签和对应的索引
    labels = np.array(dataset[label_column])
    unique_labels = np.unique(labels)

    # 按标签分组索引
    indices_per_label = {label: np.where(labels == label)[0] for label in unique_labels}

    # 随机采样每个类别的样本
    np.random.seed(seed)
    selected_indices = []
    for label in unique_labels:
        indices = indices_per_label[label]
        if len(indices) < k:
            # 如果某个类别的样本不足k个，取全部
            selected = indices.tolist()
        else:
            # 随机选择k个样本
            selected = np.random.choice(indices, k, replace=False).tolist()
        selected_indices.extend(selected)

    # 打乱所有选中的样本（可选）
    np.random.shuffle(selected_indices)

    # 生成子集
    return dataset.select(selected_indices)


mascotds='../dataset/processed/re/mascot_v2/'
ds=DatasetDict.load_from_disk(mascotds)

train_dataset=ds['train']
k_shots = [1, 5, 10, 15]
fewshot_datasets = {
    k: get_fewshot_dataset(train_dataset, k, label_column="label", seed=42)
    for k in k_shots
}

# 例如，获取每个类别5个样本的子集
# fewshot_5 = fewshot_datasets[1]
# print(f"Few-Shot 5 样本数量: {len(fewshot_5)}")

for k in k_shots:
    fewshot_train=fewshot_datasets[k]
    fewshot_ds=DatasetDict({
    "train": fewshot_train,
    "validation": ds['validation'],
    "test": ds["test"]
    })
    savepath='../dataset/processed/re/fewshot/mascot{}'.format(k)
    fewshot_ds.save_to_disk(savepath)