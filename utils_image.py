import torch
import torch.utils.data
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from torchvision import transforms
import csv
from PIL import Image
from functools import partial
import pandas


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data, attack_label, attr, data_path, batch_size):

    ### data load
    channel = 3
    hideen = 768
    if "_" in attr:
        attr = attr.split("_")
    num_classes, shape_img, train, test = prepare_dataset(data, attack_label, attr, data_path)
    train_data = get_train(train, batch_size)
    test_data = get_test(test, batch_size)
   
    num_classes1 = num_classes[0]
    if data == 'utkface':
        num_classes2 = num_classes[1]
    if data == 'celeba':
        num_classes2 = num_classes[1:]

    channel = channel 
    hideen =  hideen
    return train_data, test_data,  num_classes1, num_classes2, channel, hideen


def get_train(train_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               )
    return train_loader 


def get_test(test_dataset, batch_size):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              drop_last=True,
                                              shuffle=False)
    return test_loader 


def prepare_dataset(dataset, attack_label, attr, root):
    num_classes, dataset = get_model_dataset(dataset, attack_label=attack_label, attr=attr, root=root)
    length = len(dataset)
    each_length = length//10
    shape_img = (32, 32)
    train,  test,_ = torch.utils.data.random_split(dataset, [8*each_length,  2*each_length, len(dataset)-(each_length*10)], generator=torch.Generator().manual_seed(1234))
    return num_classes, shape_img, train, test


def get_model_dataset(dataset_name, attack_label,  attr, root):
    if dataset_name.lower() == "utkface":
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((64, 64)),
            # transforms.RandomHorizontalFlip(p=1)
            # transforms.ColorJitter(saturation=(2, 2))
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
            ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)

    if dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            num_classes = [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2]
            # Male
            attr_list = [[18, 21, 31], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13],
                         [14], [15], [16], [17], [19], [20], [22], [23], [24], [25], [26], [27], [28], [29], [30], [32], [33], [34], [35], [36], [37], [38],
                         [39]]
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((64, 64)),
            transforms.Resize((32, 32)),
            # transforms.RandomCrop((64, 32), padding=(0, 0, 32, 0),  fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)

    return num_classes, dataset


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        self.files = os.listdir(self.processed_path)
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            txt_file_path = os.path.join(self.processed_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')
        # print('000image', image)

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target


class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"
    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))
        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            final_attr = 0
            for i in range(len(nums)):
                final_attr += 2 ** i * self.attr[index][nums[i]]
            # print('final_attr', final_attr.numpy())
            target.append(int(final_attr))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def split_data(X):
    X = X.to(device)
    X_1 = X.to(device).clone().detach().to(device)
    X_2 = X.to(device).clone().detach().to(device)

    index1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).to(device)
    index2 = torch.tensor([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]).to(device)
    X_1 = X_1.index_fill(3, index1, 0).to(device)
    X_2 = X_2.index_fill(3, index2, 0).to(device)
    return X_1, X_2



def records_path(model, dataset,acti, num_cutlayer, batch_size, epochs):
    csvFile_1 = open(f'Results/{dataset}/{model}/VFL_client1_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results/{dataset}/{model}/VFL_client2_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    return writer_1, writer_2


def records_path_pruning(pp, model, dataset,acti, num_cutlayer, batch_size, epochs):
    csvFile_1 = open(f'Results/{dataset}/{model}/p{pp}/VFL_client1_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results/{dataset}/{model}/p{pp}/VFL_client2_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    return writer_1, writer_2


def records_path_noise(noise_scale, model, dataset,acti, num_cutlayer, batch_size, epochs):
    csvFile_1 = open(f'Results/{dataset}/{model}/n{noise_scale}/VFL_client1_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results/{dataset}/{model}/n{noise_scale}/VFL_client2_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    return writer_1, writer_2
