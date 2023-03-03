from mzz import Mzz
import pickle
import numpy as np
import pandas as pd
from acvl_utils.array_manipulation.slicer import slicer
from os.path import join
import random
from skimage.transform import resize


class DiademIslets:
    def __init__(self, image_dir, split_filepath, labels_filepath, islets_filepath, transform, min_islet_size=50**2, min_num_islets=4, num_islets=16, islet_crop_margin=0,
                 combine_stainings=True, stainings=("Glucagon", "Insulin"), resize=(128, 128), train=True):
        self.image_dir = image_dir
        self.train = train
        self.num_islets = num_islets
        self.islet_crop_margin = islet_crop_margin
        self.stainings = stainings
        self.resize = resize
        self.transform = transform
        self.combine_stainings = combine_stainings

        if train:
            set_type = "train"
        else:
            set_type = "val"

        labels_csv = pd.read_csv(labels_filepath)
        original_names = labels_csv["slide_id"].values.tolist()
        label_mapping = labels_csv["label"].values.tolist()
        label_mapping = {original_names[i]: label_mapping[i] for i in range(len(original_names))}
        case_ids = labels_csv["case_id"].values.tolist()
        case_mapping = {original_names[i]: case_ids[i] for i in range(len(original_names))}
        case_ids = list(set(case_ids))

        split_csv = pd.read_csv(split_filepath)
        split = split_csv[set_type].values.tolist()

        with open(islets_filepath, 'rb') as handle:
            islets = pickle.load(handle)

        mzz_names = list(islets.keys())
        name_mapping = self.remap_names(mzz_names, original_names)
        stain_mapping = self.identify_staining(mzz_names)
        unordered_cases = [{"mzz_name": mzz_names[i], "original_name": name_mapping[mzz_names[i]], "staining": stain_mapping[mzz_names[i]], "label": label_mapping[name_mapping[mzz_names[i]]], "case_id": case_mapping[name_mapping[mzz_names[i]]]} for i in range(len(mzz_names))]
        unordered_cases, case_mapping, label_mapping = self.filter_by_split(unordered_cases, case_mapping, label_mapping, split)
        self.cases = self.sort_by_staining(islets, unordered_cases, case_ids)
        self.filter_by_islet_size(self.cases, min_islet_size)
        self.filter_by_num_islets(self.cases, min_num_islets)
        if self.combine_stainings:
            self.cases = self.filter_by_available_stainings(self.cases, stainings)
            self.length = len(self.cases)
        else:
            self.length = 0
            self.indices = {}
            for case_id, case in self.cases.items():
                for staining_index, staining in enumerate(case["stainings"].keys()):
                    self.indices[self.length+staining_index] = {"case_index": case_id, "staining": staining}
                self.length += len(case["stainings"])
        self.case_ids = list(self.cases.keys())

        self._resize_islets(self.cases)

        # with open("/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_stacked_islets/islets_resize_128.pkl", 'wb') as handle:
        #     pickle.dump(islets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index):
        islets = []
        if self.combine_stainings:
            case = self.cases[self.case_ids[index]]
            for staining in self.stainings:
                islets_staining = self.extract_random_islets(case, staining)
                islets.append(islets_staining)
            islets = np.concatenate(islets, axis=0)
        else:
            case, staining = self.cases[self.indices[index]["case_index"]], self.indices[index]["staining"]
            islets = self.extract_random_islets(case, staining)
        islets = np.transpose(islets, (3, 0, 1, 2))[np.newaxis, ...]
        item = {"data": islets}
        item = self.transform(**item)
        data = item["data"].squeeze(0)
        label = case["label"]
        # print("input: {}, target: {}".format(data.shape, label))
        return data, label

    def __len__(self):
        return self.length

    def remap_names(self, mzz_names, original_names):
        name_mapping = {}
        for mzz_name in mzz_names:
            original_name = self.remap_name(mzz_name, original_names)
            name_mapping[mzz_name] = original_name
        return name_mapping

    def remap_name(self, mzz_name, original_names):
        staining_name = mzz_name.split("-")[0]
        case = mzz_name.split("-")[1]
        if "SST" in staining_name:
            staining_name = "SST"
            case = mzz_name.split("-")[2]
        original_names_staining = [name for name in original_names if staining_name in name]
        original_name = [name for name in original_names_staining if case in name]

        if len(original_name) != 1:
            raise RuntimeError("Could not determine original name. MZZ name: {}, possible names: {}".format(mzz_name, original_name))

        original_name = original_name[0]
        return original_name

    def identify_staining(self, mzz_names):
        stainings = {"CD31": "CD31", "Glucagon": "Glucagon", "Insulin": "Insulin", "Perilipin": "Perilipin-1", "Somatostatin": "Somatostatin", "SST1": "Somatostatin", "Tubulin": "Tubulin-b3"}
        stain_mapping = {}
        for mzz_name in mzz_names:
            staining = mzz_name.split("-")[0]
            try:
                staining = stainings[staining]
            except:
                raise RuntimeError("Could not match staining: {}".format(staining))
            stain_mapping[mzz_name] = staining
        return stain_mapping

    def filter_by_split(self, cases, case_mapping, label_mapping, split):
        cases = [case for case in cases if case["original_name"] in split]
        case_mapping = {original_name: case_id for original_name, case_id in case_mapping.items() if original_name in split}
        label_mapping = {original_name: label for original_name, label in label_mapping.items() if original_name in split}
        return cases, case_mapping, label_mapping

    def sort_by_staining(self, islets, unordered_cases, case_ids):
        ordered_cases = {}
        for case_id in case_ids:
            cases = [case for case in unordered_cases if case_id == case["case_id"]]
            if cases:
                ordered_case = {"case_id": cases[0]["case_id"], "label": cases[0]["label"], "stainings": {}}
                for case in cases:
                    ordered_case["stainings"][case["staining"]] = {case["mzz_name"]: islets[case["mzz_name"]]}
                ordered_cases[case_id] = ordered_case
        return ordered_cases

    def filter_by_islet_size(self, cases, min_islet_size):
        for case_id, case in cases.items():
            for staining_name, case_staining_islets in case["stainings"].items():
                mzz_name = list(case_staining_islets.keys())[0]
                filtered_islets = []
                for islet in case_staining_islets[mzz_name]:
                    if islet["size"] >= min_islet_size:
                        filtered_islets.append(islet)
                case_staining_islets[mzz_name] = filtered_islets

    def filter_by_num_islets(self, cases, min_num_islets):
        for case_id, case in cases.items():
            stainings = {}
            for staining_name, case_staining_islets in case["stainings"].items():
                mzz_name = list(case_staining_islets.keys())[0]
                if len(case_staining_islets[mzz_name]) >= min_num_islets:
                    stainings[staining_name] = case_staining_islets
            case["stainings"] = stainings

    def filter_by_available_stainings(self, cases, stainings):
        filtered_cases = {}
        for case_id, case in cases.items():
            remove = False
            for staining in stainings:
                if staining not in case["stainings"]:
                    remove = True
            if not remove:
                filtered_cases[case_id] = case
        return filtered_cases

    # def extract_random_islets(self, case, staining):
    #     mzz_name = list(case["stainings"][staining].keys())[0]
    #     image = Mzz(path=join(self.image_dir, staining, mzz_name + ".mzz"))
    #     indices = np.random.randint(low=0, high=len(case["stainings"][staining][mzz_name]), size=self.num_islets)
    #     islets = []
    #     for index in indices:
    #         bbox = case["stainings"][staining][mzz_name][index]["bbox"]
    #         bbox[:, 0] -= self.islet_crop_margin
    #         bbox[:, 1] += self.islet_crop_margin
    #         islet = image[slicer(image, bbox)]
    #         islet = resize(islet, (*self.resize, 3), order=1, anti_aliasing=False, preserve_range=True)
    #         islets.append(islet)
    #     islets = np.asarray(islets)
    #     return islets

    def extract_random_islets(self, case, staining):
        mzz_name = list(case["stainings"][staining].keys())[0]
        indices = np.random.randint(low=0, high=len(case["stainings"][staining][mzz_name]), size=self.num_islets)
        islets = []
        for index in indices:
            islet = case["stainings"][staining][mzz_name][index]["islet"]
            islets.append(islet)
        islets = np.asarray(islets)
        return islets

    def _resize_islets(self, cases):
        for case_id, case in cases.items():
            for staining_name, staining_wsi in case["stainings"].items():
                mzz_name = list(staining_wsi.keys())[0]
                for islet in staining_wsi[mzz_name]:
                    islet["islet"] = resize(islet["islet"], (*self.resize, 3), order=1, anti_aliasing=False, preserve_range=True)
                    # islet["shape"] = islet["islet"].shape[:2]
                    # islet["size"] = np.prod(islet["shape"])


if __name__ == '__main__':
    import time

    # image_dir = "/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_WSI_mzz/train"
    # split_filepath = "/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_WSI/splits/splits_0.csv"
    # labels_filepath = "/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_WSI/label.csv"
    # islets_filepath = "/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_stacked_islets/islets_resize_128.pkl"
    #
    # dataset = DiademIslets(image_dir, split_filepath, labels_filepath, islets_filepath, transform=None, train=False)
    #
    # start_time = time.time()
    # for item in dataset:
    #     print("Time: ", time.time() - start_time)
    #     start_time = time.time()

    from augmentation.policies.diadem_islets import baseline_transform
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform
    from batchgenerators.transforms.utility_transforms import NumpyToTensor
    from batchgenerators.augmentations.normalizations import mean_std_normalization

    transform = Compose(
        [
            MeanStdNormalizationTransform((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            NumpyToTensor(keys="data")
        ]
    )

    # transform = mean_std_normalization

    data = np.random.rand(1, 3, 20, 128, 128)
    # transform = baseline_transform(0, 0)
    data = {"data": data}
    data = transform(**data)
    print(data["data"].shape)