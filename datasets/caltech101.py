import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from collections import defaultdict
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import pdb
import random

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # self.tracker = self.get_tracker(train)
        
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        
        self.tracker = self.get_tracker(train)
        # tracker = self.tracker
        # self.image_pair = self.make_image_pair(train, tracker)

        super().__init__(train_x=train, val=val, test=test)
    
    def get_tracker(self, train):
        tracker = defaultdict(list)
        for i in range(len(train)):   
            label = train[i].label        # 0~50
            impath = train[i].impath
            tracker[label].append(impath)
        return tracker
    

    # # make the image pair
    # def make_image_pair(self, train, tracker):
    #     tracker_pair = defaultdict(list)
    #     for i in range(len(train)):   
    #         labels = [j for j in range(0,int(len(self.tracker)))]
    #         label_1 = train[i].label
    #         labels.remove(label_1)
    #         label_2 = random.choice(labels)          # make a label pair (not same label, differnet two label)
            
    #         impath_1 = train[i].impath
    #         impath_2 = random.choice(tracker[label_2]) # choose the sample which is different label

    #         label = [label_1, label_2]
    #         impath = [impath_1, impath_2]
    #         tracker_pair[i].append([label, impath])
    #     return tracker_pair
    