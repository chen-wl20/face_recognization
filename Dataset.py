import torchvision.datasets as datasets
import os 
import numpy as np
from tqdm import tqdm
from torchvision import transforms
#generate triplet face dataset from folders of images

class TripletFaceDataset(datasets.ImageFolder):
    def __init__(self, dir, n_triplets, transform=None):
        super(TripletFaceDataset, self).__init__(dir, transform)
        self.n_triplets = n_triplets
        print("Generating {} triplets...".format(self.n_triplets))
        self.training_triplets = self.generate_triplets(self.imgs, self.n_triplets, len(self.classes))
    @staticmethod
    def generate_triplets(imgs, num_triplets, n_classes):
        def create_indices(_imgs):
            inds = dict()
            for idx, (img_path, label) in enumerate(_imgs):
                if label not in inds:
                    inds[label] = []
                inds[label].append(img_path)
            return inds
        triplets = []
        indices = create_indices(imgs)
        for x in tqdm(range(num_triplets)):
            c1 = np.random.randint(0, n_classes-1)
            c2 = np.random.randint(0, n_classes-1)
            while len(indices[c1]) < 2:
                c1 = np.random.randint(0, n_classes-1)

            while c1 == c2:
                c2 = np.random.randint(0, n_classes-1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            if len(indices[c2]) ==1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices[c2]) - 1)
            print(indices[c1], indices[c1][n1])
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
        return triplets
    def __getitem__(self, index):
        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """
        
            img = self.loader(img_path)
            return self.transform(img)
        a, p, n,c1,c2 = self.training_triplets[index]

        # transform images if required
        img_a, img_p, img_n = transform(a), transform(p), transform(n)
        return img_a, img_p, img_n,c1,c2
    def __len__(self):
        return len(self.training_triplets)


'''
dir = 'gray_set'
num_triplets = 20
dataset = TripletFaceDataset(dir, num_triplets, transform=transforms.ToTensor())
for step, (img_a, img_p, img_n, c1, c2) in enumerate(dataset):
    print(step, img_a.shape, img_p.shape, img_n.shape, c1, c2)
'''

class Pairs_Dataset(datasets.ImageFolder):
    def __init__(self, dir, n_pairs, transform=None):
        super(Pairs_Dataset, self).__init__(dir, transform)
        self.n_pairs = n_pairs
        print("Generating{} pairs...".format(self.n_pairs))
        self.training_pairs = self.generate_pairs(self.imgs, self.n_pairs, len(self.classes))
    @staticmethod
    def generate_pairs(imgs, num_pairs, n_classes):
        def create_indices(_imgs):
            inds = dict()
            for idx, (img_path, label) in enumerate(_imgs):
                if label not in inds:
                    inds[label] = []
                inds[label].append(img_path)
            return inds
        pairs = []
        indices = create_indices(imgs)
        for x in tqdm(range(num_pairs)):
            if x<num_pairs/2:
                c1 = np.random.randint(0, n_classes-1)
                while len(indices[c1])<2:
                    c1=np.random.randint(0, n_classes-1)
                c2 = c1
                if len(indices[c1])==2:
                    n1 = 0
                    n2 = 1
                else:
                    n1 = np.random.randint(0, len(indices[c1])-1)
                    n2 = np.random.randint(0, len(indices[c2])-1)
                    while n2==n1:
                        n2 = np.random.randint(0, len(indices[c2])-1)
            else:
                c1 = np.random.randint(0, n_classes-1)
                c2 = np.random.randint(0, n_classes-1)
                while c1==c2:
                    c2 = np.random.randint(0, n_classes-1)
                if len(indices[c1])==1:
                    n1 = 0
                else:
                    n1 = np.random.randint(0, len(indices[c1])-1)
                if len(indices[c2])==1:
                    n2 = 0
                else:
                    n2 = np.random.randint(0, len(indices[c2])-1)
                
            pairs.append([indices[c1][n1], indices[c2][n2], (c1==c2)])
        return pairs
    def __getitem__(self, index):
        def transform(img_path):
            img = self.loader(img_path)
            return self.transform(img)
        a, b, is_same = self.training_pairs[index]
        img_a, img_b = transform(a), transform(b)
        return img_a, img_b, is_same
    def __len__(self):
        return len(self.training_pairs)