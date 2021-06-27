import torch 

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transforms):
        self.imgs = imgs
        self.labels = labels 
        self.transforms = transforms 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        img = self.transforms(image=img)["image"]
        return {
            "image": torch.tensor(img, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float)
        }