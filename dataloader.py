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
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2, 0, 1)
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long)
        }