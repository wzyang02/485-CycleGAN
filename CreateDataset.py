class CreateDataset(Dataset):
    def __init__(self, datatype):
        self.dataPath_A = DATA_DIR+datatype+"A"
        self.dataPath_B = DATA_DIR+datatype+"B"
        self.transform =  A.Compose(
           [A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

        self.imageList_A = sorted([os.path.join(self.dataPath_A, f) for f in os.listdir(self.dataPath_A)])
        self.imageList_B = sorted([os.path.join(self.dataPath_B, f) for f in os.listdir(self.dataPath_B)])
        imageLen = min(len(self.imageList_A), 1000)
        imageLen = min(len(self.imageList_B), imageLen)
        self.imageList_A = self.imageList_A[:imageLen]
        self.imageList_B = self.imageList_B[:imageLen]
        self.imageCount_A = len(self.imageList_A)
        self.imageCount_B = len(self.imageList_B)

    def __len__(self):
        return max(len(self.imageList_A), len(self.imageList_B))

    def __getitem__(self, index):
        image_A = self.imageList_A[index % self.imageCount_A]
        image_B = self.imageList_B[index % self.imageCount_A]
        image_A = np.array(Image.open(image_A).convert("RGB"))
        image_B = np.array(Image.open(image_B).convert("RGB"))
        augmentations = self.transform(image=image_A, image0=image_B)
        image_A = augmentations["image"]
        image_B = augmentations["image0"]

        return image_A, image_B