from PIL import Image
from torch.utils.data import Dataset

class ASLDataset(Dataset):
    def __init__(self, df, transforms=None, load_mem=True, class_map=None):
        super().__init__()
        self.dataframe = df
        self.transforms = transforms
        self.class_map = class_map
        self.images = []
        self.load_mem = load_mem
        # En caso de load_mem=True, cargo las imágenes previamente para no cargarlas a la hora de entrenar
        # Esto puede dar problemas si la memoria disponible no es suficiente para cargar todas las imágenes
        # En ese caso sería recomendable cargar en el momento de __getitem__ (load_mem=False)
        if load_mem:
            for _, row in self.dataframe.iterrows():
                img_tensor = self.load_and_process(row["path"])
                self.images.append(img_tensor)
            assert len(self.images) == len(self.dataframe)

    def load_and_process(self, path):
        img_tensor = Image.open(path)
        img_tensor = self.transforms(img_tensor) if self.transforms else img_tensor
        return img_tensor

    def get_class(self, cl):
        return self.class_map[cl] if self.class_map else cl

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, ind):
        if self.load_mem:
            return self.images[ind], self.get_class(self.dataframe.iloc[ind]["class"])
        row = self.dataframe.iloc[ind]
        img_tensor = self.load_and_process(row["path"])
        return img_tensor, self.get_class(row["class"])
