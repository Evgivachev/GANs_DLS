from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np


class MyDataset(Dataset):
    def __init__(self, folder, rescaleSize):
        super().__init__()
        # список файлов для загрузки
        self.folder = folder
        self.files = sorted(list(folder.rglob('*.jpg')))
        # режим работы
        self.len_ = len(self.files)
        self.rescaleSize = rescaleSize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.len_

    @staticmethod
    def load_sample(file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        if len(x.shape) == 2:
            x = np.stack((x, x, x), axis=2)
        x = self.transform(x)
        return x

        # if self.mode == 'test':
        #    return x
        # else:
        #    label = self.labels[index]
        # label_id = self.label_encoder.transform([label])
        # y = label_id.item()
        # return x, y

    def _prepare_sample(self, image):
        image = image.resize((self.rescaleSize, self.rescaleSize))
        return np.array(image)
