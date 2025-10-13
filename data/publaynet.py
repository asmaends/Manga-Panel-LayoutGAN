from pathlib import Path
from pycocotools.coco import COCO

import torch
from torch_geometric.data import Data

from data.base import BaseDataset


class PubLayNet(BaseDataset):
    labels = ['panel']  # Sadece bir kategori: panel

    def __init__(self, split='train', transform=None):
        super().__init__('paneldata', split, transform)

    def download(self):
        super().download()

    def process(self):
        raw_dir = Path(self.raw_dir)
        train_list = []
        val_list = []

        for split_file in ['train', 'val']:
            data_list = []
            coco = COCO(raw_dir / f'{split_file}.json')
            for img_id in sorted(coco.getImgIds()):
                ann_img = coco.loadImgs(img_id)
                W = float(ann_img[0]['width'])
                H = float(ann_img[0]['height'])
                name = ann_img[0]['file_name']

                # NOT: Genişlik yüksekten büyükse de kabul ediyoruz!
                # if H < W:
                #     continue

                def is_valid(element):
                    x1, y1, width, height = element['bbox']
                    x2, y2 = x1 + width, y1 + height
                    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                        return False
                    if x2 <= x1 or y2 <= y1:
                        return False
                    return True

                elements = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
                _elements = list(filter(is_valid, elements))
                filtered = len(elements) != len(_elements)
                elements = _elements

                N = len(elements)
                # NOT: Panel sayısı sınırlarını kaldırıyoruz
                if N == 0:
                    continue

                boxes = []
                labels = []

                for element in elements:
                    x1, y1, width, height = element['bbox']
                    xc = x1 + width / 2.
                    yc = y1 + height / 2.
                    b = [xc / W, yc / H, width / W, height / H]
                    boxes.append(b)

                    labels.append(0)  # tümü 'panel'

                boxes = torch.tensor(boxes, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)

                data = Data(x=boxes, y=labels)
                data.attr = {
                    'name': name,
                    'width': W,
                    'height': H,
                    'filtered': filtered,
                    'has_canvas_element': False,
                }
                data_list.append(data)

            if split_file == 'train':
                train_list = data_list
            else:
                val_list = data_list

        # shuffle train verisi
        if len(train_list) == 0:
            raise RuntimeError("Train list boş, JSON dosyası uygun mu?")
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(train_list), generator=generator)
        train_list = [train_list[i] for i in indices]

        s = int(len(train_list) * .95)
        torch.save(self.collate(train_list[:s]), self.processed_paths[0])
        torch.save(self.collate(train_list[s:]), self.processed_paths[1])
        torch.save(self.collate(val_list), self.processed_paths[2])
