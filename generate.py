import pickle
import argparse
from pathlib import Path

import torch
import numpy as np

from util import set_seed, convert_layout_to_image
from model.layoutganpp import Generator
from data import get_dataset  # Ekledik


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='(Kullanılmıyor) batch size')
    parser.add_argument('-o', '--out_path', type=str,
                        default='output/generated_layouts.pkl',
                        help='output pickle path')
    parser.add_argument('--num_save', type=int, default=1,
                        help='kaç adet layout (sayfa) üretileceği')
    parser.add_argument('--num_elements', type=int, default=10,
                        help='her layoutta kaç panel üretileceği')
    parser.add_argument('--seed', type=int, help='manual seed')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt['args']

    # Dataset'i yükle ve num_classes al
    dataset = get_dataset(train_args['dataset'], 'test')
    num_label = dataset.num_classes

    latent_size = train_args['latent_size']

    # Load model
    netG = Generator(latent_size, num_label,
                     d_model=train_args['G_d_model'],
                     nhead=train_args['G_nhead'],
                     num_layers=train_args['G_num_layers'],
                     ).eval().to(device)
    netG.load_state_dict(ckpt['netG'])

    # Generate layouts
    results = []
    with torch.no_grad():
        for i in range(args.num_save):
            n = args.num_elements  # kaç panel olacak

            label = torch.randint(0, num_label, (1, n), device=device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
            z = torch.randn(1, n, latent_size, device=device)

            bbox = netG(z, label, padding_mask)
            b = bbox[0].cpu().numpy()
            l = label[0].cpu().numpy()

            convert_layout_to_image(
                b, l, dataset.colors, (120, 80)
            ).save(out_dir / f'generated_{i}.png')

            results.append((b, l))

    # Save as pickle
    with out_path.open('wb') as fb:
        pickle.dump(results, fb)

    print('Generated layouts are saved at:', args.out_path)


if __name__ == '__main__':
    main()
