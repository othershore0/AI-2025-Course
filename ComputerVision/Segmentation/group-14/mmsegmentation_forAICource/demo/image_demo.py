# Copyright (c) OpenMMLab. All rights reserved.
# Altered by TonyYu

import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)

    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file,
        save_dir="result")
    images = [args.img , args.out_file]
    titles = ['Original', 'Crack Segmentation']
    fig, axes = plt.subplots(1, 2)
    for ax, img_path, title in zip(axes, images, titles):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    filen = args.out_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    comp='result/'+filen+'_comparison_result.png'
    plt.savefig(comp, dpi=1200)
    plt.show()


if __name__ == '__main__':
    main()
