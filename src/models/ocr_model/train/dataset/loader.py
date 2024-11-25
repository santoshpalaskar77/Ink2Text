from PIL import Image
from pathlib import Path
import datasets
import json


#DIR_URL = Path('/scratch/santosh/large_data/combined_data')

import os


DIR_URL =  Path('/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/dataset')
# e.g. DIR_URL = Path('/home/OleehyO/TeXTeller/src/models/ocr_model/train/dataset')


class LatexFormulas(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "latex_formula": datasets.Value("string")
            })
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dir_path = Path(dl_manager.download(str(DIR_URL)))
        assert dir_path.is_dir()

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'dir_path': dir_path,
                }
            )
        ]

    # def _generate_examples(self, dir_path: Path):
    #     images_path   = dir_path / 'images'
    #     formulas_path = dir_path / 'formulas.jsonl'

    #     img2formula = {}
    #     with formulas_path.open('r', encoding='utf-8') as f:
    #         for line in f:
    #             single_json = json.loads(line)
    #             img2formula[single_json['img_name']] = single_json['formula']

    #     for img_path in images_path.iterdir():
    #         if img_path.suffix not in ['.jpg', '.png']:
    #             continue
    #         yield str(img_path), {
    #             "image": Image.open(img_path),
    #             "latex_formula": img2formula[img_path.name]
    #         }
    from PIL import Image

    MIN_HEIGHT = 12
    MIN_WIDTH = 30

    def _generate_examples(self, dir_path: Path):
        MIN_HEIGHT = 12
        MIN_WIDTH = 30
        images_path = dir_path / 'images'
        formulas_path = dir_path / 'formulas.jsonl'

        img2formula = {}
        with formulas_path.open('r', encoding='utf-8') as f:
            for line in f:
                single_json = json.loads(line)
                img2formula[single_json['img_name']] = single_json['formula']

        for img_path in images_path.iterdir():
            if img_path.suffix not in ['.jpg', '.png']:
                continue

            # Open the image and check dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                if height < MIN_HEIGHT or width < MIN_WIDTH:
                    print(f"Skipping image {img_path.name} due to small dimensions ({width}x{height})")
                    continue

            # Yield the valid image and its corresponding formula
            yield str(img_path), {
                "image": img,
                "latex_formula": img2formula[img_path.name]
            }