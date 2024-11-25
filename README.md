# Ink2Text: Tool to convert Images/PDF to LaTeX document 

![ink2text](https://github.com/user-attachments/assets/c428e743-012b-45c6-82cf-fd8ff4d3d00d) 
## Installation Requirements
- It is recommended to create a conda environment with the following:
  - `Python 3.10.15`
  - The necessary Python packages can be installed from the `requirements.txt` file using,
    - `pip install -r requirements.txt`
## Instructions for Running the Code
- Must clone this repo to the desired machine
- Download the checkpoint from: [Checkpoints](https://drive.google.com/drive/folders/1z7MMYoh_bCl0YjJm_aPhvXf68aeOL-E0?usp=drive_link)
   - _checkpoint_for_text_ is for formula and text head: image contains formula and images. It can also be used for formula recognition, or we can use TexTeller, as it is mainly trained for formula recognition
   - _checkpoint_text_ is for text recognition
- Put checkpoints _checkpoint_for_text_ and _checkpoint_text_ in the directory
  ```
   src/checkpoints
   ```
- Download the weights [formula detection](https://drive.google.com/file/d/1aSn8nAjGXuYLI0bc1i9Qg1z4Tz0hbdY5/view?usp=drive_link) of the formula detection model to the `src/models/det_model/model/`directory
-  By default, Ink2Text use PaddleOCR model for recognition and detection
-  Download the PaddleOCR weights [Recognition](https://drive.google.com/file/d/1vEfregonJp4Wga8UWfYX7bPREXCmf7WL/view?usp=sharing) for the English text recognition 
-  Download the PaddleOCR weights [Detection](https://drive.google.com/file/d/1TqOceYNw2m92ayWg9DCCbGQ_VYV3-u5C/view?usp=drive_link) for the English text detection
-  Place the weights of the recognition/detection model in the `det/` or `rec/` directories within `src/models/third_party/paddleocr/checkpoints/`, and rename them to `default_model.onnx`.
## Running code using interface 
- Your directory should be **src**
  ```
   cd src
  ```
- Run the sh file in the terminal:
 ```
 ./start_web.sh
```
- You can open the link in the browser
  
![ink22text](https://github.com/user-attachments/assets/0777ef87-979d-4e11-86e9-d25257c9f1c3)

- You can upload the desired PNG/JGP/PDF file, which will produce the corresponding latex tex file and pdf file. It also corrects grammatical mistakes
## Running code using terminal 
- inference.py can be used for inference
- Use the finetuned texteller model by:
  ```
  latex_rec_model = TexTeller.from_pretrained('checkpoints/checkpoint_for_text')
  ```
  in the inference.py file
- Run the code in the terminal:
  ```
  python inference.py -img "/path/to/image.{jpg,png}"
  ```
- To apply grammatical correction, use the command
 ```
python inference.py -img "/path/to/image.{jpg,png}" -grammar
```

## 🏋️‍♂️ Training

### Dataset

We provide an example dataset in the `src/models/ocr_model/train/dataset/` directory, you can place your own images in the `images/` directory and annotate each image with its corresponding formula in `formulas.jsonl`.

After preparing your dataset, you need to **change the `DIR_URL` variable to your own dataset's path** in `**/train/dataset/loader.py`

### Training the Model

1. Modify `num_processes` in `src/train_config.yaml` to match the number of GPUs available for training (default is 1).
2. In the `src/` directory, run the following command:

   ```bash
   accelerate launch --config_file ./train_config.yaml -m models.ocr_model.train.train
   ```

You can set your own tokenizer and checkpoint paths in `src/models/ocr_model/train/train.py` (refer to `train.py` for more information). If you are using the same architecture and vocabulary as Ink2Text, you can also fine-tune Ink2Text's default weights with your own dataset.

