import os
import argparse
import cv2 as cv

from pathlib import Path
from onnxruntime import InferenceSession
from models.thrid_party.paddleocr.infer import predict_det, predict_rec
from models.thrid_party.paddleocr.infer import utility

from models.utils import mix_inference
from models.ocr_model.utils.to_katex import to_katex
from models.ocr_model.utils.inference import inference as latex_inference

from models.ocr_model.model.TexTeller import TexTeller
from models.det_model.inference import PredictConfig
from happytransformer import HappyTextToText, TTSettings
import tempfile

import re 
from happytransformer import HappyTextToText
from pdf_png import trim_pdf_image_to_png
# Initialize the grammar correction model
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
def extract_math_expressions(text):
    math_expressions = []
    cleaned_text = text

    # Regex to find math expressions
    pattern = r'(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])'
    matches = re.finditer(pattern, cleaned_text)

    for i, match in enumerate(matches):
        math_expr = match.group(0)
        math_expressions.append(math_expr)
                    # Replace math expression with a unique marker
        cleaned_text = cleaned_text.replace(math_expr, f"@@math_{i}@@", 1)

    return cleaned_text.strip(), math_expressions

            # Function to restore math expressions after grammar correction
def restore_math_expressions(text, math_expressions):
    # Debugging at the start to check inputs
    # import pdb; pdb.set_trace()
    
    for i, math_expr in enumerate(math_expressions):
        # Ensure the exact marker exists in text
        marker = f"math_{i}@@"
        
        # Confirm if the marker exists in the text before replacement
        if marker in text:
            text = text.replace(marker, math_expr)
        else:
            print(f"Marker {marker} not found in text.")  # Debugging info

    return text.strip()


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', 
        type=str, 
        required=True,
        help='path to the input image'
    )
    parser.add_argument(
        '--inference-mode', 
        type=str,
        default='cpu',
        help='Inference mode, select one of cpu, cuda, or mps'
    )
    parser.add_argument(
        '--num-beam', 
        type=int,
        default=1,
        help='number of beam search for decoding'
    )
    parser.add_argument(
        '-mix', 
        action='store_true',
        help='use mix mode'
    )
    parser.add_argument(
        '-grammar', 
        action='store_true',
        help='correction of grammatical error'
    )
    
    args = parser.parse_args()
    latex_rec_model = TexTeller.from_pretrained('checkpoint/checkpoint_for_text')
    
    tokenizer = TexTeller.get_tokenizer()#'/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/train_result/test2/checkpoint-6000')

    temp_dir = '/home/santoshpalaskar77/IE_643/TexTeller/src/tempt_folder/'
    img_path = args.img
    os.makedirs(temp_dir, exist_ok=True)
    if img_path.lower().endswith('.pdf'):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_filename = 'output_image.png'
            output_path = os.path.join(temp_dir, output_filename)
            png_file_path = trim_pdf_image_to_png(img_path, output_path)
            img = cv.imread(png_file_path)
    else:
        img = cv.imread(img_path)
    if not args.mix:
        res = latex_inference(latex_rec_model, tokenizer, [img], args.inference_mode, args.num_beam)
        res = to_katex(res[0])
        if not args.grammar:
            print(res)
        else:
            cleaned_text, math_expressions = extract_math_expressions(res)

            # Prepend grammar context
            grammar_context = "grammar: " + cleaned_text + "."

            # Run the grammar correction model, ensuring placeholders are not altered
            corrected_text = happy_tt.generate_text(grammar_context).text.strip()
            
            # Restore the mathematical expressions
            final_text = restore_math_expressions(corrected_text, math_expressions)

            # Trim any unwanted characters from the final text
            final_text1 = final_text.rstrip('@')
            print("Final Text:", final_text1)
           
           
           
           
           
    else:
        infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
        latex_det_model = InferenceSession("./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")

        use_gpu = args.inference_mode == 'cuda'
        SIZE_LIMIT = 20 * 1024 * 1024
        det_model_dir =  "./models/thrid_party/paddleocr/checkpoints/det/default_model.onnx"
        rec_model_dir =  "./models/thrid_party/paddleocr/checkpoints/rec/default_model.onnx"
        # The CPU inference of the detection model will be faster than the GPU inference (in onnxruntime)
        det_use_gpu = False
        rec_use_gpu = use_gpu and not (os.path.getsize(rec_model_dir) < SIZE_LIMIT)

        paddleocr_args = utility.parse_args()
        paddleocr_args.use_onnx = True
        paddleocr_args.det_model_dir = det_model_dir
        paddleocr_args.rec_model_dir = rec_model_dir

        paddleocr_args.use_gpu = det_use_gpu
        detector = predict_det.TextDetector(paddleocr_args)
        paddleocr_args.use_gpu = rec_use_gpu
        recognizer = predict_rec.TextRecognizer(paddleocr_args)
        
        lang_ocr_models = [detector, recognizer]
        latex_rec_models = [latex_rec_model, tokenizer]
        res = mix_inference(img_path, infer_config, latex_det_model, lang_ocr_models, latex_rec_models, args.inference_mode, args.num_beam)
        #print(res)
        if not args.grammar:
            print(res)
        else:
            # Extract math expressions
            cleaned_text, math_expressions = extract_math_expressions(res)

            # Prepend grammar context
            grammar_context = "grammar: " + cleaned_text + "."

            # Run the grammar correction model, ensuring placeholders are not altered
            corrected_text = happy_tt.generate_text(grammar_context).text.strip()
            # Restore the mathematical expressions
            final_text = restore_math_expressions(corrected_text, math_expressions)

            # Trim any unwanted characters from the final text
            final_text1 = final_text.rstrip('@')

            # Print each step for debugging
            # print('input:',res)
            # print("Masked Text:", grammar_context)
            # print("Corrected Text:", corrected_text)
            print("Final Text:", final_text1)
           
