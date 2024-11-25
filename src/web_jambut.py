import os
import io
import re
import base64
import tempfile
import shutil
import streamlit as st

from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton
from onnxruntime import InferenceSession
from models.thrid_party.paddleocr.infer import predict_det, predict_rec
from models.thrid_party.paddleocr.infer import utility

from models.utils import mix_inference
from models.det_model.inference import PredictConfig

from models.ocr_model.model.TexTeller import TexTeller
from models.ocr_model.utils.inference import inference as latex_recognition
from models.ocr_model.utils.to_katex import to_katex

import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF


st.set_page_config(
    page_title="Ink2Text",  # Page title
    page_icon="✏️"  # Page icon
)

html_string = '''
    <div style="background: linear-gradient(to bottom, #f0f8ff, #e6f7ff); padding: 20px; border-radius: 15px;">
        <h1 style="color: #1a1a1a; text-align: center; font-family: 'Arial', sans-serif; font-weight: bold;">
            <img src="https://github.com/user-attachments/assets/52adcbf8-e1d2-41e2-96b0-cb491df3fa83" width="90" style="vertical-align: middle; margin-right: 15px;">
            𝐼𝑛𝑘<sup style="color: #ff6347;">2</sup>𝑇𝑒𝑥𝑡
            <img src="https://github.com/user-attachments/assets/52adcbf8-e1d2-41e2-96b0-cb491df3fa83" width="90" style="vertical-align: middle; margin-left: 15px;">
        </h1>
        <h3 style="color: gray; text-align: center; font-family: 'Verdana', sans-serif;">
            Developed by <span style="color: #ff6347;">Team JAMBUT</span>
        </h3>
    </div>
'''


suc_gif_html = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://github.com/user-attachments/assets/29d0164e-142c-4695-b372-c037be092d7b" width="50">
        <img src="https://github.com/user-attachments/assets/29d0164e-142c-4695-b372-c037be092d7b" width="50">
        <img src="https://github.com/user-attachments/assets/29d0164e-142c-4695-b372-c037be092d7b" width="50">
    </h1>
'''

fail_gif_html = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
    </h1>
'''
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import streamlit as st
import pdfkit
from pdf2image import convert_from_path

import os
# Function to save markdown content as PDF
from weasyprint import HTML
import tempfile
import numpy as np 
import re 
from happytransformer import HappyTextToText
from pdf_png import trim_pdf_image_to_png

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

@st.cache_resource
# Initialize the grammar correction model
def trim_pdf_image_to_png(pdf_path, output_path):
    # Convert the PDF to images
    images = convert_from_path(pdf_path)

    # Select the first (and only) page
    image = images[0]

    # Convert the image to an array
    image_array = np.array(image)

    # Convert to grayscale and create a binary mask (where white pixels are True)
    gray_image = np.mean(image_array, axis=2)  # Convert to grayscale
    binary_mask = gray_image > 240  # Adjust threshold as needed for white

    # Get coordinates of non-white pixels
    coords = np.argwhere(~binary_mask)

    # Crop the image if there are non-white pixels
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        trimmed_image = image.crop((x0, y0, x1 + 1, y1 + 1))
    else:
        trimmed_image = image  # No cropping needed

    # Save the trimmed image as a PNG file
    trimmed_image.save(output_path, format='PNG')
    return output_path
def extract_math_expressions(text):
    math_expressions = []
    cleaned_text = text

    # Regex to find math expressions
    pattern = r'(\$.*?\$|\n?\\\(.*?\)|\n?\\\[.*?\\\]|\n?\\begin\{.*?\}.*?\\end\{.*?\}|\n?\\frac\{.*?\}\{.*?\})'


    #pattern = r'(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])'
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
        # Define the three possible marker formats
        marker_full = f"@@math_{i}@@"
        marker_suffix = f"math_{i}@@"
        marker_prefix = f"@@math_{i}"
        
        # Check if any of the markers exist in the text
        if marker_full in text:
            text = text.replace(marker_full, math_expr)
        elif marker_suffix in text:
            text = text.replace(marker_suffix, math_expr)
        elif marker_prefix in text:
            text = text.replace(marker_prefix, math_expr)
        else:
            print(f"No matching marker found for math_{i} in text.")  # Debugging info

    return text.strip()
def correct_text_with_math(katex_res2, grammar_model):
    """
    Processes text with math expressions by applying grammar correction while preserving math expressions.

    Parameters:
    - katex_res (str): The original text containing both text and math expressions.
    - grammar_model: A grammar correction model with a `generate_text()` method.

    Returns:
    - str: The final corrected text with math expressions restored.
    """
    # Step 1: Extract text and math expressions
    cleaned_text, math_expressions = extract_math_expressions(katex_res2)

    # Step 2: Prepare grammar correction context
    grammar_context = "grammar: " + cleaned_text 

    # Step 3: Run grammar correction while preserving math placeholders
    corrected_text = grammar_model.generate_text(grammar_context).text.strip()

    # Step 4: Restore math expressions into the corrected text
    final_text = restore_math_expressions(corrected_text, math_expressions)

    # Step 5: Trim any unwanted characters from the final text
    final_text1 = final_text.rstrip('@')
    
    return final_text1

def get_texteller_text(use_onnx, accelerator):
    return TexTeller.from_pretrained('/scratch/santosh/ink2text/train_result/text_data_checkpoint/checkpoint-23340', use_onnx=use_onnx, onnx_provider=accelerator)

def get_texteller_formula(use_onnx, accelerator):
    return TexTeller.from_pretrained(os.environ['CHECKPOINT_DIR'], use_onnx=use_onnx, onnx_provider=accelerator)

def get_texteller_for_text(use_onnx, accelerator):
    return TexTeller.from_pretrained('/scratch/santosh/ink2text/train_result/test0_for_text/checkpoint-10500', use_onnx=use_onnx, onnx_provider=accelerator)

@st.cache_resource
def get_tokenizer():
    return TexTeller.get_tokenizer(os.environ['TOKENIZER_DIR'])

@st.cache_resource
def get_det_models(accelerator):
    infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
    latex_det_model = InferenceSession(
        "./models/det_model/model/rtdetr_r50vd_6x_coco.onnx", 
        providers=['CUDAExecutionProvider'] if accelerator == 'cuda' else ['CPUExecutionProvider']
    )
    return infer_config, latex_det_model

@st.cache_resource()
def get_ocr_models(accelerator):
    use_gpu = accelerator == 'cuda'

    SIZE_LIMIT = 20 * 1024 * 1024
    det_model_dir = "./models/thrid_party/paddleocr/checkpoints/det/default_model.onnx"
    rec_model_dir = "./models/thrid_party/paddleocr/checkpoints/rec/default_model.onnx"
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
    return [detector, recognizer]

def get_image_base641(img_file):
    buffered = io.BytesIO()
    img_file.seek(0)
    img = Image.open(img_file)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
def get_image_base64(img):
    # Create a BytesIO buffer
    buffered = io.BytesIO()
    # Save the image in PNG format to the buffer
    img.save(buffered, format="PNG")
    # Return the base64 encoded image as a string
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def on_file_upload():
    st.session_state["UPLOADED_FILE_CHANGED"] = True

def change_side_bar():
    st.session_state["CHANGE_SIDEBAR_FLAG"] = True

if "start" not in st.session_state:
    st.session_state["start"] = 1
    st.toast('Hooray!', icon='🎉')

if "UPLOADED_FILE_CHANGED" not in st.session_state:
    st.session_state["UPLOADED_FILE_CHANGED"] = False

if "CHANGE_SIDEBAR_FLAG" not in st.session_state:
    st.session_state["CHANGE_SIDEBAR_FLAG"] = False

if "INF_MODE" not in st.session_state:
    st.session_state["INF_MODE"] = "Formula and Text recognition"


##############################     <sidebar>    ##############################

with st.sidebar:
    num_beams = 1

    st.markdown("# 🛠️ Config")
    st.markdown("")

    inf_mode = st.selectbox(
        "Inference mode",
        ("Formula recognition", "Text recognition", "Formula and Text recognition"),
        on_change=change_side_bar
    )

    num_beams = st.number_input(
        'Number of beams',
        min_value=1,
        max_value=20,
        step=1,
        on_change=change_side_bar
    )

    accelerator = st.radio(
        "Accelerator",
        ("cpu", "cuda"),
        on_change=change_side_bar
    )

    st.markdown("## Seedup")
    use_onnx = st.toggle("ONNX Runtime ")



##############################     </sidebar>    ##############################


################################     <page>    ################################

texteller = get_texteller_for_text(use_onnx, accelerator)
tokenizer = get_tokenizer()
latex_rec_models = [texteller, tokenizer]

if inf_mode == "Text recognition":
    texteller = get_texteller_text(use_onnx, accelerator)
    latex_rec_models = [texteller, tokenizer]
    infer_config, latex_det_model = get_det_models(accelerator)
    lang_ocr_models = get_ocr_models(accelerator)
if inf_mode == "Formula recognition":
    texteller = get_texteller_formula(use_onnx, accelerator)
    latex_rec_models = [texteller, tokenizer]
    infer_config, latex_det_model = get_det_models(accelerator)
    lang_ocr_models = get_ocr_models(accelerator)

st.markdown(html_string, unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    " ",
    type=['jpg', 'png','pdf'],
    on_change=on_file_upload
)

paste_result = pbutton(
    label="📋 Paste an image",
    background_color="#5BBCFF",
    hover_background_color="#3498db",
)
st.write("")

if st.session_state["CHANGE_SIDEBAR_FLAG"] == True:
    st.session_state["CHANGE_SIDEBAR_FLAG"] = False
elif uploaded_file or paste_result.image_data is not None:
    if st.session_state["UPLOADED_FILE_CHANGED"] == False and paste_result.image_data is not None:
        uploaded_file = io.BytesIO()
        paste_result.image_data.save(uploaded_file, format='PNG')
        uploaded_file.seek(0)

    if st.session_state["UPLOADED_FILE_CHANGED"] == True:
        st.session_state["UPLOADED_FILE_CHANGED"] = False



if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    
    # Handle PDF input
    if uploaded_file.type == 'application/pdf':
        pdf_path = os.path.join(temp_dir, 'uploaded_file.pdf')
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        png_path = os.path.join(temp_dir, 'trimmed_image.png')
        trimmed_png_path = trim_pdf_image_to_png(pdf_path, png_path)
        img = Image.open(trimmed_png_path)  # Load the trimmed image as an Image object
        img.save("q1.jpg")
        img_base64 = get_image_base64(img)
    # Handle image input (JPG/PNG)
    else:
        img = Image.open(uploaded_file)  # Load directly as an Image object
        img_base64 = get_image_base641(uploaded_file)
    png_file_path = os.path.join(temp_dir, 'image.png')
    img.save(png_file_path, 'PNG')
    # Convert the image to base64
    with st.container(height=300):

        st.markdown(f"""
        <style>
        .centered-container {{
            text-align: center;
        }}
        .centered-image {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-height: 350px;
            max-width: 100%;
        }}
        </style>
        <div class="centered-container">
            <img src="data:image/png;base64,{img_base64}" class="centered-image" alt="Input image">
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
    .centered-container {{
        text-align: center;
    }}
    </style>
    <div class="centered-container">
        <p style="color:gray;">Input image ({img.height}✖️{img.width})</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    with st.spinner("Predicting..."):
        if inf_mode == "Formula recognition":
            TexTeller_result = latex_recognition(
                texteller,
                tokenizer,
                [png_file_path],
                accelerator=accelerator,
                num_beams=num_beams
            )[0]
            katex_res = to_katex(TexTeller_result)
        elif inf_mode == "Text recognition":
            TexTeller_result = latex_recognition(
                texteller,
                tokenizer,
                [png_file_path],
                accelerator=accelerator,
                num_beams=num_beams
            )[0]
            katex_res = to_katex(TexTeller_result)
        elif inf_mode == "Formula and Text recognition":
            TexTeller_result = latex_recognition(
                texteller,
                tokenizer,
                [png_file_path],
                accelerator=accelerator,
                num_beams=num_beams
            )[0]
            katex_res = to_katex(TexTeller_result)
        else:
            katex_res = mix_inference(png_file_path, infer_config, latex_det_model, lang_ocr_models, latex_rec_models, accelerator, num_beams)

        st.success('Completed!', icon="✅")
        st.markdown(suc_gif_html, unsafe_allow_html=True)
        st.text_area(":blue[***  𝑃r𝑒d𝑖c𝑡e𝑑 𝑓o𝑟m𝑢l𝑎  ***]", katex_res, height=150)

        if inf_mode == "Formula recognition":
            katex_res = '$' + katex_res + '$'
            st.markdown(katex_res)
        elif inf_mode == "Text recognition":
           # katex_res = katex_res.replace(r"\(", "$").replace(r"\)", "$")
            st.markdown(katex_res)
            st.markdown("Grammatically correct sentence")
            katex_res = correct_text_with_math(katex_res, happy_tt)
            st.markdown(katex_res)
            
        elif inf_mode == "Formula and Text recognition":
            katex_res1 = katex_res.replace(r"\(", "$").replace(r"\)", "$").replace(r"\\", "  \n")
            st.markdown(katex_res1)
            if st.button('Correct the Grammar'):
                # Perform grammatical correction using your function
                katex_res1 = correct_text_with_math(katex_res1, happy_tt)
                
                # Show the corrected result
                st.markdown(katex_res1)

        elif inf_mode == "Paragraph recognition":
            mixed_res = re.split(r'(\$\$.*?\$\$)', katex_res)
            for text in mixed_res:
                if text.startswith('$$') and text.endswith('$$'):
                    st.latex(text[2:-2])
                else:
                    st.markdown(text)
        st.write("")
        st.write("")
        import subprocess
        import os
        

        katex_res1 = rf"{katex_res}"
        katex_res1 = str(katex_res1)
        # Define LaTeX content as a string
        latex_content = f"""
        \\documentclass{{article}}
        \\begin{{document}}
        {katex_res1}
        \\end{{document}}
        """
        
        # Define file paths
        tex_file_path = "/home/santoshpalaskar77/IE_643/TexTeller/src/path_to_pdf/output.tex"
        pdf_output_dir = os.path.dirname(tex_file_path)  # Get the directory of the .tex file
        pdf_path = os.path.join(pdf_output_dir, "output.pdf")  # Define the output PDF path

        # Write LaTeX content to .tex file
        with open(tex_file_path, "w") as tex_file:
            tex_file.write(latex_content)  # Write the content of latex_content

        # Compile the .tex file to PDF using pdflatex and specify output directory
        try:
            subprocess.run(["pdflatex", "-output-directory", pdf_output_dir, tex_file_path], check=True)
            print("PDF generated successfully.")
        except subprocess.CalledProcessError as e:
            print("An error occurred while generating the PDF:", e)

        # Clean up auxiliary files
        for ext in ['aux', 'log']:
            aux_file = os.path.join(pdf_output_dir, f"output.{ext}")
            if os.path.exists(aux_file):
                os.remove(aux_file)

        # Initialize pdf_data
        pdf_data = None

        # Read the PDF file in binary mode
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            print("PDF read successfully.")  # For confirmation
        except FileNotFoundError:
            print("The PDF file does not exist.")

        # Create a download button for the PDF if it was read successfully
        if pdf_data:
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="output.pdf",
                mime="application/pdf"
            )
        else:
            st.error("PDF file could not be generated or found.")

        # Read the .tex file for download
        try:
            with open(tex_file_path, 'r') as tex_file:
                tex_data = tex_file.read()
                
            # Create a download button for the .tex file
            st.download_button(
                label="Download .tex File",
                data=tex_data,
                file_name="output.tex",
                mime="application/x-tex"
            )
        except FileNotFoundError:
            st.error("The .tex file does not exist.")

        with st.expander(":star2: :gray[Tips for better results]"):
            st.markdown('''
                * :mag_right: Use a clear and high-resolution image.
                * :scissors: Crop images as accurately as possible.
                * :jigsaw: Split large multi line formulas into smaller ones.
                * :page_facing_up: Use images with **white background and black text** as much as possible.
                * :book: Use a font with good readability.
            ''')
        shutil.rmtree(temp_dir)

    paste_result.image_data = None

################################     </page>    ################################
