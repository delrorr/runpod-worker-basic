import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
import tempfile
import numpy as np
from PIL import Image
import random
import json
import base64
from io import BytesIO

MODEL_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"

# Initialize model (you might want to move this to a separate initialization function)
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
text_to_video_pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
image_to_video_pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

for pipe in [text_to_video_pipe, image_to_video_pipe]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    pipe.to("cuda")

# Constants
MOD_VALUE = 32
DEFAULT_H_SLIDER_VALUE = 896
DEFAULT_W_SLIDER_VALUE = 896
NEW_FORMULA_MAX_AREA = 720 * 1024
SLIDER_MIN_H, SLIDER_MAX_H = 256, 1024
SLIDER_MIN_W, SLIDER_MAX_W = 256, 1024
MAX_SEED = np.iinfo(np.int32).max
FIXED_FPS = 24
MIN_FRAMES_MODEL = 25
MAX_FRAMES_MODEL = 193

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature"

def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area, min_slider_h, max_slider_h, min_slider_w, max_slider_w, default_h, default_w):
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w
    aspect_ratio = orig_h / orig_w

    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))
    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)

    new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))

    return new_h, new_w

def base64_to_pil(image_base64):
    """Convert base64 image string to PIL Image"""
    if image_base64.startswith('data:image'):
        # Remove data URL prefix if present
        image_base64 = image_base64.split(',', 1)[1]
    
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data)).convert('RGB')

def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_video_from_event(event):
    """
    Generate video from event JSON payload
    Expected event structure:
    {
        "input_image": "base64_string" (optional),
        "prompt": "text prompt",
        "height": 896,
        "width": 896,
        "negative_prompt": "negative text",
        "duration_seconds": 2.0,
        "guidance_scale": 0.0,
        "steps": 4,
        "seed": 42,
        "randomize_seed": false
    }
    """
    # Parse event data
    if isinstance(event, str):
        event_data = json.loads(event)
    else:
        event_data = event
    
    # Extract parameters with defaults
    input_image_base64 = event_data.get('input_image')
    prompt = event_data.get('prompt', default_prompt_i2v)
    height = event_data.get('height', DEFAULT_H_SLIDER_VALUE)
    width = event_data.get('width', DEFAULT_W_SLIDER_VALUE)
    negative_prompt = event_data.get('negative_prompt', default_negative_prompt)
    duration_seconds = event_data.get('duration_seconds', 2.0)
    guidance_scale = event_data.get('guidance_scale', 0.0)
    steps = event_data.get('steps', 4)
    seed = event_data.get('seed', 42)
    randomize_seed = event_data.get('randomize_seed', False)
    
    # Convert base64 to PIL image if provided
    input_image_pil = None
    if input_image_base64:
        try:
            input_image_pil = base64_to_pil(input_image_base64)
            # Auto-calculate dimensions if image is provided
            height, width = _calculate_new_dimensions_wan(
                input_image_pil, MOD_VALUE, NEW_FORMULA_MAX_AREA,
                SLIDER_MIN_H, SLIDER_MAX_H, SLIDER_MIN_W, SLIDER_MAX_W,
                height, width
            )
        except Exception as e:
            print(f"Error processing image: {e}")
            # Fall back to provided dimensions
    
    # Ensure dimensions are multiples of MOD_VALUE
    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    # Calculate number of frames
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    
    # Handle seed
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    
    # Generate video
    if input_image_pil is not None:
        # Resize input image to target dimensions
        resized_image = input_image_pil.resize((target_w, target_h))
        with torch.inference_mode():
            output_frames_list = image_to_video_pipe(
                image=resized_image, prompt=prompt, negative_prompt=negative_prompt,
                height=target_h, width=target_w, num_frames=num_frames,
                guidance_scale=float(guidance_scale), num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(current_seed)
            ).frames[0]
    else:
        with torch.inference_mode():
            output_frames_list = text_to_video_pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                height=target_h, width=target_w, num_frames=num_frames,
                guidance_scale=float(guidance_scale), num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(current_seed)
            ).frames[0]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
    
    # Read video file and convert to base64
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
    video_base64 = base64.b64encode(video_data).decode('utf-8')
    
    # Return result as JSON
    return {
        'status': 'success',
        'video': video_base64,
        'seed': current_seed,
        'dimensions': {'height': target_h, 'width': target_w},
        'frames': num_frames,
        'duration_seconds': duration_seconds
    }

def handler(event):
    """
    Main handler function for serverless deployment
    """
    try:
        result = generate_video_from_event(event)
        return {
            'statusCode': 200,
            'body': json.dumps(result),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'message': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

# For testing locally
if __name__ == "__main__":
    # Example test event
    test_event = {
        "prompt": "A person eating spaghetti",
        "height": 1024,
        "width": 720,
        "duration_seconds": 2.0,
        "steps": 4,
        "guidance_scale": 0.0,
        "seed": 42,
        "randomize_seed": False
    }
    
    # Or with image (you'd need to add a base64 image string)
    # test_event["input_image"] = "base64_string_here"
    
    result = handler(test_event)
    print(json.dumps(json.loads(result['body']), indent=2))