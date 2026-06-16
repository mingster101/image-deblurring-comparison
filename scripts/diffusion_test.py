import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# ===== SET DEVICE =====
device = "cuda" if torch.cuda.is_available() else exit("GPU not available. Please run on a machine with CUDA support.")

# ===== LOAD MODEL =====
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

# optimize VRAM (penting untuk RTX 3070)
pipe.enable_attention_slicing()

# ===== PATH =====
input_folder = "Motion_Deblurring/Datasets/GOPRO_Large_flat/test/blur"
output_folder = "diffusion_output"

os.makedirs(output_folder, exist_ok=True)

# ===== PARAMETER =====
prompt = "a sharp photo of a scene, high detail, no blur, realistic, not too much noise, no artifacts, 4k resolution"
strength = 0.4
guidance_scale = 10
print(pipe.device)
# ===== PROCESS =====
for file in os.listdir(input_folder):
    if not file.endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, file)

    print(f"Processing: {file}")

    init_image = Image.open(img_path).convert("RGB").resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale
    ).images[0]

    result.save(os.path.join(output_folder, file))

print("DONE.")