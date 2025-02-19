# https://huggingface.co/microsoft/Florence-2-large

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_name = "microsoft/Florence-2-large"
cache_dir = "./hf_cache"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    cache_dir=cache_dir,
).to(device)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)

prompt = "<DETAILED_CAPTION>"

image_path = "../dataset/without_inventory/0000/obs/00000000.png"
image = Image.open(image_path)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(
    generated_text,
    task="<DETAILED_CAPTION>",
    image_size=(image.width, image.height),
)

print(parsed_answer)
