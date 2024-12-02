import os
import torch
import torch.utils.checkpoint
from PIL import Image
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline

pretrained_model_name_or_path = './weights'
revision = None
variant = None
tokenizer = ChatGLMTokenizer.from_pretrained(f'{pretrained_model_name_or_path}/text_encoder')


text_encoder = ChatGLMModel.from_pretrained(f'{pretrained_model_name_or_path}/text_encoder', torch_dtype=torch.float16).half()


    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler")

vae = AutoencoderKL.from_pretrained(f"{pretrained_model_name_or_path}/vae", revision=None).half()


unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
).half()


pipeline = StableDiffusionXLInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler
    )


pipeline = pipeline.to('cuda')
prompt = ''
seed = 193
pipeline.load_lora_weights('./inpainting-lora-weights/pytorch_lora_weights.safetensors')
generator = torch.Generator(device='cuda:0').manual_seed(seed) if seed else None
image_extend = Image.open('./fj2_extend.png')
mask_image = Image.open('./fj2_mask.png')
height, width = image_extend.height, image_extend.width

image_extend = image_extend.resize((1024, 1024))
mask_image = mask_image.resize((1024, 1024))
image = pipeline(
    prompt = prompt,
    image = image_extend,
    mask_image = mask_image,
    height=image_extend.height,
    width=image_extend.width,
    guidance_scale = 7.5,
    generator= generator,
    num_inference_steps= 25,
    negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
    num_images_per_prompt = 1,
    # crops_coords_top_left=(3, 0),
    strength = 0.999
    ).images[0]
image = image.resize((width, height))
image.save(os.path.join('./inpainting-lora-weights', f"image_fj2_{seed}.png"))