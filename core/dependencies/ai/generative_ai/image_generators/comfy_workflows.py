import json
import random
def text_to_image_workflow(prompt, negative_prompt):
  seed = random.randint(1, 4294967294)
  return {
  "6": {
    "inputs": {
      "text": prompt,
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "flux/ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "flux/t5xxl_fp16.safetensors",
      "clip_name2": "flux/clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "50",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "38",
        1
      ],
      "latent_image": [
        "52",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "normal",
      "steps": 30,
      "denoise": 1,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "12",
        0
      ],
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "38": {
    "inputs": {
      "step": 0,
      "sigmas": [
        "17",
        0
      ]
    },
    "class_type": "SplitSigmas",
    "_meta": {
      "title": "SplitSigmas"
    }
  },
  "50": {
    "inputs": {
      "noise_seed": seed
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "52": {
    "inputs": {
      "width": 1024,
      "height": 768,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
            "51": {
            "inputs": {
                "images": ["8", 0]
            },
            "class_type": "SaveImageWebsocket",
            "_meta": {"title": "SaveImageWebsocket"}
        }
}

import requests
import base64
from PIL import Image
from io import BytesIO
from typing import List

def image_to_base64_str(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_prompt_for_inpainting(image: Image.Image, mask: Image.Image, prompt: str = ""):
  image_b64 = image_to_base64_str(image)
  seed = random.randint(1, 4294967294)
  mask_b64 = image_to_base64_str(mask)

  data = {
    "3": {
      "inputs": {
        "seed": seed,
        "steps": 40,
        "cfg": 1,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1,
        "model": ["31", 0],
        "positive": ["38", 0],
        "negative": ["38", 1],
        "latent_image": ["38", 2]
      },
      "class_type": "KSampler",
      "_meta": {"title": "KSampler"}
    },
    "7": {
      "inputs": {
        "text": "",
        "clip": ["34", 0]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
    },
    "8": {
      "inputs": {
        "samples": ["3", 0],
        "vae": ["32", 0]
      },
      "class_type": "VAEDecode",
      "_meta": {"title": "VAE Decode"}
    },
    "9": {
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": ["8", 0]
      },
      "class_type": "SaveImage",
      "_meta": {"title": "Save Image"}
    },
    "23": {
      "inputs": {
        "text": prompt,
        "clip": ["34", 0]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
    },
    "26": {
      "inputs": {
        "guidance": 7.5,
        "conditioning": ["23", 0]
      },
      "class_type": "FluxGuidance",
      "_meta": {"title": "FluxGuidance"}
    },
    "31": {
      "inputs": {
        "unet_name": "flux1-fill-dev.safetensors",
        "weight_dtype": "default"
      },
      "class_type": "UNETLoader",
      "_meta": {"title": "Load Diffusion Model"}
    },
    "32": {
      "inputs": {
        "vae_name": "ae.safetensors"
      },
      "class_type": "VAELoader",
      "_meta": {"title": "Load VAE"}
    },
    "34": {
      "inputs": {
        "clip_name1": "flux/clip_l.safetensors",
        "clip_name2": "flux/t5xxl_fp16.safetensors",
        "type": "flux",
        "device": "default"
      },
      "class_type": "DualCLIPLoader",
      "_meta": {"title": "DualCLIPLoader"}
    },
    "38": {
      "inputs": {
        "noise_mask": False,
        "positive": ["26", 0],
        "negative": ["7", 0],
        "vae": ["32", 0],
        "pixels": ["48", 0],
        "mask": ["50", 0]
      },
      "class_type": "InpaintModelConditioning",
      "_meta": {"title": "InpaintModelConditioning"}
    },
    "48": {
      "inputs": {
        "image_base64": image_b64
      },
      "class_type": "LoadImageBase64",
      "_meta": {"title": "load image from base64 string"}
    },
    "49": {
      "inputs": {
        "image_base64": mask_b64
      },
      "class_type": "LoadImageBase64",
      "_meta": {"title": "load image from base64 string"}
    },
    "50": {
      "inputs": {
        "channel": "red",
        "image": ["49", 0]
      },
      "class_type": "ImageToMask",
      "_meta": {"title": "Convert Image to Mask"}
    },
    "51": {
      "inputs": {
        "images": ["8", 0]
      },
      "class_type": "SaveImageWebsocket",
      "_meta": {"title": "SaveImageWebsocket"}
    }
  }

  return data