import os
import json
import torch
import re
from tqdm import tqdm
import base64
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaProcessor, LlavaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVLAdapter:
    def __init__(self, model_name="OpenGVLab/InternVL3_5-8B", device=None, dtype=torch.bfloat16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, image_path: str, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, do_sample: bool = False):
        image_tensor = load_image(image_path, max_num=12).to(self.device, dtype=self.model.dtype)
        prompt = f"<image>\n{system_prompt}\n{user_prompt}"
        generation_config = dict(
            max_new_tokens=1024,
            do_sample=True,
            eos_token_id=151645,
            pad_token_id=151643,
            )
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, image_tensor, prompt, generation_config)
        output_text = response
        if "Assistant:" in output_text:
            output_text = output_text.split("Assistant:", 1)[1].strip()
        return output_text
    
class OnlineModelAdapter:
    def __init__(self, model_name="claude-sonnet-4-20250514"):
        self.client = OpenAI(
            base_url="YOUR_API_URL",
            api_key="YOUR_API_KEY"
        )
        self.model_name = model_name
    
    def generate_response(self, image_path, system_prompt, user_prompt):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer

class LLaVAAdapter:
    def __init__(self, model_name: str = "llava-hf/LLaVA-NeXT-Video-34B-hf", use_next: bool = True):
        self.use_next = use_next
        if use_next:
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            ).eval()
        else:
            self.processor = LlavaProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            ).eval()
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    def generate_response(self, image_path: str, system_prompt: str, user_prompt: str):
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image_path}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

        response = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:", 1)[1].strip()

        return response

    
class QwenAdapter:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        ).eval()

    def generate_response(self, image_path, system_prompt, user_prompt):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"{user_prompt}"}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(torch.device('cuda'))

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
class CogVLMAdapter:
    def __init__(self, model_name="THUDM/cogvlm-chat-hf"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
    
    def generate_response(self, image_path, system_prompt, user_prompt):
        image = Image.open(image_path)
        full_prompt = f"{system_prompt}\n{user_prompt}"
        
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

def load_sudoku_dataset(label_json_path):
    with open(label_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = []
    for fname, info in data.items():
        N = info["N"]
        solution = info["solution"]
        difficulty = info["difficulty"]

        dataset.append({
            "image_path": os.path.join(os.path.dirname(label_json_path), fname),
            "solution": solution,
            "size": N,
            "difficulty": difficulty
        })
    return dataset


def image_to_tensor(img_path):
    return Image.open(img_path).convert('RGB')

def extract_matrix(text, N):
    pattern = re.compile(r"\[\s*(?:\[[^\]]+\]\s*,?\s*){%d}\]" % N)
    match = pattern.search(text)
    if not match:
        return None
    try:
        return eval(match.group(0))
    except:
        return None

def evaluate_on_dataset(model_adapter, dataset):
    results = {"A1": [], "A2": [], "A3": [], "A4": [], "B1": [], "B2": [], "B3": [], "B4": [], "C1": [], "C2": [], "C3": [], "C4": []}
    SYSTEM_CONTEXT = (
        "You are a Sudoku expert. Think the puzzle step by step, but only return the answer matrix. Always:\n"
        "1. Focus on the empty cell you are currently solving (e.g., 'E5').\n"
        "2. Check its constraints:\n"
        "- Row: '[existing numbers]'\n"
        "- Column: '[existing numbers]'\n"
        "- Palace: '[existing numbers]'\n"
        "- List possible numbers: '[Candidate numbers]'\n" 
        "3. Only place a number under the following conditions:\n"  
        "- It is the only candidate number ('Place [number] in [position] (unique candidate)'), or it is the only possible position in its row/column/region ('Place [number] in [position] (row/column/region unique)').\n" 
        "4. After placing the number, rescan the affected area.\n"
        "5. You can use the following approach: 'Hmm... A1 could be 3 or 5', 'Check box 2...', 'Verify: no conflicts?'\n\n"
        "Now you will see an image of an Sudoku puzzle. Each image is a complete Sudoku puzzle, with sizes possibly being 4x4, 6x6, or 9x9. A 6x6 palace consists of 2 rows and 3 columns. You need to understand the constraints based on all the numbers in the diagram and solve the Sudoku completely."
    )

    for idx, item in tqdm(enumerate(dataset), desc="Evaluating"):
        system_prompt = SYSTEM_CONTEXT
        user_prompt = f"Please read this {item['size']}x{item['size']} puzzle and provide the complete Sudoku solution matrix, returning it in standard nested list format: [[...], [...], ...]. Only return the answer matrix:"
        response = model_adapter.generate_response(item["image_path"], system_prompt, user_prompt)
        pred_matrix = extract_matrix(response, item['size'])
        correct = pred_matrix == item["solution"]
        results[item["difficulty"]].append(correct)
        if idx != 0 and (idx + 1) % 100 == 0:
            print_accuracy(results)

    return results

def print_accuracy(results, test_json="test/acc.json"):
    summary = {}
    print("\n准确率报告：")
    for level in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]:
        total = len(results[level])
        correct = sum(results[level])
        acc = (correct / total * 100) if total else 0
        print(f"{level.capitalize():<10} | 正确 {correct}/{total} | 准确率 {acc:.1f}%")
        summary[level] = {
            "correct": correct,
            "total": total,
            "accuracy": acc
        }

    if test_json:
        if os.path.exists(test_json):
            with open(test_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
        existing.append(summary)
        with open(test_json, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    model_choice = "online"
    model_name = "claude-sonnet-4"  # specify model name/path if needed

    if model_choice == "llava":
        model_adapter = LLaVAAdapter(model_name)
    elif model_choice == "cogvlm":
        model_adapter = CogVLMAdapter(model_name)
    elif model_choice == "qwen":    
        model_adapter = QwenAdapter(model_name)
    elif model_choice == "online":
        model_adapter = OnlineModelAdapter(model_name)
    elif model_choice == "internvl":
        model_adapter = InternVLAdapter(model_name)
    
    dataset = load_sudoku_dataset("data/sudoku_test/labels.json")
    results = evaluate_on_dataset(model_adapter, dataset)
    print_accuracy(results)
