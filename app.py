import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import gradio as gr

CLIP_MODEL = "openai/clip-vit-base-patch32"
MODEL_PATH = "./driving_vqa_v2_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrivingVQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_answers),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            img_out = self.clip.vision_model(pixel_values=pixel_values)
            txt_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            img_feat = self.clip.visual_projection(img_out.pooler_output)
            txt_feat = self.clip.text_projection(txt_out.pooler_output)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return self.classifier(torch.cat([img_feat, txt_feat], dim=-1))


print("Loading model...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
answer_to_idx: dict[str, int] = ckpt["answer_to_idx"]
idx_to_answer = {i: a for a, i in answer_to_idx.items()}
model = DrivingVQAModel(num_answers=ckpt["num_answers"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
print("Ready.")

EXAMPLE_QUESTIONS = [
    "How many cars are there?",
    "Is there a pedestrian in the scene?",
    "How many pedestrians are there?",
    "Is there a car in the scene?",
    "How many trucks are there?",
    "Is there a bus in the scene?",
    "How many traffic cones are there?",
    "Is there a barrier in the scene?",
]


def predict(image: Image.Image, question: str) -> str:
    if image is None or not question.strip():
        return "Please provide both an image and a question."
    inputs = processor(
        images=image, text=question,
        return_tensors="pt", padding="max_length",
        max_length=77, truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(logits, dim=-1)[0]
    top3 = probs.topk(min(3, len(idx_to_answer)))
    answer = idx_to_answer[top3.indices[0].item()]
    details = "\n".join(
        f"  {idx_to_answer[i.item()]}: {p.item()*100:.1f}%"
        for i, p in zip(top3.indices, top3.values)
    )
    return f"Answer: **{answer}**\n\nTop predictions:\n{details}"


with gr.Blocks(title="Autonomous Driving VQA") as demo:
    gr.Markdown("## Autonomous Driving Scene VQA\nUpload a driving scene image and ask a question about it.")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Driving Scene Image")
            question_input = gr.Textbox(label="Question", placeholder="How many cars are there?")
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=question_input,
                label="Example Questions",
            )
            submit_btn = gr.Button("Ask", variant="primary")
        with gr.Column():
            output = gr.Markdown(label="Answer")
    submit_btn.click(fn=predict, inputs=[img_input, question_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
