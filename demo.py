from typing import List, Dict

from PIL import Image
import gradio as gr

from mmkg_visual_entity_recognition import ImageEntityRecognition

model_hub: Dict[str, ImageEntityRecognition] = {}


def inference(input_image: Image.Image, model_name: str) -> List[float]:
    if model_name not in model_hub:
        model_hub[model_name] = ImageEntityRecognition(model_name)
    model = model_hub[model_name]

    results = model.inference(input_image, 5)

    return {
        label: prob
        for (_, label, prob) in results
    }


def main():
    gr.Interface(
        inference,
        inputs=[
            gr.Image(type="pil"),
            gr.Dropdown(
                label="模型",
                show_label=True,
                choices=[
                    "resnet50",
                    "swin_v2_t",
                ],
                value="resnet50",
            ),
        ],
        outputs=gr.Label(num_top_classes=5),
        examples=[
            ["samples/lion.jpg", "resnet50"],
            ["samples/cat.jpg", "resnet50"],
            ["samples/cheetah.jpg", "resnet50"],
            ["samples/hot-dog.jpg", "resnet50"],
        ],
        allow_flagging="never",
    ).launch(share=True)


if __name__ == "__main__":
    main()
