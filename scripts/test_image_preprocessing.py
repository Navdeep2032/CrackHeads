from PIL import Image, ImageDraw

from image_preprocessing import preprocess_receipt_image


def test_preprocessing_upscales_and_returns_two_variants() -> None:
    image = Image.new("RGB", (300, 500), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    for y in range(30, 450, 30):
        draw.line((20, y, 280, y), fill="black", width=2)
    color, gray = preprocess_receipt_image(image, target_width=600, binarize=True)
    assert color.width == 600
    assert color.height == 1000
    assert gray.size == color.size
    assert gray.mode == "L"
