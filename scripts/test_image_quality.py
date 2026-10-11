from PIL import Image, ImageDraw

from image_quality import assess_receipt_image


def test_clear_receipt_image_is_ready() -> None:
    image = Image.new("RGB", (800, 1000), (205, 205, 205))
    draw = ImageDraw.Draw(image)
    for y in range(80, 900, 45):
        draw.line((80, y, 720, y), fill="black", width=4)
    result = assess_receipt_image(image)
    assert result.status == "ready"


def test_small_dark_image_requests_retake() -> None:
    result = assess_receipt_image(Image.new("RGB", (120, 120), (15, 15, 15)))
    assert result.status == "retake_photo"
    assert result.reasons
    assert result.guidance
