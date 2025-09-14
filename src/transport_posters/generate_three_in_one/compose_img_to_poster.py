from PIL import Image, ImageDraw


def compose_img_to_poster(big_path: str, small1_path: str, small2_path: str, out_path: str,
                          bg_color=(255, 255, 255), gap=20, margin=40,
                          border_width=2, border_color=(180, 180, 180)):
    """
    Makes a poster from three pictures:
    - big_path - big picture on the left
    - small1_path - small picture on the top right
    - small2_path - small picture on the bottom right
    The result is saved in out_path.
    """
    big = Image.open(big_path).convert("RGB")
    s1 = Image.open(small1_path).convert("RGB")
    s2 = Image.open(small2_path).convert("RGB")

    h = big.height
    big_w = int(h)
    right_w = big_w // 2

    canvas_w = big_w + right_w + margin * 2 + gap
    canvas_h = h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    def paste_with_border(im: Image.Image, x: int, y: int, w: int, h: int):
        im_resized = im.resize((w, h))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([x - border_width, y - border_width,
                        x + w + border_width - 1, y + h + border_width - 1],
                       outline=border_color, width=border_width)
        canvas.paste(im_resized, (x, y))

    paste_with_border(big, margin, margin, big_w, h)

    small_h = (h - gap) // 2
    paste_with_border(s1, margin + big_w + gap, margin, right_w, small_h)
    paste_with_border(s2, margin + big_w + gap, margin + small_h + gap, right_w, small_h)

    canvas.save(out_path, "PNG")
