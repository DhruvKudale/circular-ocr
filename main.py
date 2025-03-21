import cv2
import pytesseract
from layout import resize_image_cv2
from ocr import circular_to_txt
import html2text


def get_image_ocr(image_path, ocr_mode, lang):
    img = resize_image_cv2(image_path)
    img_file = 'temp.jpg'
    cv2.imwrite(img_file, img)
    # img is ready to be OCRed
    if ocr_mode == 1:
        return pytesseract.image_to_string(img_file, lang=lang)
        # Plain Tesseract
    if ocr_mode == 2:
        html_content = circular_to_txt(img_file, lang)
        markdown_content = html2text.html2text(html_content)
        return markdown_content

if __name__ == '__main__':
    image_path = 'sample/page.jpg'
    ocr_mode = 2
    lang = 'eng+hin'
    response = get_image_ocr(image_path, ocr_mode, lang)
    print(response)