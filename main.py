import cv2
import pytesseract
from layout import resize_image_cv2
from ocr import circular_to_txt, get_easy_ocr
import markdownify
import os

def get_image_ocr(image_path, ocr_mode, lang, ocr_engine):
    img = resize_image_cv2(image_path)
    img_file = 'temp.jpg'
    cv2.imwrite(img_file, img)
    # img is ready to be OCRed
    if ocr_mode == 1:
        if ocr_engine == 'tess':
            return pytesseract.image_to_string(img_file, lang=lang)
        else:
            return get_easy_ocr(img_file)
        # Plain Tesseract
    if ocr_mode == 2:
        html_content = circular_to_txt(img_file, lang, ocr_engine)
        # print(html_content)
        markdown_content = markdownify.markdownify(html_content)
        # Remove temp files
        files_to_remove = ['temp.jpg', 'temp-table.jpg', 'temp-text.jpg']
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)
        return markdown_content

if __name__ == '__main__':
    image_path = 'circular-ocr/sample/page.jpg'
    ocr_mode = 2
    ocr_engine = 'easy' # Could be tess or easy
    lang = 'eng+hin'
    response = get_image_ocr(image_path, ocr_mode, lang, ocr_engine)
    print('*' * 50)
    print(response)