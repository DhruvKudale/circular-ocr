import cv2
import pytesseract
from layout import get_circular_layout
from tables import get_rows_from_tatr, get_cols_from_tatr, order_rows_cols, get_cells_from_rows_cols, align_otsl_from_rows_cols, convert_to_html
from bs4 import BeautifulSoup
from PIL import Image


## EASY OCR
import easyocr
reader = easyocr.Reader(['hi']) # this needs to run only once to load the model into memory

# def get_full_table_ocr_data(img, lang="eng"):
#     """Runs OCR once on the full image and returns word-level bounding boxes."""
#     try:
#         ocr_data = pytesseract.image_to_data(img, config='--psm 6', lang=lang, output_type=pytesseract.Output.DICT)
#         # print(ocr_data)
#         return ocr_data
#     except Exception as e:
#         print(f"Error in OCR extraction: {e}")
#         return None

def get_cell_ocr(img, bbox, lang, ocr_engine):
    cell_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cell_pil_img = Image.fromarray(cell_img)
    if ocr_engine == 'tess':
        ocr_result = pytesseract.image_to_string(cell_pil_img, config='--psm 6', lang = lang)
        ocr_result = ocr_result.replace("\n", " ")
        ocr_result = ocr_result[:-1]
    else:
        ocr_result = get_easy_ocr(cell_img)
    return ocr_result.replace("|", "")


# def calculate_iou(box1, box2):
#     """Calculate Intersection over Union (IoU) between two bounding boxes."""
#     x1, y1, x2, y2 = box1
#     bx1, by1, bx2, by2 = box2
#
#     # Determine intersection box
#     inter_x1 = max(x1, bx1)
#     inter_y1 = max(y1, by1)
#     inter_x2 = min(x2, bx2)
#     inter_y2 = min(y2, by2)
#
#     # Calculate area of intersection
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#
#     # Calculate areas of both boxes
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (bx2 - bx1) * (by2 - by1)
#
#     # Calculate IoU
#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou


# def find_text_for_cell(ocr_data, cell_bbox, iou_threshold = 0.0000000000000001, used_indices = None):
#     """Finds and combines text from OCR data with IoU support and reading order correction.
#        Ensures no word is used in multiple cells.
#     """
#     if used_indices is None:
#         used_indices = set()
#
#     cell_text = []
#     x1, y1, x2, y2 = cell_bbox
#
#     # Collect words that overlap with the cell box using IoU
#     words_in_cell = []
#
#     for i in range(len(ocr_data['text'])):
#         if i in used_indices:
#             # Skip words that have already been used in another cell
#             continue
#
#         word = ocr_data['text'][i].strip()
#         if not word:  # Skip empty entries
#             continue
#
#         # Extract word bounding box
#         word_x, word_y = ocr_data['left'][i], ocr_data['top'][i]
#         word_w, word_h = ocr_data['width'][i], ocr_data['height'][i]
#         word_x2, word_y2 = word_x + word_w, word_y + word_h
#         word_bbox = [word_x, word_y, word_x2, word_y2]
#         word_conf = int(ocr_data['conf'][i])
#
#         # Calculate IoU and check if it's above threshold
#         if calculate_iou(cell_bbox, word_bbox) >= iou_threshold and word_conf >= 50:
#             words_in_cell.append((word, word_y, word_x))
#             used_indices.add(i)  # Mark this word as used
#
#     # Sort words by reading order: first by Y (top to bottom) and then X (left to right)
#     words_in_cell.sort(key=lambda item: (int(0.075 * item[1]), item[2]))
#
#     # Extract only the words and join them
#     cell_text = [word[0] for word in words_in_cell]
#
#     return " ".join(cell_text).strip().replace("|", ""), used_indices
#
def get_table_ocr_all_at_once(cropped_img, soup, lang, x1, y1, ocr_engine):
    # Full Table OCR
    # ocr_data = get_full_table_ocr_data(cropped_img, lang=lang)
    # used_indices = None
    # h, w, c = cropped_img.shape
    for bbox in soup.find_all('td'):
        # Replace the content inside the div with its 'title' attribute value
        ocr_bbox = bbox['bbox'].split(' ')
        ocr_bbox = list(map(int, ocr_bbox))
        # bbox.string, used_indices = find_text_for_cell(ocr_data, ocr_bbox, used_indices=used_indices)
        # if ocr_bbox[3] - ocr_bbox[1] > int(0.1 * h) or len(bbox.string.split()) > 5:
        #     # For multiline take no risks
        #     bbox.string = ''
        # if bbox.string.strip() == "":
        bbox.string = get_cell_ocr(cropped_img, ocr_bbox, lang, ocr_engine)
        # Correct wrt table coordinates
        ocr_bbox[0] += x1
        ocr_bbox[1] += y1
        ocr_bbox[2] += x1
        ocr_bbox[3] += y1
        bbox['bbox'] = f'{ocr_bbox[0]} {ocr_bbox[1]} {ocr_bbox[2]} {ocr_bbox[3]}'
    return soup

def get_text_hocr(image_path, bbox, lang, ocr_engine):
    image = cv2.imread(image_path)
    cropped_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    cv2.imwrite('temp-text.jpg', cropped_image)
    if ocr_engine == 'tess':
        extracted_text = pytesseract.image_to_string('temp-text.jpg', lang=lang, config='--psm 6')
    else:
        extracted_text = get_easy_ocr('temp-text.jpg')
    hocr = f'<p bbox=\"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\">' + extracted_text + '</p>'
    return hocr

def get_table_hocr(finalimgtoocr, bbox, lang, ocr_engine):
    image = cv2.imread(finalimgtoocr)
    cropped_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    img_file = 'temp-table.jpg'
    cv2.imwrite(img_file, cropped_image)
    rows = get_rows_from_tatr(img_file)
    cols = get_cols_from_tatr(img_file)
    print(str(len(rows)) + ' rows detected')
    print(str(len(cols)) + ' cols detected')
    rows, cols = order_rows_cols(rows, cols)
    cells = get_cells_from_rows_cols(rows, cols)
    ## Corner case if no cells detected
    if len(cells) == 0 or len(cols) == 0 or len(rows) == 0:
        if ocr_engine == 'tess':
            extracted_text = pytesseract.image_to_string('temp-text.jpg', lang=lang, config='--psm 6')
        else:
            extracted_text = get_easy_ocr('temp-text.jpg')
        hocr = extracted_text
        return hocr
    print('Logical TSR')
    row = 'C' * len(cols) + 'N'
    otsl_string = row * len(rows)
    # corrected_otsl = align_otsl_from_rows_cols(otsl_string, len(rows), len(cols))
    corrected_otsl = otsl_string
    # Correction
    corrected_otsl = corrected_otsl.replace("E", "C")
    corrected_otsl = corrected_otsl.replace("F", "C")
    print('OTSL => ' + otsl_string)
    print("Corrected OTSL => " + corrected_otsl)
    html_string, struc_cells = convert_to_html(corrected_otsl, len(rows), len(cols), cells)
    # Parse the HTML
    soup = BeautifulSoup('<html>' + html_string + '</html>', 'html.parser')
    soup = get_table_ocr_all_at_once(cropped_image, soup, lang, 0, 0, ocr_engine)
    for tag in soup.find_all(["td"]):
        if "bbox" in tag.attrs:
            del tag["bbox"]
    return str(soup)[6:-7]

def get_easy_ocr(image):
    result = reader.readtext(image, detail=0)
    return ' '.join(result)

def circular_to_txt(img_path, lang, ocr_engine):
        finalimgtoocr = img_path
        dets = get_circular_layout(finalimgtoocr)
        hocr_elements = ''
        tab_cnt = 1
        for det in dets:
            cls = det[0]
            bbox = det[1]
            if cls == 5:
                # Tables
                tab_hocr = get_table_hocr(finalimgtoocr, bbox, lang, ocr_engine)
                hocr_elements += f'<p bbox=\"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\">' + tab_hocr + '</p>'
                tab_cnt += 1
            # elif cls == 2: # Abandon class
            #     continue
            else:
                hocr = get_text_hocr(finalimgtoocr, bbox, lang, ocr_engine)
                # Can use class_names[cls] for classname instead
                hocr_elements += hocr

        final_hocr = hocr_elements
        return final_hocr
