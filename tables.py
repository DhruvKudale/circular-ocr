import torchvision
from PIL import Image
import torch
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
feature_extractor = DetrImageProcessor()
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")

def get_cols_from_tatr(img_file, col_thresh = 0.7, col_nms = 0.1):
    image = Image.open(img_file).convert("RGB")
    width, height = image.size
    image.resize((int(width * 0.5), int(height * 0.5)))
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    target_sizes = [image.size[::-1]]

    # For Columns
    col_results = feature_extractor.post_process_object_detection(outputs, threshold=col_thresh, target_sizes=target_sizes)[0]
    col_scores_t = col_results['scores']
    col_labels_t = col_results['labels']
    col_boxes_t = col_results['boxes']
    col_scores = []
    col_boxes = []
    for score, label, (xmin, ymin, xmax, ymax) in zip(col_scores_t.tolist(), col_labels_t.tolist(),
                                                      col_boxes_t.tolist()):
        name = model.config.id2label[label]
        if name == 'table column':
            col_scores.append(score)
            col_boxes.append((xmin, ymin, xmax, ymax))
    try:
        keep = torchvision.ops.nms(torch.tensor(col_boxes), torch.tensor(col_scores), iou_threshold = col_nms)
        final_col_results = torch.tensor(col_boxes)[keep]
    except:
        return []
    return final_col_results.tolist()

def get_rows_from_tatr(img_file, col_thresh = 0.7, col_nms = 0.1):
    image = Image.open(img_file).convert("RGB")
    width, height = image.size
    image.resize((int(width * 0.5), int(height * 0.5)))
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    target_sizes = [image.size[::-1]]

    # For Rows
    col_results = feature_extractor.post_process_object_detection(outputs, threshold=col_thresh, target_sizes=target_sizes)[0]
    col_scores_t = col_results['scores']
    col_labels_t = col_results['labels']
    col_boxes_t = col_results['boxes']
    col_scores = []
    col_boxes = []
    for score, label, (xmin, ymin, xmax, ymax) in zip(col_scores_t.tolist(), col_labels_t.tolist(),
                                                      col_boxes_t.tolist()):
        name = model.config.id2label[label]
        if name == 'table row':
            col_scores.append(score)
            col_boxes.append((xmin, ymin, xmax, ymax))
    try:
        keep = torchvision.ops.nms(torch.tensor(col_boxes), torch.tensor(col_scores), iou_threshold = col_nms)
        final_col_results = torch.tensor(col_boxes)[keep]
    except:
        return []
    return final_col_results.tolist()

def get_cells_from_rows_cols(rows, cols):
    i = 1
    ordered_cells = {}
    for row in rows:
        cells = []
        for col in cols:
            # Extract the required values and construct a new sublist
            cell = [int(col[0]), int(row[1]), int(col[2]), int(row[3])]
            # Append the new sublist to the cells list
            cells.append(cell)
        ordered_cells[i] = cells
        i = i + 1
    return ordered_cells

def order_rows_cols(rows, cols):
    # Order rows from top to bottom based on y1 (second value in the bounding box)
    rows = sorted(rows, key=lambda x: x[1])
    # Order columns from left to right based on x1 (first value in the bounding box)
    cols = sorted(cols, key=lambda x: x[0])
    return rows, cols


def align_otsl_from_rows_cols(otsl_string, rows, cols):
    N = len(otsl_string)
    C = cols
    R = rows
    if N != (C + 1) * R:
        # Needs correction
        actual_N = (C + 1) * R
        if N > actual_N:
            otsl_string = otsl_string[:actual_N]
            otsl_string = otsl_string[:-1] + 'N'
        else:
            diff = actual_N - N
            suffix = 'C' * (diff - 1) + 'N'
            otsl_string = otsl_string + suffix

    # Make sure Ns are at correct position !!
    # Remove if N is misplaced
    otsl_string_list = []
    for i in range(len(list(otsl_string))):
        char = otsl_string[i]
        if i > 0 and (i + 1) % (C + 1) == 0:
            char = 'N'
        else:
            if otsl_string[i] == 'N':
                char = 'C'
        otsl_string_list.append(char)
    final_otsl_string = ''.join(otsl_string_list)
    return final_otsl_string


def get_conv_html_from_otsl_with_cells(otsl_matrix, R, C, cells):
    html_string = '<table><tbody>'
    struc_cells = []
    # Generate string
    for i in range(R):
        html_string += '<tr>'
        for j in range(C + 1):
            e = otsl_matrix[i][j]
            if e == 'C':
                td_cell = cells[i + 1][j]
                rs, cs = get_cell_spans(otsl_matrix, i, j)
                if rs and cs:
                    # There is rowspan and colspan
                    extension = cells[i + 1 + cs][j + rs]
                    td_cell = [td_cell[0], td_cell[1], extension[2], extension[3]]
                    html_string += f'<td rowpsan="{rs + 1}" colspan="{cs + 1}" bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                elif rs and not cs:
                    # There is only row span
                    extension = cells[i + 1][j + rs]
                    td_cell = [td_cell[0], td_cell[1], extension[2], td_cell[3]]
                    html_string += f'<td colspan="{rs + 1}" bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                elif not rs and cs:
                    # There is only col span
                    extension = cells[i + 1 + cs][j]
                    td_cell = [td_cell[0], td_cell[1], td_cell[2], extension[3]]
                    html_string += f'<td rowspan="{cs + 1}" bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                else:
                    # Normal cell
                    html_string += f'<td bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                struc_cells.append(td_cell)
            elif e == 'N':
                # New row will start
                html_string += '</tr>'
            else:
                continue
    html_string += '</tbody></table>'
    return html_string, struc_cells

def convert_to_html(otsl_string, R, C, cells):
    N = len(otsl_string)
    if N != (C + 1) * R:
        # Needs correction
        actual_N = (C + 1) * R
        if N > actual_N:
            otsl_string = otsl_string[:actual_N]
            otsl_string = otsl_string[:-1] + 'N'
        else:
            diff = actual_N - N
            suffix = 'C' * (diff - 1) + 'N'
            otsl_string = otsl_string + suffix

    # Init OTSL matrix
    otsl_matrix = [[otsl_string[i * (C + 1) + j] for j in range(C + 1)] for i in range(R)]

    # Handle for 'U' in first row, replace by 'C'
    for i in range(len(otsl_matrix[0])):
        if otsl_matrix[0][i] == 'U':
            otsl_matrix[0][i] = 'C'

    # Handle for L in first column, replace by 'C'
    for i in range(R):
        if otsl_matrix[i][0] == 'L':
            otsl_matrix[i][0] = 'C'

    # Return converted string
    return get_conv_html_from_otsl_with_cells(otsl_matrix, R, C, cells)

def count_contiguous_occurrences(s, target_char):
    count = 0
    for char in s:
        if char == target_char:
            count += 1
        else:
            break
    return count

def get_cell_spans(otsl_matrix, i, j):
    entry = otsl_matrix[i][j]
    if entry != 'C':
        return 0, 0
    else:
        row_seq = ''.join(otsl_matrix[i])[j + 1:]
        col_seq = ''.join(row[j] for row in otsl_matrix)[i + 1:]
        rs = count_contiguous_occurrences(row_seq, 'L')
        cs = count_contiguous_occurrences(col_seq, 'U')
        return rs, cs