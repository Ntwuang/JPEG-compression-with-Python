import cv2 
import numpy as np
import math
from collections import Counter, defaultdict
import heapq
import json
import sys
import jpeglib as jpeg
Q = np.loadtxt('D:\\Local Disk\\Python\\Quantization_Matrix.txt', delimiter='\t', dtype=int)
Q2 = np.loadtxt('D:\\Local Disk\\Python\\Quantization_Matrix_Chr.txt', delimiter='\t', dtype=int)
pi = math.pi
#This need resize function
def resize(arr, w, h):
    if (h%8!=0):
        h += (8 - h%8)
    if (w%8!=0):
        w += (8 - w%8)
    arr = cv2.resize(arr, (w, h), cv2.INTER_LINEAR)
    return arr, w, h    
def color_detect(arr, y, cb, cr, h, w):
    for i in range(0, h):
        for j in range(0, w):
            color = arr[i][j]
            y[i][j] = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            cb[i][j] = -0.1687*color[0] + 0.3313*color[1] + 0.5*color[2] + 128
            cr[i][j] = 0.5*color[0] - 0.4187*color[1] - 0.0813*color[2] + 128
    return y, cb, cr
def down_sampling(arr, h, w):
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            avg = (arr[i][j] + arr[i+1][j] + arr[i][j+1] + arr[i+1][j+1])/4
            arr[i][j] = avg
            arr[i+1][j] = avg
            arr[i][j+1] = avg
            arr[i+1][j+1] = avg
# DCT Transformation function for 8x8 blocks
def dct_trans(arr):
    dct_result = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                ci = 1 / (8 ** 0.5)
            else:
                ci = (2 / 8) ** 0.5
            if j == 0:
                cj = 1 / (8 ** 0.5)
            else:
                cj = (2 / 8) ** 0.5
            sum_val = 0
            for k in range(8):
                for l in range(8):
                    dct1 = (arr[k][l] - 128) * math.cos((2 * k + 1) * i * pi / (2 * 8)) * math.cos((2 * l + 1) * j * pi / (2 * 8))
                    sum_val += dct1
            dct_result[i][j] = ci * cj * sum_val
    return dct_result
# DCT Transformation for the entire image
def dct_full(arr, h, w):
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            arr[i:(i + 8), j:(j + 8)] = dct_trans(arr[i:(i + 8), j:(j + 8)])
    return arr 
# Quantization Matrix: Q50
def quantization(arr, h, w, quant = Q):  
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            arr[i:(i+8), j:(j+8)] = np.round(arr[i:(i+8), j:(j+8)]/quant)
# FINAL: Zigzag + Huffman encode
def zigzag(arr):
    n = 8
    zigzag_order = []
    for i in range(2 * n - 1):
        if i < n:
            if i % 2 == 0:
                x, y = i, 0
                while x >= 0:
                    zigzag_order.append(arr[x][y])
                    x -= 1
                    y += 1
            else:
                x, y = 0, i
                while y >= 0:
                    zigzag_order.append(arr[x][y])
                    x += 1
                    y -= 1
        else:
            if i % 2 == 0:
                x, y = n - 1, i - n + 1
                while y < n:
                    zigzag_order.append(arr[x][y])
                    x -= 1
                    y += 1
            else:
                x, y = i - n + 1, n - 1
                while x < n:
                    zigzag_order.append(arr[x][y])
                    x += 1
                    y -= 1
    return zigzag_order
def zigzagfull(arr, h, w):
    zigzag_full = []
    for i in range (0, h, 8):
        for j in range (0, w, 8):
            zigzag_full.extend(zigzag(arr[i:(i+8), j:(j+8)]))
    return zigzag_full
def run_length_encode(arr):
    rle = []
    cnt0 = 0
    for i in range(1, 64):
        if arr[i] == 0:
            cnt0 += 1
        else:
            while cnt0 > 15:
                rle.append((15, 0))
                cnt0 -= 16
            rle.append((cnt0, int(arr[i])))
            cnt0 = 0
    rle.append((0, 0))  # EOB
    return rle
def rlefull(arr):
    rle_full = []
    for i in range (0, len(arr), 64):
        rle_full.extend(run_length_encode(arr[i:(i+64)]))
    return rle_full
def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    if heap:
        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    else:
        return []
def encode_block(arr):
    freq = Counter(arr)
    tree = build_huffman_tree(freq)
    code = generate_huffman_codes(tree)
    return code
def generate_huffman_codes(huffman_tree):
    huffman_codes = {}
    for symbol, code in huffman_tree:
        huffman_codes[symbol] = code
    return huffman_codes
def encode_value(value):
    if value < 0:
        return '1' + format(-value, 'b')
    else:
        return '0' + format(value, 'b')
def huffman_encode(value, huffman_codes):
    encoded_value = huffman_codes.get(abs(value), '')
    sign_bit = '1' if value < 0 else '0'
    return sign_bit + encoded_value
def encode_ac_coefficient(run_length_pair, huffman_codes):
    run, value = run_length_pair
    encoded_run = format(run, 'b').zfill(4)  # Ensure 4 bits for run length
    if value < 0:
        encoded_value = '1' + format(-value, 'b')
    else:
        encoded_value = '0' + format(value, 'b')
    return huffman_codes.get((run, abs(value)), '') + encoded_value
def full_encode(zigzag, rle):
    dc = []
    diff = 0
    for i in range(0, len(zigzag), 64):
        dc.append(int(zigzag[i] - diff))
        diff = zigzag[i]
    dc_code = encode_block(dc)
    cdt_place = np.where(np.all(np.array(rle) == (0, 0), axis=1))[0]
    encoded_block = []
    ac_code = []
    start = 0
    cnt = 0
    for i in cdt_place:
        subset = rle[start:i+1]
        sub_code = encode_block(subset)
    # Encode DC component
        dc_value = dc[cnt]
        dc_encoded = huffman_encode(dc_value, dc_code) + encode_value(dc_value)
    # Encode AC components
        ac_encoded = ''.join([encode_ac_coefficient(pair, sub_code) for pair in subset])
    # Concatenate DC and AC encoded strings
        encoded_block.append(dc_encoded + ac_encoded)
        ac_code.append(sub_code)
        start = i + 1
        cnt += 1
    return encoded_block, dc_code, ac_code
# def write_encoded_to_text_file(encoded_block, dc_codes, ac_codes, filename):
#     # Convert dc_codes and ac_codes to lists of tuples
#     dc_codes_list = list(dc_codes.items())
#     ac_codes_list = [[(str(k), v) for k, v in code.items()] for code in ac_codes]
#     # Combine the data into a single list
#     data = [dc_codes_list, ac_codes_list, encoded_block]
#     # Write the data to the file
#     with open(filename, 'w') as f:
#         json.dump(data, f)

def write_encoded_to_text_file(encoded_block, dc_codes, ac_codes, filename):
    try:
        # Convert dc_codes and ac_codes to lists of tuples
        dc_codes_list = list(dc_codes.items())
        ac_codes_list = [list(code.items()) for code in ac_codes]

        # Combine the data into a single list
        data = [dc_codes_list, ac_codes_list, encoded_block]

        # Write the data to the file
        with open(filename, 'w') as f:
            json.dump(data, f)
        print("Data successfully written to", filename)
    except Exception as e:
        print("Failed to write data to file:", e)

img = cv2.imread('D:\\Local Disk\\Python\\sample_1.bmp') 
dummy = img
x = img.shape[0]
y = img.shape[1]
dummy, y, x = resize(dummy, y, x)
y_img = np.empty(shape=(x, y))
cb_img = np.empty(shape=(x, y))
cr_img = np.empty(shape=(x, y))
y_img, cb_img, cr_img = color_detect(dummy, y_img, cb_img, cr_img, x, y)
np.savetxt('y_img.txt', y_img, fmt='%d', delimiter='  ')
# cv2.imwrite('output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
OriginalBits = sys.getsizeof(y_img) + sys.getsizeof(cb_img) + sys.getsizeof(cr_img)
print(f"Number of Bits originally is {OriginalBits}")
y_img = dct_full(y_img, x, y)
np.savetxt('y_DCT.txt', y_img, fmt='%d', delimiter='  ')
dct_full(cb_img, x, y)
dct_full(cr_img, x, y)
quantization(y_img, x, y)
np.savetxt('y_quantization.txt', y_img, fmt='%d', delimiter=' ')
quantization(cb_img, x, y, Q2)
quantization(cr_img, x, y, Q2)
y_zigzag = zigzagfull(y_img, x, y)
np.savetxt('y_zigzag.txt', y_img, fmt='%d')
cb_zigzag = zigzagfull(cb_img, x, y)
cr_zigzag = zigzagfull(cr_img, x, y)
y_rle = rlefull(y_zigzag)
np.savetxt('y_RLE.txt', y_rle, fmt='%d')
cb_rle = rlefull(cb_zigzag)
cr_rle = rlefull(cr_zigzag)
y_encoded_block, y_dc_code, y_ac_code = full_encode(y_zigzag, y_rle)
cb_encoded_block, cb_dc_code, cb_ac_code = full_encode(cb_zigzag, cb_rle)
cr_encoded_block, cr_dc_code, cr_ac_code = full_encode(cb_zigzag, cb_rle)
# print(y_dc_code)
def size(encoded, dc, ac):
    encoded_block_size = sys.getsizeof(encoded)
    dc_codes_size = sys.getsizeof(dc)
    ac_codes_size = sum(sys.getsizeof(code) for code in ac)
    return (ac_codes_size + dc_codes_size + encoded_block_size)
CompressedBits = (size(y_encoded_block, y_dc_code, y_ac_code) + size(cb_encoded_block, cb_dc_code, cb_ac_code) + size(cr_encoded_block, cr_dc_code, cr_ac_code))
print(f"Number of bit after compression is {CompressedBits}")
print(f"The compression ratio is {OriginalBits/CompressedBits}")
write_encoded_to_text_file(y_encoded_block,y_dc_code, y_ac_code, 'output_filename.json')