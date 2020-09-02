import pytesseract
from pytesseract import Output
from PIL import Image
import pandas as pd
img = Image.open(r'../Desktop/jfile2-1.jpg')
img2 = img.crop((1194, 93, 1879, 1376))
img2.save("img2.jpg")
custom_config = r'-c preserve_interword_spaces=1 --oem 3 --psm 6 -l jpn'
d = pytesseract.image_to_data(img2, config=custom_config, output_type='data.frame')
# df = pd.DataFrame.from_dict(d, orient='index')
df = pd.DataFrame(d)
# print(df)
# clean up blanks
df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]
# sort blocks vertically
sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
print(sorted_blocks)
for block in sorted_blocks:
    curr = df1[df1['block_num']==block]
    sel = curr[curr.text.str.len()>3]
    char_w = (sel.width/sel.text.str.len()).mean()
    print("curr strarts")
    print(curr)
    print("curr end, sel start")
    print(sel)
    print("sel end char_w start")
    print(char_w)
    prev_par, prev_line, prev_left = 0, 0, 0
    text = ''
    for ix, ln in curr.iterrows():
        # add new line when necessary
        if prev_par != ln['par_num']:
            text += '\n'
            prev_par = ln['par_num']
            prev_line = ln['line_num']
            prev_left = 0
        elif prev_line != ln['line_num']:
            text += '\n'
            prev_line = ln['line_num']
            prev_left = 0

        added = 0  # num of spaces that should be added
        if ln['left']/char_w > prev_left + 1:
            added = int((ln['left'])/char_w) - prev_left
            text += ' ' * added
        if str(ln['text'])=='nan':
            ln['text'] = ''  
        # print(ln['text'])
        text += str(ln['text']) + ' '
        prev_left += len(str(ln['text'])) + added + 1
    text += '\n'
    print(text)
