import os
import math
from math import ceil, floor
import pdfplumber
from resume_parser.layout_config import RESUME_HEADERS
from sentence_transformers import SentenceTransformer, util

def fun_k(N: int):
    return 1+(3.22*(math.log10(N)))

def get_cosin_similarity(text: str):
    cosine_scores = 0
    try:
        label_attributes = []
        for i in RESUME_HEADERS:
            label_attributes += RESUME_HEADERS[i]
        
        label_attributes = ' '.join(label_attributes)

        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cpu')
        embeddings1 = model.encode(text.lower(), convert_to_tensor=True)
        embeddings2 = model.encode(label_attributes.lower(), convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
    except Exception as e:
        print("get_cosin_similarity :: Excepiton :: ", str(e) )

    return cosine_scores

def generate_ngrams(words, n_range=2):
    ngrams = []
    for i in range(0, len(words)-(n_range-1)):
        ngram_words = words[i:i+n_range]
        ngram_text = " ".join([n_word['text'] for n_word in ngram_words])
        doc_top = min([n_word['doctop'] for n_word in ngram_words])
        n_x1, n_y1, n_x2, n_y2 =  ngram_words[0]['x0'], ngram_words[0]['top'], ngram_words[-1]['x1'], ngram_words[-1]['bottom']
        ngrams.append(
            {'x0': n_x1, 'top': n_y1, 'x1': n_x2, 'bottom': n_y2, 'text': ngram_text, 'doctop': doc_top, 'upright': True, 'direction': ngram_words[0]['direction']} 
            )

    for _ in range(n_range-1):
        ngrams.append({'x0':-1, 'x1':-1, 'top': -1, 'bottom':-1, 'text': None})

    return ngrams

def reformed_lines_dict_data(all_words_in_coords) -> dict:
    """
        Reformatting the OCR ouput in a line based format for identifying 
        words in same line struct.
    """
    try:
        lines = {}
        if all_words_in_coords:
            all_words_in_coords = [word for word in all_words_in_coords if word['text'].strip()]
            all_words_in_coords = sorted(all_words_in_coords, key=lambda x:(x['top'], x['x0']))
            curr_line = all_words_in_coords[0]['top']

            for index, word in enumerate(all_words_in_coords):
                if index <= len(all_words_in_coords)-1:
                    curr_top = all_words_in_coords[index]['top']
                    height = abs(word['bottom'] - word['top'])

                    if curr_top-(height//2) <= curr_line and curr_top+height <= curr_line+(height*1.5):
                        if lines:
                            lines[curr_line].append(word)
                        else:
                            curr_line = ceil(curr_top)
                            lines[curr_line] = [word]
                    else:
                        curr_line = ceil(curr_top)
                        lines.update({curr_line:[word]})
        if lines:
            lines = {curr_top:sorted(words, key=lambda x:x['x0']) for curr_top,words in sorted(lines.items())}
    except Exception as e:
        print(f"reformed_lines_dict_data :: Exception :: {e}")
    return lines

def words_concat_or_not(line, avg):
    lines = []
    try:
        for idx in range(0, len(line)-1, 2):
            if line[idx+1]['x0']-line[idx]['x1'] <= avg:
                new_line = {'text': line[idx]['text']+" "+line[idx+1]['text'], 'x0':line[idx]['x0'], 'x1':line[idx+1]['x1'],  'top':line[idx]['top'], 'doctop':line[idx]['doctop'], 'bottom':line[idx+1]['bottom'], 'upright':line[idx+1]['upright'], 'direction':line[idx]['direction']}
                lines.append(new_line)
            else:
                if line[idx] not in lines:
                    lines.append(line[idx])
                try:
                    if idx < len(line) and line[idx+1] not in lines:
                        lines.append(line[idx+1])
                except Exception as e:
                    pass
        
        if len(line)%2 != 0:
            lines.append(line[-1])

    except Exception as e:
        print("concat_or_not :: Exception :: ", str(e))

    if not lines: lines = line
    return lines

def lines_concat_or_not(lines_list, avg_height_diff):
    formed_sentence = []
    processed = []
    for line_idx in range(0, len(lines_list)-1):
        if line_idx in processed: continue
        try:
            if  0 < (lines_list[line_idx+1]['top']-lines_list[line_idx]['bottom']) <= avg_height_diff and lines_list[line_idx]['x0']<lines_list[line_idx+1]['x1']:
                new_sentence = {'text': lines_list[line_idx]['text']+", "+lines_list[line_idx+1]['text'],'x0': lines_list[line_idx]['x0'],'top':lines_list[line_idx]['top'],'x1': lines_list[line_idx+1]['x1'],'doctop':lines_list[line_idx]['doctop'],'bottom': lines_list[line_idx+1]['bottom'],'upright':lines_list[line_idx+1]['upright'],'direction': lines_list[line_idx]['direction']}

                formed_sentence.append(new_sentence)
                processed.append(line_idx+1)
            else:
                if lines_list[line_idx] not in formed_sentence:
                    formed_sentence.append(lines_list[line_idx])
                try:
                    if line_idx < len(lines_list) and lines_list[line_idx+1] not in formed_sentence:
                        formed_sentence.append(lines_list[line_idx+1])
                        processed.append(line_idx+1)
                except Exception as e:
                    pass
        except Exception as e:
            print("lines_concat_or_not :: Exception :: ", str(e))

    if len(lines_list)%2 != 0:
        formed_sentence.append(lines_list[-1])

    if not formed_sentence: formed_sentence = lines_list
    
    return formed_sentence

def form_sentences(lines_list):
    lines_dict = reformed_lines_dict_data(lines_list)
    formed_line = []
    formed_sentences = []

    for _, line in lines_dict.items():
        if len(line) <= 2 and line:
            if len(line)<=1:
                formed_line.extend(line)
            else:
                half_avg = sum([ceil(line[0]['x1'] - line[0]['x0'])//2, ceil(line[-1]['x1']-line[-1]['x0'])//2])/2
                if line[-1]['x0'] - line[0]['x1'] <= half_avg:
                    formed_line.extend([{'text': line[0]['text']+" "+line[-1]['text'], 'x0':line[0]['x0'], 'x1':line[-1]['x1'],  'top':line[0]['top'], 'doctop':line[0]['doctop'], 'bottom':line[-1]['bottom'], 'upright':line[-1]['upright'], 'direction':line[0]['direction']}])
                else:
                    formed_line.extend(line)
        else:
            avg_diff_list = [ceil(line[idx]['x1'] - line[idx]['x0'])//2 for idx in range(len(line)-1)]
            avg = ceil(sum(avg_diff_list)/len(avg_diff_list))
            new_line = []
            while True:
                new_line = words_concat_or_not(line, avg)
                if line == new_line:
                    break
                line = new_line

            formed_line.extend(new_line)
    
    line_diff_list = [formed_line[idx+1]['top']-formed_line[idx]['bottom'] for idx in range(len(formed_line)-1)]
    if len(line_diff_list):
        avg_height_diff = floor(sum(line_diff_list)/len(line_diff_list))

    else: avg_height_diff = 1

    formed_sentences = []
    lines = formed_line.copy()
    while True:
        formed_sentences = lines_concat_or_not(formed_line, avg_height_diff)
        if formed_line == formed_sentences:
            break
        formed_line = formed_sentences

    return lines, formed_sentences

class ResumeRecon:
    
    """
        Store multiple useful information about a pdf document.
        1. Document Raw Text. - DONE
        2. Bold words. - Done
        3. Document Columns/Section Info. - (Columns Done)
        4. Bullet points. - Pending

    """
    def __init__(self, path):
        self.pages_shape = {}
        self.headers = {}
        self.columns = {}
        self.chars = {}
        self.words = {}
        self.formed_headers = {}
        self.tables = {}
        self.document = path
        self.get_basic_data()

    def get_basic_data(self):
        """
            Find bold characters and form words.
        """
        with pdfplumber.open(self.document) as pdf:
            pages = pdf.pages
            for idx, page in enumerate(pages):
                self.pages_shape[idx] = {'width': page.width, 'height': page.height}
                self.chars[idx] = page.chars
                self.words[idx] = page.extract_words()
                self.tables[idx] = page.extract_tables()

    def get_bold_letters(self):
        self.bold_letters = {}

        for page,chars in self.chars.items():
            page_bchars = []
            for char in chars:
                if 'bold' in char['fontname'].lower():
                    page_bchars.append(char)

            page_bchars = sorted(page_bchars, key=lambda x: (x['top'], x['x0']))
            words = []
            temp_words = []
            for char in page_bchars:
                if char['text'].strip():
                    temp_words.append(char)
                else:
                    if temp_words:
                        word = ''.join([i['text'] for i in temp_words]) 
                        zip_cords =  [i for i in zip(*[(i['x0'], i['top'], i['x1'], i['bottom']) for i in temp_words])]
                        cords = (min(zip_cords[0]), min(zip_cords[1]), max(zip_cords[2]), max(zip_cords[3]))
                        words.append((word, cords))
                        temp_words = []

            self.bold_letters[page] = words
    
    def get_height_based_seggregated_words(self):
        data = self.words
        self.height_data  = {}
        try:
            for page in data:
                self.height_data[page] = {}
                heights = set([(i['bottom']-i['top']) for i in data[page]])
                for height in heights:
                    word_list = [word for word in data[page] if (word['bottom']-word['top'])==height]
                    self.height_data[page][str(height)] = word_list
        except Exception as e:
            print("Exception :: get_height_based_seggregated_words :: ", str(e))
    
    def identify_header_tags(self):
        self.header_tags = {}
        self.word_tags = {}
        data = self.height_data

        for page in data:
            total_classes = []

            # Dynamically identify interval length
            heights = [float(height) for height in data[page]]
            for height in data[page]:
                total_classes.append(len(data[page][height]))

            if len(total_classes) > 1:
                k = fun_k(sum(total_classes))
                intervals = (max(heights)-min(heights))/k
                
                # seggregate into heading tags (h1, h2, h3)
                tag_heights = {}
                tmp_heights = sorted(heights)
                idx = 0
                while tmp_heights:
                    temp_height = tmp_heights[0]+intervals

                    new_heights = []
                    for h in tmp_heights:
                        tmp_heights = sorted(tmp_heights)
                        if h<=temp_height:
                            new_heights.append(h)
                            tmp_heights.remove(h)
                    idx+=1 
                    tag_heights[f"h{idx}"] = new_heights
                
                self.header_tags[page] = tag_heights

        # adding words with respect to each tag
        for page in self.header_tags:
            page_words = {}
            for height_tag, heights in self.header_tags[page].items():
                words = []
                for height in heights:
                    words += [i for i in self.height_data[page][str(height)]]
                
                page_words[height_tag] = words
            
            self.word_tags[page] = page_words

    def get_possible_header_words(self):
        identified_header = 0
        self.formed_headers = {}
        page_headers = {}
        try:
            for page, tagged_words in self.word_tags.items():
                identified_header = 0
                for tag, list in tagged_words.items():
                    if len(list)<50:
                        match_score = get_cosin_similarity(' '.join([i['text'] for i in list]))
                        if match_score > identified_header:
                            identified_header = match_score
                            page_headers[page] = tag

            for page, htag in page_headers.items():
                self.headers[page] = self.word_tags[page][htag]
        
        except Exception as e:
            print("get_possible_header_words :: Exception :: ", str(e))

        for page, words_list in self.headers.items():
            formed_word_list = form_sentences(words_list)[1]
            self.formed_headers[page] = formed_word_list

    def seggregate_columns(self):
        formed_headers = self.formed_headers.copy()
        sections = {}
        for page in formed_headers:
            sections[page] = {}
            pdf_width = self.pages_shape[page].get('width')
            pdf_height = self.pages_shape[page].get('height')
            left_sec_words = []
            right_sec_words = []

            while formed_headers[page]:
                formed_headers[page] = sorted(formed_headers[page], key=lambda x: x['x0'])
                for word in formed_headers[page]:

                    if word.get('x0')<pdf_width//4:
                        left_sec_words.append(word)
                        formed_headers[page].remove(word)
                    else:
                        right_sec_words.append(word)
                        formed_headers[page].remove(word)
            
            if right_sec_words:
                left_col_left = min([i.get('x0') for i in left_sec_words])
                left_col_top = min([i.get('top') for i in left_sec_words])
                left_col_right = min([i.get('x0') for i in right_sec_words])

                right_col_left = min([i.get('x0') for i in right_sec_words])-20
                right_col_top = min([i.get('top') for i in right_sec_words])
                # right_col_right = max([i.get('x1') for i in right_sec_words])

                right_col_right = pdf_width
                bottom = pdf_height

                left_sec = {'x0': left_col_left, 'top': left_col_top, 'x1': left_col_right, 'bottom': bottom}
                right_sec = {'x0': right_col_left, 'top': right_col_top, 'x1': right_col_right, 'bottom': bottom}
            else:
                left_col_left = min([i.get('x0') for i in left_sec_words])
                left_col_top = min([i.get('top') for i in left_sec_words])
                left_col_right = pdf_width

                bottom = pdf_height

                left_sec = {'x0': left_col_left, 'top': left_col_top, 'x1': left_col_right, 'bottom': bottom}
                right_sec = {'x0': 0, 'top': 0, 'x1': 0, 'bottom': 0}

            sections[page]['left'] = left_sec
            sections[page]['right'] = right_sec


        for page, section in sections.items():
            self.columns[page] = {}
            for div in section:
                self.columns[page][div] = [word for word in self.words[page] if (word['x0']>=section[div]['x0'] and \
                    word['top'] >= section[div]['top'] and \
                        word['x1'] <= section[div]['x1'] and \
                            word['bottom'] <= section[div]['bottom'])]

        page_words = self.words.copy()
        for page, section in sections.items():
            self.columns[page] = {}
            self.columns[page]['free_div'] = page_words[page].copy()
            for div in section:
                self.columns[page][div] = []
                for word in page_words[page]:
                    if (word['x0']>=section[div]['x0'] and \
                        word['top'] >= section[div]['top'] and \
                            word['x1'] <= section[div]['x1'] and \
                                word['bottom'] <= section[div]['bottom']):
                        
                        self.columns[page][div].append(word)
                        self.columns[page]['free_div'].remove(word)

    def segment_header_words(self):
        self.segments = {}
        gram_words = {}

        for page, section in self.columns.items():
            page_headers = [i['text'] for i in  self.formed_headers[page]]
            self.segments[page] = {}
            gram_words[page] = {}

            no_segment_words = []
            for div, data in section.items():
                # GENERATE NGRAM WORDS based on header lengths 
                for header_x in page_headers:
                    if len(header_x.split())>1:
                        n_gram_words = generate_ngrams(data, len(header_x.split()))
                        gram_words[page][len(header_x.split())] = n_gram_words
                
                # segment words based on header match
                header = None
                try:
                    for idx, word in enumerate(data):
                        if word in self.formed_headers[page] or \
                            gram_words[page].get(2, data)[idx] in self.formed_headers[page] or \
                                gram_words[page].get(3, data)[idx] in self.formed_headers[page]:
                            try:
                                if word in self.formed_headers[page]:
                                    header = word['text']
                                    self.segments[page][header] = []
                                    continue
                            
                                elif gram_words[page].get(2, data)[idx] in self.formed_headers[page]:
                                    header = gram_words[page].get(2, data)[idx]['text']
                                    self.segments[page][header] = []
                                    continue

                                elif gram_words[page].get(3, data)[idx] in self.formed_headers[page]:
                                    header = gram_words[page].get(3, data)[idx]['text']
                                    self.segments[page][header] = []
                                    continue
                            except Exception as e:
                                print("segment_header_words ::  EXCEPTION :: ", str(e))
                        else:
                            if isinstance(self.segments[page].get(header, None), list):
                                self.segments[page][header].append(word)
                            else:
                                no_segment_words.append(word)
                except Exception as e:
                    print("segment_header_words ::  EXCEPTION :: ", str(e))
            self.segments[page]['FREE_TEXT'] = no_segment_words

    def process_resume(self):
        self.get_height_based_seggregated_words()
        self.identify_header_tags()
        self.get_possible_header_words()
        self.seggregate_columns()
        self.segment_header_words()
        return (self.headers, self.columns, self.segments)
       