# !apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-resumeparse pstotext tesseract-ocr
# !sudo apt-get install libenchant1c2a


# !pip install tika
# !pip install docx2txt
# !pip install phonenumbers
# !pip install pyenchant
# !pip install stemming

from __future__ import division
import nltk

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('maxent_ne_chunker')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('brown')

import re
import os
from datetime import date

import nltk
import docx2txt
import pandas as pd
from tika import parser
import phonenumbers
import pdfplumber

import logging

import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

import sys
import operator
import string
import nltk
from stemming.porter2 import stem

# load pre-trained model
base_path = os.path.dirname(__file__)


nlp = spacy.load('en_core_web_sm')
custom_nlp2 = spacy.load(os.path.join(base_path,"degree","model"))

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)



class resumeparse(object):

    objective = (
        'career goal',
        'objective',
        'career objective',
        'employment objective',
        'professional objective',
        'summary',
        'career summary',
        'professional summary',
        'summary of qualifications',
        # 'digital'
    )

    work_and_employment = (
        'employment history',
        'work history',
        'work experience',
        'experience',
        'professional experience',
        'professional background',
        'additional experience',
        'career related experience',
        'related experience',
        'programming experience',
        'freelance',
        'freelance experience',
        'army experience',
        'military experience',
        'military background',
    )

    education_and_training = (
        'academic background',
        'academic experience',
        'programs',
        'courses',
        'related courses',
        'education',
        'educational background',
        'educational qualifications',
        'educational training',
        'education and training',
        'training',
        'academic training',
        'professional training',
        'course project experience',
        'related course projects',
        'internship experience',
        'internships',
        'apprenticeships',
        'college activities',
        'certifications',
        'special training',
    )

    skills_header = (
        'credentials',
        'qualifications',
        'areas of experience',
        'areas of expertise',
        'areas of knowledge',
        'skills',
        "other skills",
        "other abilities",
        'career related skills',
        'professional skills',
        'specialized skills',
        'technical skills',
        'computer skills',
        'personal skills',
        'computer knowledge',        
        'technologies',
        'technical experience',
        'proficiencies',
        'languages',
        'language competencies and skills',
        'programming languages',
        'competencies'
    )

    misc = (
        'activities and honors',
        'activities',
        'affiliations',
        'professional affiliations',
        'associations',
        'professional associations',
        'memberships',
        'professional memberships',
        'athletic involvement',
        'community involvement',
        'refere',
        'civic activities',
        'extra-Curricular activities',
        'professional activities',
        'volunteer work',
        'volunteer experience',
        'additional information',
        'interests'
    )

    accomplishments = (
        'achievement',
        'licenses',
        'presentations',
        'conference presentations',
        'conventions',
        'dissertations',
        'exhibits',
        'papers',
        'publications',
        'professional publications',
        'research',
        'research grants',
        'project',
        'research projects',
        'personal projects',
        'current research interests',
        'thesis',
        'theses',
    )
    
    def convert_docx_to_txt(docx_file):
        """
            A utility function to convert a Microsoft docx files to raw text.

            This code is largely borrowed from existing solutions, and does not match the style of the rest of this repo.
            :param docx_file: docx file with gets uploaded by the user
            :type docx_file: InMemoryUploadedFile
            :return: The text contents of the docx file
            :rtype: str
        """
        # try:
        #     text = docx2txt.process(docx_file)  # Extract text from docx file
        #     clean_text = text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
        #     resume_lines = clean_text.splitlines()  # Split text blob into individual lines
        #     resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]  # Remove empty strings and whitespaces

        #     return resume_lines
        # except KeyError:
        #     text = textract.process(docx_file)
        #     text = text.decode("utf-8")
        #     clean_text = text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
        #     resume_lines = clean_text.splitlines()  # Split text blob into individual lines
        #     resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]  # Remove empty strings and whitespaces
        #     return resume_lines
        try:
            text = parser.from_file(docx_file, service='text')['content']
        except RuntimeError as e:            
            logging.error('Error in tika installation:: ' + str(e))
            logging.error('--------------------------')
            logging.error('Install java for better result ')
            text = docx2txt.process(docx_file)
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "
        try:
            clean_text = re.sub(r'\n+', '\n', text)
            clean_text = clean_text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
            resume_lines = clean_text.splitlines()  # Split text blob into individual lines
            resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if
                            line.strip()]  # Remove empty strings and whitespaces
            return resume_lines, text
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "

    def convert_pdf_to_txt(pdf_file):
        """
        A utility function to convert a machine-readable PDF to raw text.

        This code is largely borrowed from existing solutions, and does not match the style of the rest of this repo.
        :param input_pdf_path: Path to the .pdf file which should be converted
        :type input_pdf_path: str
        :return: The text contents of the pdf
        :rtype: str
        """
        # try:
            # PDFMiner boilerplate
            # pdf = pdfplumber.open(pdf_file)
            # full_string= ""
            # for page in pdf.pages:
            #   full_string += page.extract_text() + "\n"
            # pdf.close()

            
        try:
            raw_text = parser.from_file(pdf_file, service='text')['content']
        except RuntimeError as e:            
            logging.error('Error in tika installation:: ' + str(e))
            logging.error('--------------------------')
            logging.error('Install java for better result ')
            pdf = pdfplumber.open(pdf_file)
            raw_text= ""
            for page in pdf.pages:
              raw_text += page.extract_text() + "\n"
            pdf.close()                
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "
        try:
            full_string = re.sub(r'\n+', '\n', raw_text)
            full_string = full_string.replace("\r", "\n")
            full_string = full_string.replace("\t", " ")

            # Remove awkward LaTeX bullet characters

            full_string = re.sub(r"\uf0b7", " ", full_string)
            full_string = re.sub(r"\(cid:\d{0,2}\)", " ", full_string)
            full_string = re.sub(r'• ', " ", full_string)

            # Split text blob into individual lines
            resume_lines = full_string.splitlines(True)

            # Remove empty strings and whitespaces
            resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]

            return resume_lines, raw_text
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "
            
    def find_segment_indices(string_to_search, resume_segments, resume_indices):
        for i, line in enumerate(string_to_search):

            if line[0].islower():
                continue

            header = line.lower()

            if [o for o in resumeparse.objective if header.startswith(o)]:
                try:
                    resume_segments['objective'][header]
                except:
                    resume_indices.append(i)
                    header = [o for o in resumeparse.objective if header.startswith(o)][0]
                    resume_segments['objective'][header] = i
            elif [w for w in resumeparse.work_and_employment if header.startswith(w)]:
                try:
                    resume_segments['work_and_employment'][header]
                except:
                    resume_indices.append(i)
                    header = [w for w in resumeparse.work_and_employment if header.startswith(w)][0]
                    resume_segments['work_and_employment'][header] = i
            elif [e for e in resumeparse.education_and_training if header.startswith(e)]:
                try:
                    resume_segments['education_and_training'][header]
                except:
                    resume_indices.append(i)
                    header = [e for e in resumeparse.education_and_training if header.startswith(e)][0]
                    resume_segments['education_and_training'][header] = i
            elif [s for s in resumeparse.skills_header if header.startswith(s)]:
                try:
                    resume_segments['skills'][header]
                except:
                    resume_indices.append(i)
                    header = [s for s in resumeparse.skills_header if header.startswith(s)][0]
                    resume_segments['skills'][header] = i
            elif [m for m in resumeparse.misc if header.startswith(m)]:
                try:
                    resume_segments['misc'][header]
                except:
                    resume_indices.append(i)
                    header = [m for m in resumeparse.misc if header.startswith(m)][0]
                    resume_segments['misc'][header] = i
            elif [a for a in resumeparse.accomplishments if header.startswith(a)]:
                try:
                    resume_segments['accomplishments'][header]
                except:
                    resume_indices.append(i)
                    header = [a for a in resumeparse.accomplishments if header.startswith(a)][0]
                    resume_segments['accomplishments'][header] = i

    def slice_segments(string_to_search, resume_segments, resume_indices):
        resume_segments['contact_info'] = string_to_search[:resume_indices[0]]

        for section, value in resume_segments.items():
            if section == 'contact_info':
                continue

            for sub_section, start_idx in value.items():
                end_idx = len(string_to_search)
                if (resume_indices.index(start_idx) + 1) != len(resume_indices):
                    end_idx = resume_indices[resume_indices.index(start_idx) + 1]

                resume_segments[section][sub_section] = string_to_search[start_idx:end_idx]

    def segment(string_to_search):
        resume_segments = {
            'objective': {},
            'work_and_employment': {},
            'education_and_training': {},
            'skills': {},
            'accomplishments': {},
            'misc': {}
        }

        resume_indices = []

        resumeparse.find_segment_indices(string_to_search, resume_segments, resume_indices)
        if len(resume_indices) != 0:
            resumeparse.slice_segments(string_to_search, resume_segments, resume_indices)
        else:
            resume_segments['contact_info'] = []

        return resume_segments

    def calculate_experience(resume_text):
        #
        # def get_month_index(month):
        #   month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        #   return month_dict[month.lower()]

        def correct_year(result):
            if len(result) < 2:
                if int(result) > int(str(date.today().year)[-2:]):
                    result = str(int(str(date.today().year)[:-2]) - 1) + result
                else:
                    result = str(date.today().year)[:-2] + result
            return result

        # try:
        experience = 0
        start_month = -1
        start_year = -1
        end_month = -1
        end_year = -1

        not_alpha_numeric = r'[^a-zA-Z\d]'
        number = r'(\d{2})'

        months_num = r'(01)|(02)|(03)|(04)|(05)|(06)|(07)|(08)|(09)|(10)|(11)|(12)'
        months_short = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
        months_long = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
        month = r'(' + months_num + r'|' + months_short + r'|' + months_long + r')'
        regex_year = r'((20|19)(\d{2})|(\d{2}))'
        year = regex_year
        start_date = month + not_alpha_numeric + r"?" + year
        

        end_date = r'((' + number + r'?' + not_alpha_numeric + r"?" + month + not_alpha_numeric + r"?" + year + r')|(present|current))'
        longer_year = r"((20|19)(\d{2}))"
        year_range = longer_year + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))" + r'(' + longer_year + r'|(present|current))'
        date_range = r"(" + start_date + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))" + end_date + r")|(" + year_range + r")"

        
        regular_expression = re.compile(date_range, re.IGNORECASE)
        
        regex_result = re.search(regular_expression, resume_text)
        
        while regex_result:

            date_range = regex_result.group()
            try:
                year_range_find = re.compile(year_range, re.IGNORECASE)
                year_range_find = re.search(year_range_find, date_range)
                replace = re.compile(r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))", re.IGNORECASE)
                replace = re.search(replace, year_range_find.group().strip())

                start_year_result, end_year_result = year_range_find.group().strip().split(replace.group())
                start_year_result = int(correct_year(start_year_result))
                if end_year_result.lower().find('present') != -1 or end_year_result.lower().find('current') != -1:
                    end_month = date.today().month  # current month
                    end_year_result = date.today().year  # current year
                else:
                    end_year_result = int(correct_year(end_year_result))


            except:

                start_date_find = re.compile(start_date, re.IGNORECASE)
                start_date_find = re.search(start_date_find, date_range)

                non_alpha = re.compile(not_alpha_numeric, re.IGNORECASE)
                non_alpha_find = re.search(non_alpha, start_date_find.group().strip())

                replace = re.compile(start_date + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))", re.IGNORECASE)
                replace = re.search(replace, date_range)
                date_range = date_range[replace.end():]

                start_year_result = start_date_find.group().strip().split(non_alpha_find.group())[-1]

                # if len(start_year_result)<2:
                #   if int(start_year_result) > int(str(date.today().year)[-2:]):
                #     start_year_result = str(int(str(date.today().year)[:-2]) - 1 )+start_year_result
                #   else:
                #     start_year_result = str(date.today().year)[:-2]+start_year_result
                # start_year_result = int(start_year_result)
                start_year_result = int(correct_year(start_year_result))

                if date_range.lower().find('present') != -1 or date_range.lower().find('current') != -1:
                    end_month = date.today().month  # current month
                    end_year_result = date.today().year  # current year
                else:
                    end_date_find = re.compile(end_date, re.IGNORECASE)
                    end_date_find = re.search(end_date_find, date_range)

                    end_year_result = end_date_find.group().strip().split(non_alpha_find.group())[-1]

                    # if len(end_year_result)<2:
                    #   if int(end_year_result) > int(str(date.today().year)[-2:]):
                    #     end_year_result = str(int(str(date.today().year)[:-2]) - 1 )+end_year_result
                    #   else:
                    #     end_year_result = str(date.today().year)[:-2]+end_year_result
                    # end_year_result = int(end_year_result)
                    end_year_result = int(correct_year(end_year_result))

            if (start_year == -1) or (start_year_result <= start_year):
                start_year = start_year_result
            if (end_year == -1) or (end_year_result >= end_year):
                end_year = end_year_result

            resume_text = resume_text[regex_result.end():].strip()
            regex_result = re.search(regular_expression, resume_text)
        print(end_year , start_year)
        return end_year - start_year  # Use the obtained month attribute

    # except Exception as exception_instance:
    #   logging.error('Issue calculating experience: '+str(exception_instance))
    #   return None

    def get_experience(resume_segments):
        total_exp = 0
        if len(resume_segments['work_and_employment'].keys()):
            text = ""
            for key, values in resume_segments['work_and_employment'].items():
                text += " ".join(values) + " "
            total_exp = resumeparse.calculate_experience(text)
            return total_exp, text
        else:
            text = ""
            for key in resume_segments.keys():
                if key != 'education_and_training':
                    if key == 'contact_info':
                        text += " ".join(resume_segments[key]) + " "
                    else:
                        for key_inner, value in resume_segments[key].items():
                            text += " ".join(value) + " "
            total_exp = resumeparse.calculate_experience(text)
            return total_exp, text
        return total_exp, " "

    def find_phone(text):
        try:
            return list(iter(phonenumbers.PhoneNumberMatcher(text, None)))[0].raw_string
        except:
            try:
                return re.search(
                    r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})',
                    text).group()
            except:
                return ""

    def extract_email(text):
        email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None

    def extract_name(resume_text):
        nlp_text = nlp(resume_text)

        # First name and Last name are always Proper Nouns
        # pattern_FML = [{'POS': 'PROPN', 'ENT_TYPE': 'PERSON', 'OP': '+'}]

        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        matcher.add('NAME', None, pattern)

        matches = matcher(nlp_text)

        for match_id, start, end in matches:
            span = nlp_text[start:end]
            return span.text
        return ""

    def extract_university(text, file):
        df = pd.read_csv(file, header=None)
        universities = [i.lower() for i in df[1]]
        college_name = []
        listex = universities
        listsearch = [text.lower()]

        for i in range(len(listex)):
            for ii in range(len(listsearch)):
                
                if re.findall(listex[i], re.sub(' +', ' ', listsearch[ii])):
                
                    college_name.append(listex[i])
        
        return college_name

    def job_designition(text, file):
        job_titles = []

        file = open(file, "r")
        designation = file.readlines()
        designation = [re.sub('\n', '', i).lower() for i in designation]

        __nlp = nlp(text.lower())
        matcher = PhraseMatcher(nlp.vocab)

        terms = designation

        # Only run nlp.make_doc to speed things up
        patterns = [nlp.make_doc(text) for text in terms if len(nlp.make_doc(text)) < 10]
        matcher.add("Job title", None, *patterns)

        matches = matcher(__nlp)
        for match_id, start, end in matches:
            span = __nlp[start:end]
            job_titles.append(span.text)
        return job_titles

    def get_degree(text):
        doc = custom_nlp2(text)
        degree = []

        degree = [ent.text.replace("\n", " ") for ent in list(doc.ents) if ent.label_ == 'Degree']
        return list(dict.fromkeys(degree).keys())

    def is_punctuation(word):
        return len(word) == 1 and word in string.punctuation

    def is_numeric(word):
        try:
            float(word) if '.' in word else int(word)
            return True
        except ValueError:
            return False

    class KeywordExtractor:
        """Extracts keywords and keyphrases from text input"""

        def __init__(self):
            self.stopwords = set(nltk.corpus.stopwords.words())
            self.top_fraction = 1

        def _generate_candidate_keywords(self, sentences, max_length=3):
            """Creates a list of candidate keywords, or phrases of at most max_length words, from a set of sentences"""
            phrase_list = []
            for sentence in sentences:
                words = map(lambda x: "|" if x in self.stopwords else x,
                            nltk.word_tokenize(sentence.lower()))

                phrase = []
                for word in words:
                    if word == "|" or resumeparse.is_punctuation(word):
                        if len(phrase) > 0:
                            if len(phrase) <= max_length:
                                phrase_list.append(phrase)
                            phrase = []
                    else:
                        phrase.append(word)
                if len(phrase):
                    phrase_list.append(phrase)

            return phrase_list

        def _calculate_word_scores(self, phrase_list):
            """Scores words according to frequency and tendency to appear in multi-word key phrases"""
            word_freq = nltk.FreqDist()
            word_multiplier = nltk.FreqDist()
            for phrase in phrase_list:
                # Give a higher score if word appears in multi-word candidates
                multi_word = min(2, len(list(filter(lambda x: not resumeparse.is_numeric(x), phrase))))
                for word in phrase:
                    # Normalize by taking the stem
                    word_freq[stem(word)] += 1
                    word_multiplier[stem(word)] += multi_word
            for word in word_freq.keys():
                word_multiplier[word] = word_multiplier[word] / float(word_freq[word])  # Take average
            word_scores = {}
            for word in word_freq.keys():
                word_scores[word] = word_freq[word] * word_multiplier[word]

            return word_scores

        def _calculate_phrase_scores(self, phrase_list, word_scores, metric='avg'):
            """Scores phrases by taking the average, sum, or max of the scores of its words"""
            phrase_scores = {}
            for phrase in phrase_list:
                phrase_score = 0
                if metric in ['avg', 'sum']:
                    for word in phrase:
                        phrase_score += word_scores[stem(word)]
                    phrase_scores[" ".join(phrase)] = phrase_score
                    if metric == 'avg':
                        phrase_scores[" ".join(phrase)] = phrase_score / float(len(phrase))
                elif metric == 'max':
                    for word in phrase:
                        phrase_score = word_scores[stem(word)] if word_scores[
                                                                      stem(word)] > phrase_score else phrase_score
                    phrase_scores[" ".join(phrase)] = phrase_score

            return phrase_scores

        def extract(self, text, max_length=3, metric='avg', incl_scores=False):
            """Extract keywords and keyphrases from input text in descending order of score"""
            sentences = nltk.tokenize.line_tokenize(text)
            # sentences = nltk.sent_tokenize(text)
            phrase_list = self._generate_candidate_keywords(sentences, max_length=max_length)

            word_scores = self._calculate_word_scores(phrase_list)
            phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores, metric=metric)
            sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)
            n_phrases = len(sorted_phrase_scores)

            if incl_scores:
                return sorted_phrase_scores[0:int(n_phrases / self.top_fraction)]
            else:
                return map(lambda x: x[0], sorted_phrase_scores[0:int(n_phrases / self.top_fraction)])


    def main(file_path):
        """Extracts keywords in order of relevance from a text file. Requires one command line argument.
        Args:
            file_path (str): path to text file
        """
        kwe = resumeparse.KeywordExtractor()
        with open(file_path) as file:
            
            content = file.read()
            
        keywords = kwe.extract(content, incl_scores=True)
        

    def extract_skills(text_string, file):
        all_skills = list()

        with open(file,'r',encoding='utf-8') as input_file:
            for line in input_file:
                all_skills.append(line.strip().lower())
        extractor = resumeparse.KeywordExtractor()

        skills = [s for s in extractor.extract(str(text_string)) if s in all_skills]

        return skills

    def read_file(file):
        # file = "/content/Asst Manager Trust Administration.docx"
        file = os.path.join(file)
        if file.endswith('docx') or file.endswith('doc'):
            resume_lines, raw_text = resumeparse.convert_docx_to_txt(file)
        elif file.endswith('pdf'):
            resume_lines, raw_text = resumeparse.convert_pdf_to_txt(file)

        elif file.endswith('txt'):

            with open(file, 'r', encoding='latin') as f:
                resume_lines = f.readlines()

        else:
            resume_lines = None
        resume_segments = resumeparse.segment(resume_lines)
        
        
        full_text = " ".join(resume_lines)

        email = resumeparse.extract_email(full_text)
        phone = resumeparse.find_phone(full_text)
        name = resumeparse.extract_name(" ".join(resume_segments['contact_info']))
        total_exp, text = resumeparse.get_experience(resume_segments)
        university = resumeparse.extract_university(full_text, os.path.join(base_path,'world-universities.csv'))

        designition = resumeparse.job_designition(full_text, os.path.join(base_path,"titles_combined.txt"))
        designition = list(dict.fromkeys(designition).keys())

        degree = resumeparse.get_degree(full_text)
        skills = " ".join(resumeparse.extract_skills(full_text, os.path.join(base_path,"LINKEDIN_SKILLS_ORIGINAL.txt")))
        return {
            "email": email,
            "phone": phone,
            "name": name,
            "total_exp": total_exp,
            "university": university,
            "designition": designition,
            "degree": degree,
            "skills": skills
        }

