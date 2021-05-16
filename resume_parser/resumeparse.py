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
# import phonenumbers
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

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

file = os.path.join(base_path, "titles_combined.txt")
file = open(file, "r", encoding='utf-8')
designation = [line.strip().lower() for line in file]
designitionmatcher = PhraseMatcher(nlp.vocab)
patterns = [
    nlp.make_doc(text) for text in designation if len(nlp.make_doc(text)) < 10
]
designitionmatcher.add("Job title", None, *patterns)

file = os.path.join(base_path, "skills.txt")
file = open(file, "r", encoding='utf-8')
skill = [line.strip().lower() for line in file]
skillsmatcher = PhraseMatcher(nlp.vocab)
patterns = [
    nlp.make_doc(text) for text in skill if len(nlp.make_doc(text)) < 10
]
skillsmatcher.add("Job title", None, *patterns)


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

    skills_header = ('credentials', 'qualifications', 'areas of experience',
                     'areas of expertise', 'areas of knowledge', 'skills',
                     "other skills", "other abilities",
                     'career related skills', 'professional skills',
                     'specialized skills', 'technical skills',
                     'computer skills', 'personal skills',
                     'computer knowledge', 'technologies',
                     'technical experience', 'proficiencies', 'languages',
                     'language competencies and skills',
                     'programming languages', 'competencies')

    misc = ('activities and honors', 'activities', 'affiliations',
            'professional affiliations', 'associations',
            'professional associations', 'memberships',
            'professional memberships', 'athletic involvement',
            'community involvement', 'refere', 'civic activities',
            'extra-Curricular activities', 'professional activities',
            'volunteer work', 'volunteer experience', 'additional information',
            'interests')

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
            clean_text = clean_text.replace("\r", "\n").replace(
                "\t", " ")  # Normalize text blob
            resume_lines = clean_text.splitlines(
            )  # Split text blob into individual lines
            resume_lines = [
                re.sub('\s+', ' ', line.strip()) for line in resume_lines
                if line.strip()
            ]  # Remove empty strings and whitespaces
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
            raw_text = ""
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
            resume_lines = [
                re.sub('\s+', ' ', line.strip()) for line in resume_lines
                if line.strip()
            ]

            return resume_lines, raw_text
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "

    def find_segment_indices(string_to_search, resume_segments,
                             resume_indices):
        for i, line in enumerate(string_to_search):

            if line[0].islower():
                continue

            header = line.lower()

            if [o for o in resumeparse.objective if header.startswith(o)]:
                try:
                    resume_segments['objective'][header]
                except:
                    resume_indices.append(i)
                    header = [
                        o for o in resumeparse.objective
                        if header.startswith(o)
                    ][0]
                    resume_segments['objective'][header] = i
            elif [
                    w for w in resumeparse.work_and_employment
                    if header.startswith(w)
            ]:
                try:
                    resume_segments['work_and_employment'][header]
                except:
                    resume_indices.append(i)
                    header = [
                        w for w in resumeparse.work_and_employment
                        if header.startswith(w)
                    ][0]
                    resume_segments['work_and_employment'][header] = i
            elif [
                    e for e in resumeparse.education_and_training
                    if header.startswith(e)
            ]:
                try:
                    resume_segments['education_and_training'][header]
                except:
                    resume_indices.append(i)
                    header = [
                        e for e in resumeparse.education_and_training
                        if header.startswith(e)
                    ][0]
                    resume_segments['education_and_training'][header] = i
            elif [
                    s for s in resumeparse.skills_header
                    if header.startswith(s)
            ]:
                try:
                    resume_segments['skills'][header]
                except:
                    resume_indices.append(i)
                    header = [
                        s for s in resumeparse.skills_header
                        if header.startswith(s)
                    ][0]
                    resume_segments['skills'][header] = i
            elif [m for m in resumeparse.misc if header.startswith(m)]:
                try:
                    resume_segments['misc'][header]
                except:
                    resume_indices.append(i)
                    header = [
                        m for m in resumeparse.misc if header.startswith(m)
                    ][0]
                    resume_segments['misc'][header] = i
            elif [
                    a for a in resumeparse.accomplishments
                    if header.startswith(a)
            ]:
                try:
                    resume_segments['accomplishments'][header]
                except:
                    resume_indices.append(i)
                    header = [
                        a for a in resumeparse.accomplishments
                        if header.startswith(a)
                    ][0]
                    resume_segments['accomplishments'][header] = i

    def slice_segments(string_to_search, resume_segments, resume_indices):
        resume_segments['contact_info'] = string_to_search[:resume_indices[0]]

        for section, value in resume_segments.items():
            if section == 'contact_info':
                continue

            for sub_section, start_idx in value.items():
                end_idx = len(string_to_search)
                if (resume_indices.index(start_idx) +
                        1) != len(resume_indices):
                    end_idx = resume_indices[resume_indices.index(start_idx) +
                                             1]

                resume_segments[section][sub_section] = string_to_search[
                    start_idx:end_idx]

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

        resumeparse.find_segment_indices(string_to_search, resume_segments,
                                         resume_indices)
        if len(resume_indices) != 0:
            resumeparse.slice_segments(string_to_search, resume_segments,
                                       resume_indices)
        else:
            resume_segments['contact_info'] = []

        return resume_segments

    def find_phone(text):
        try:
            mob_num_regex = r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)[-\.\s]??\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'
            phone = re.findall(re.compile(mob_num_regex), text)
            if phone:
                number = ''.join(phone[0])
                return number
        except:
            return ""

    def extract_email(text):
        email = re.findall(
            r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", text)
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

    def job_designition(text):
        job_titles = []

        __nlp = nlp(text.lower())

        matches = designitionmatcher(__nlp)
        for match_id, start, end in matches:
            span = __nlp[start:end]
            job_titles.append(span.text)
            break
        return job_titles

    def extract_skills(text):

        skills = []

        __nlp = nlp(text.lower())
        # Only run nlp.make_doc to speed things up

        matches = skillsmatcher(__nlp)
        for match_id, start, end in matches:
            span = __nlp[start:end]
            skills.append(span.text)
        skills = list(set(skills))
        return skills

    def read_file(file):
        if file.name.endswith('docx') or file.name.endswith('doc'):
            resume_lines, raw_text = resumeparse.convert_docx_to_txt(file)
        elif file.name.endswith('pdf'):
            resume_lines, raw_text = resumeparse.convert_pdf_to_txt(file)
        elif file.name.endswith('txt'):
            with open(file, 'r', encoding='latin') as f:
                resume_lines = f.readlines()

        else:
            resume_lines = None
        resume_segments = resumeparse.segment(resume_lines)

        full_text = " ".join(resume_lines)
        full_text = re.sub(r'\[(bookmark:.*?)\]', '', full_text)

        email = resumeparse.extract_email(full_text)
        phone = resumeparse.find_phone(full_text)
        name = resumeparse.extract_name(full_text)

        designition = resumeparse.job_designition(full_text)
        designition = list(dict.fromkeys(designition).keys())

        skills = ""

        if len(resume_segments['skills'].keys()):
            for key, values in resume_segments['skills'].items():
                skills += re.sub(key,
                                 '',
                                 ",".join(values),
                                 flags=re.IGNORECASE)
            skills = skills.strip().strip(",").split(",")

        if len(skills) == 0:
            skills = resumeparse.extract_skills(full_text)
        skills = list(dict.fromkeys(skills).keys())

        return {
            "email": email,
            "phone": phone,
            "name": name,
            "designition": designition,
            "skills": skills,
        }
