Resume parser
=============
Premium resume parsing services have been moved to `Resume-Parser <https://resume-parser.com/application/resumes>`_. Please try the demo for free and give us your `feedback <https://resume-parser.com>`_



    A resume parser used for extracting information from resumes

Built with ❤︎ and :coffee: by `Kumar
Rajwani <https://github.com/kbrajwani>`__ and `Brian
Njoroge <https://github.com/Brianjoroge>`__

--------------

Features
========

-  Extract name
-  Extract email
-  Extract mobile numbers
-  Extract skills
-  Extract total experience
-  Extract college name
-  Extract degree
-  Extract designation
-  Extract company names

Installation
============

-  You can install this package using

.. code:: bash

    pip install resume-parser

-  For NLP operations we use spacy and nltk. Install them using below
   commands:

.. code:: bash

    # spaCy
    python -m spacy download en_core_web_sm

    # nltk
    python -m nltk.downloader stopwords
    python -m nltk.downloader punkt
    python -m nltk.downloader averaged_perceptron_tagger
    python -m nltk.downloader universal_tagset
    python -m nltk.downloader wordnet
    python -m nltk.downloader brown
    python -m nltk.downloader maxent_ne_chunker

Supported File Formats
======================

-  PDF and DOCx and TXT files are supported on all Operating Systems

Usage
=====

-  Import it in your Python project

.. code:: python

    from resume_parser import resumeparse

    data = resumeparse.read_file('/path/to/resume/file')

Result
======

The module would return a dictionary with result as follows:

::

    {'degree': ['BSc','MSc'],
         'designition': [
             'content writer',
             'data scientist',
             'systems administrator',
         ],
         'email': 'maunarokguy@gmail.com',
         'name': 'Brian Njoroge',
         'phone': '+918511593595',
         'skills': [
             'Python',
             ' C++',
             'Power BI',
             'Tensorflow',
             'Keras',
             'Pytorch',
             'Scikit-Learn',
             'Pandas',
             'NLTK',
             'OpenCv',
             'Numpy',
             'Matplotlib',
             'Seaborn',
             'Django',
             'Linux',
             'Docker'],
         'total_exp': 3,
         'university': ['gujarat university', 'wuhan university', 'egerton university']}

The Premium API would return a result as follows:

::


    {'Basics': {
        'Full Name': 'Brian Njoroge',
        'Title': '',
        'First Name': 'Brian',
        'Last Name': 'Njoroge',
        'DOB': '',
        'Email': 'maunarokguy@gmail.com',
        'Facebook': '',
        'Github': 'https://github.com/brinjoro',
        'Languages': ['English', 'Swahili'],
        'Linkedin': 'linkedin.com/in/brian-njoroge-13708473',
        'Medium': 'https://medium.com/@dlmade',
        'Phone': '8511593595',
        'Position': 'Machine Learning Engineer',
        'Stackoverflow': '',
        'Summary': '',
        'Twitter': ''},
        
        'Education': [{
            'Area': 'Msc in Machine learning and artificial intelligence',
            'Education_duration': '07/2018 - 07/2020',
            'Institution': ' Gujarat University, India '},
            {
                'Area': 'BSc. Computer Science ',
                'Education_duration': '06/2013 - 12/2017',
                'Institution': ' Egerton University'
            }],
            
        'Work': [{'Company': ' Solusoft Technologies ',
                  'Position': 'Machine Learning Developer ',
                  'Working_duration': '01/2019 - 12/2020'},
                 {'Company': 'Muva Technologies ',
                  'Position': '01/2019 - 12/2020',
                  'Working_duration': '12/2017 - 07/2018'}],
                  
        'Skills': ['Awslabs',
                   'Tesseract OCR',
                   'NLP', 'Computer Vision', 'Flask', 'Tensorflow',
                   'Pytorch', 'NLTK', 'SKlearn', 'django', 'rasa',
                   'Keras', 'Web Scraping',
                   'docker', 'git'],
                   
        'Interests': [
            'Deep Learning Machine Learning',
            'Artificial Intelligence',
            'Basketball'],
            
        'ProjectName': [
                        'Image classification',
                        'Pdf mining',
                        'Video classification',
                        'Logo detection',
                        'Chatbot'],
                        
        'Publications': [
            'A novel data augmentation using conditional GAN with multi-pseudo label for age estimation from forensic dentistry',
            'Multi-level Multi-scale deep feature encoding for chronological age estimation from OPG images ',
            'Active Learning for Time Series Classification ',
        ],
        
        'References': [
            "Name : Peter Inziano"
            "Phone : +254726113987"
        ]

    }

[<img alt="alt_text"  src="coffee.png" />](https://www.payumoney.com/paybypayumoney/#/147695053B73CAB82672E715A52F9AA5)
