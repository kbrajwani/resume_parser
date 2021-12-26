Resume parser
=============
Premium resume parsing services have been moved to [Resume-Parser](https://www.resume-parser.com/application/resumes). Please try the demo for free and give us your [feedback](https://www.resume-parser.com)

Premium resume parsing services have been moved to `Resume-Parser <https://www.resume-parser.com/application/resumes>`__. 
Please try the demo for free and give us your `feedback <https://www.resume-parser.com>`__
::

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

[<img alt="alt_text"  src="coffee.png" />](https://www.payumoney.com/paybypayumoney/#/147695053B73CAB82672E715A52F9AA5)