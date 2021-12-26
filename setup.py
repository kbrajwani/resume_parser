from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))
setup(
  name = 'resume_parser',         # How you named your package folder (MyLib)
  packages = ['resume_parser'],   # Chose the same as "name"
  version = '0.8.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A resume parser used for extracting information from resumes',   # Give a short description about your library
  author = 'kumar',                   # Type in your name
  author_email = 'kumarrajwani1811@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/kbrajwani/resume_parser',   # Provide either the link to your github or to your website
  long_description=open('README.rst', encoding="utf8").read(),  
  # download_url = 'https://github.com/kbrajwani/resume_parser/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['resume', 'parser', 'cv'],   # Keywords that define your package best
  include_package_data=True,  
  install_requires=[            # I get to this in a second            
            'docx2txt>=0.8',
            'nltk>=3.5',
            'numpy>=1.19.1',
            'pandas>=1.1.0',
            'pdfminer.six>=20200517',            
            'pdfplumber>=0.5.23',
            'phonenumbers>=8.12.7',
            'spacy>=2.3.2',       
            'stemming>=1.0.1',
            'tika>=1.24',            
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  zip_safe=False,
)