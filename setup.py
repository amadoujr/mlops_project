from setuptools import setup, find_packages

setup(
    name='sentiment_analyzer',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={"":"src"},    
    install_requires=[
        # project dependancies
    ],
    entry_points={
        'console_scripts': [
            'predict=sentiment_analyzer.predict:main',
            'promote=sentiment_analyzer.promote:main',
            'promote=sentiment_analyzer.promote:main'
        ],
    },
)
