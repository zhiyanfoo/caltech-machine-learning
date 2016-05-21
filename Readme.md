# Caltech's Machine Learning Course

## What is this?
A collection of answers to [Caltech's Machine Learning Course](https://work.caltech.edu/telecourse.html) by Professor Yaser Abu-Mostafa and, more importantly, the code used to arrive at the said answers. Also included is a script to download the material needed for completing the course. The course material was not included in this repository due to the course's restriction on the redistribution of its material.

## Requirements

>     python3

and many additional python libraries, listed in requirements.txt.

to install said libraries run

```bash
pip3 install -r requirements.txt
```

## What to do

Once you have installed python3 and the additional libraries you should first get the course materials.

```bash
python3 get_course_materials.py
```

To run the experiments, for example for ass1, do
```bash
python3 hw1.py
```

## Answers

My answer to every question is found at the bottom of the hw*.py, in a python dictionary. Below is a random example so as not to spoil anything for anyone.

>     ans = { 
>          1 : 'a',
>          2 : 'cxa',
>          3 : 'd',
>          4 : 'c?d',
>          }

The 'x' indicates I got a non-coding question wrong, followed by the correct answer. The question mark means that the results I got from my code contradicts the given solution, even after a debugging attempt. 

This occured rarely, only once prior to ass8, in ass6 question 4. In ass8 and and ass9, I did not use [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), the support vector machine library the course recommeded, instead choosing the popular [scikit-learn](http://scikit-learn.org/stable/index.html#) library. The different implementations of the learning algorithims might explain the few discrepencies between my results and the given solution, although the discrepencies could just be the result of an error in my part. As such only the code before ass8 should be considered reliable.

## License
MIT license; see LICENSE.