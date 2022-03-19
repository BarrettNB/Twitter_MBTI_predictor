# Twitter Myers-Briggs Type Indicator Predictor

## Introduction

Katharine Cook Briggs and Isabel Briggs Myers adapted Carl Jung’s ideas on human personality in their 1944 work, the Briggs Myers Type Indicator. They changed its name to the Myers-Briggs Type Indicator (MBTI) in 1956. It theorizes four independent axes of human personality:

<ol>
  <li>Extraversion [<i>sic</i>] vs. Introversion (E vs. I): Whether a person gains emotional energy by being around others vs. being alone.</li>
  <li>Sensing vs. iNtuiting [<i>sic</i>] (S vs. N): Whether a person thinks primarily about concrete things that can be detected by the five senses vs. abstract concepts that cannot be.</li>
  <li>Feeling vs. Thinking (F vs. T): Whether one is generally warm-hearted vs. tough-minded.</li>
  <li>Judging vs. Perceiving (J vs. P): Whether one tends to be decisive vs. open-minded.</li>
</ol>

For example, an ESFJ tends to be extroverted, sensory-oriented, warm-hearted, and decisive.

Many people have used the MBTI to help them gain awareness of themselves and others. The indicator is not without its critics, however, suggesting that one’s results may need to be taken with a grain of salt.

Either way, our task is to attempt to classify Twitter users’ Myers-Briggs type based on their tweets. We use a bag of words method, where we analyze the words contained within each tweet. This model could be useful for advertisers trying to reach a targeted demographic, a particular personality type that might fit the product they’re trying to sell. Marketers might also use this to understand their customers better. For the individual, a web app might be developed to help them understand how their online personality is represented based on their tweet history.

## Documents

[Final report](https://docs.google.com/document/d/1XFLDJz3YmNM6RGq2ZA4BL03vJ7vegTmBUQck0Pnx9eA/edit?usp=sharing)

[Presentation](https://github.com/BarrettNB/Twitter_MBTI_predictor/blob/main/Reports/Twitter_MBTI_presentation.pdf)

## Data Sources

## Notebooks

Data is collected and screened in Part 1. Part 2 runs through exploratory data analysis. From trends learned here, Part 3 trains the data through machine learning algorithms and analyzes and optimizes the results.

[Part 1: Data Collection](https://github.com/BarrettNB/Twitter_MBTI_predictor/blob/main/1_Tweepy_reader.ipynb)

[Part 2: Exploratory Data Analysis](https://github.com/BarrettNB/Twitter_MBTI_predictor/blob/main/2_EDA.ipynb)

[Part 3: Machine Learning and Analysis](https://github.com/BarrettNB/Twitter_MBTI_predictor/blob/main/3_Modeling.ipynb)
