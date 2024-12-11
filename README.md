# Introduction

Automatic audio summarization involves conversion of spoken language into compact and meaningful text under the premise of retaining contextual integrity. The project deals with a system that is particularly designed to process audio files and produce transcriptions with emphasized keywords representing the core elements of their content.
Our Audio Summarizer aims to include the acoustic knowledge in transcript summarization. Initially we explored a multimodal learning approach, where an LSTM-based model was used to generate textual summaries of audio transcriptions while concurrently utilizing a CNN to extract pitch vectors to recognize the delivery and tone of the spoken content. These results from separate models would later be combined to highlight important segments and ensure that both the verbal content and its delivery were considered.
However, due to the complexity of the problem, we pivoted our efforts on using a GradientBoost Classifier to directly predict segments as significant or insignificant.

# Demo

https://www.youtube.com/watch?v=RESxX2fOEBg
