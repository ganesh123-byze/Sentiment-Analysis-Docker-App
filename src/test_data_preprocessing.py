from data_preprocessing import TextPreprocessor

processor = TextPreprocessor()

sample = "I really loved this movie! It was amazing and fantastic!!!"

print(processor.clean_text(sample))
