from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def compound_score(text: str) -> float:
    return float(_analyzer.polarity_scores(text)['compound'])
