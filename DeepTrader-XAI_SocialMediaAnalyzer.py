import requests
import logging
from transformers import pipeline

class SocialMediaAnalyzer:
    def __init__(self):
        self.nlp = pipeline("sentiment-analysis")
    
    def fetch_tweets(self, query, count=100):
        url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={count}"
        headers = {"Authorization": f"Bearer {os.getenv('TWITTER_BEARER_TOKEN')}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logging.error(f"Error fetching tweets: {e}")
            return []

    def analyze_sentiment(self, tweets):
        sentiments = [self.nlp(tweet["text"][:512])[0] for tweet in tweets]
        return sentiments